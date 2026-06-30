"""
TCNFusionVQVAE  — TCN encoders per group + cross-modal attention + single VQ.
ModalFusionVQVAE — patch-based VQ-VAE with explicit cross-modal fusion.

Encoder pipeline (per window):
  1. Patch + embed   — split by sensor group, patchify, project to d_model
  2. Intra-modal SA  — 1 TransformerBlock per group (acc / gyr / pressure)
  3. Cross-modal CA  — acc attends to gyr  → acc_from_gyr  (N, T', d)
                     — acc attends to pres → acc_from_pres (N, T', d)
  4. Fusion          — cat(acc_from_gyr, acc_from_pres) → Linear → z_e (N, T', d)
  5. VQ              — single shared codebook

Decoder:
  Per-group linear patch decoder: z_q → patch_size × g_channels per group,
  then reshape back to (N, g_channels, T).  Simple, symmetric, avoids TCN.
  If reconstruction detail is needed, this can be upgraded to a TCN decoder later.

Token temporal support (patch_size = 2^n_downsample at target_sr=50 Hz):
  n_downsample=3 →  8 samples →  160 ms / token
  n_downsample=4 → 16 samples →  320 ms / token
  n_downsample=5 → 32 samples →  640 ms / token
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from src.transformer import TransformerBlock, _precompute_rope, _apply_rotary
from src.vqvae import Encoder, Decoder, HIDDEN_DIM, LATENT_DIM, CODEBOOK_DIM, CODEBOOK_SIZE

SENSOR_GROUPS = [(0, 3), (3, 6), (6, 7)]   # acc, gyr, pressure
GROUP_NAMES   = ["acc", "gyr", "pressure"]


# ── Building blocks ───────────────────────────────────────────────────────────

class GroupPatchEmbed(nn.Module):
    """(B, g_ch, T) → (B, T', d_model) — non-linear CNN patch embedding.

    Two Conv1d layers with GELU run within each patch independently,
    followed by mean-pooling over the patch time axis.  This gives the
    embedding non-linear capacity that a single linear projection lacks,
    breaking the early EMA fixed-point that plagued the linear version.
    """

    def __init__(self, g_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.g_channels = g_channels
        mid = max(d_model // 4, g_channels)
        self.cnn = nn.Sequential(
            nn.Conv1d(g_channels, mid,     kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mid,        d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        # saved so _load_model can reconstruct patch_size without inspecting weight shapes
        self.register_buffer("_patch_size", torch.tensor(patch_size, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, g_ch, T)
        B, C, T = x.shape
        T_trim   = (T // self.patch_size) * self.patch_size
        n_patches = T_trim // self.patch_size
        # view each patch as an independent short sequence
        x = x[:, :, :T_trim].reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B * n_patches, C, self.patch_size)
        x = self.cnn(x).mean(dim=-1)           # (B*n_patches, d_model)
        return self.norm(x.reshape(B, n_patches, -1))   # (B, T', d)


class CrossAttn(nn.Module):
    """
    Pre-LN cross-attention: Q from one modality, KV from another.
    RoPE applied to Q and K (same length T' for both).
    Output = residual update: query + attended context.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        query:   torch.Tensor,          # (B, T', d)
        context: torch.Tensor,          # (B, T', d)
        cos:     torch.Tensor,          # (T', head_dim//2)
        sin:     torch.Tensor,          # (T', head_dim//2)
    ) -> torch.Tensor:                  # (B, T', d)  — residual update
        B, Tq, _ = query.shape
        Tk = context.shape[1]

        q  = self.q_proj(self.norm_q(query))
        kv = self.kv_proj(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)

        def reshape(t, T):
            return t.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = _apply_rotary(reshape(q, Tq), cos[:Tq], sin[:Tq])
        k = _apply_rotary(reshape(k, Tk), cos[:Tk], sin[:Tk])
        v = reshape(v, Tk)

        dp  = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, Tq, -1)
        return query + self.drop(self.out_proj(out))


class GroupPatchDecoder(nn.Module):
    """(B, T', d_model) → (B, g_ch, T)  — patch decoder."""

    def __init__(self, g_channels: int, patch_size: int, d_model: int, deep: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.g_channels = g_channels
        if deep:
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, g_channels * patch_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, g_channels * patch_size),
            )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:   # (B, T', d)
        B, T_prime, _ = z_q.shape
        patches = self.proj(z_q)                              # (B, T', C*P)
        patches = patches.reshape(B, T_prime, self.g_channels, self.patch_size)
        return patches.permute(0, 2, 1, 3).reshape(B, self.g_channels, -1)  # (B, C, T)


# ── Main model ────────────────────────────────────────────────────────────────

class ModalFusionVQVAE(nn.Module):
    """
    Patch-based VQ-VAE with cross-modal fusion before single-codebook quantization.

    Args:
        patch_size    : samples per token (= 2^n_downsample at target_sr=50)
        d_model       : transformer hidden dim
        n_heads       : attention heads
        dim_ff        : feedforward dim in TransformerBlock
        codebook_dim  : VQ projection dim (< d_model is fine)
        codebook_size : number of codebook entries
    """

    def __init__(
        self,
        patch_size:    int   = 8,
        d_model:       int   = 256,
        n_heads:       int   = 4,
        n_layers:      int   = 1,
        dim_ff:        int   = 512,
        codebook_dim:  int   = 64,
        codebook_size: int   = 512,
        decay:         float = 0.8,
        commitment:    float = 1.0,
        dropout:       float = 0.1,
        max_seq_len:   int   = 1024,
        deep_decoder:  bool  = True,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.d_model       = d_model
        self.codebook_size = codebook_size

        # 1. per-group patch embeddings
        self.patch_embeds = nn.ModuleList([
            GroupPatchEmbed(end - start, patch_size, d_model)
            for start, end in SENSOR_GROUPS
        ])

        # 2. intra-modality self-attention (n_layers per group)
        self.intra_attn = nn.ModuleList([
            nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, dropout)
                           for _ in range(n_layers)])
            for _ in SENSOR_GROUPS
        ])

        # 3. cross-modal: acc attends to gyr / pressure
        self.cross_gyr  = CrossAttn(d_model, n_heads, dropout)
        self.cross_pres = CrossAttn(d_model, n_heads, dropout)

        # 4. fusion: cat(acc_from_gyr, acc_from_pres) → d_model
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 5. VQ — single shared codebook
        self.vq = VectorQuantize(
            dim                     = d_model,
            codebook_dim            = codebook_dim,
            codebook_size           = codebook_size,
            decay                   = decay,
            commitment_weight       = commitment,
            kmeans_init             = True,
            threshold_ema_dead_code = 2,
        )

        # 6. per-group patch decoders
        self.decoders = nn.ModuleList([
            GroupPatchDecoder(end - start, patch_size, d_model, deep=deep_decoder)
            for start, end in SENSOR_GROUPS
        ])

        # RoPE buffers (shared across intra + cross attention)
        cos, sin = _precompute_rope(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _rope(self, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rope_cos[:T], self.rope_sin[:T]

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Run encoder through fusion, return (z_e, T_in).
        x: (B, 7, T)
        """
        T_in = x.shape[-1]

        # 1. patch + embed per group
        feats = [
            embed(x[:, start:end, :])
            for embed, (start, end) in zip(self.patch_embeds, SENSOR_GROUPS)
        ]   # list of (B, T', d)

        T_prime = feats[0].shape[1]
        cos, sin = self._rope(T_prime)

        # 2. intra-modality self-attention
        new_feats = []
        for group_blocks, f in zip(self.intra_attn, feats):
            for blk in group_blocks:
                f = blk(f, cos, sin)
            new_feats.append(f)
        feats = new_feats   # acc_f, gyr_f, pres_f

        acc_f, gyr_f, pres_f = feats

        # 3. cross-modal: acc ← gyr,  acc ← pressure
        acc_from_gyr  = self.cross_gyr(acc_f,  gyr_f,  cos, sin)   # (B, T', d)
        acc_from_pres = self.cross_pres(acc_f, pres_f, cos, sin)    # (B, T', d)

        # 4. fusion
        z_e = self.fusion(
            torch.cat([acc_from_gyr, acc_from_pres], dim=-1)
        )   # (B, T', d)

        return z_e, T_in

    def forward(self, x: torch.Tensor):
        """
        x: (B, 7, T)
        Returns: (x_hat, indices, vq_loss, z_e)
          x_hat   : (B, 7, T)
          indices : (B, T')
          vq_loss : scalar
          z_e     : (B, T', d)  — pre-quantization, for smooth loss
        """
        z_e, T_in = self.encode_features(x)

        # 5. VQ
        z_q, indices, vq_loss = self.vq(z_e)   # (B, T', d), (B, T'), scalar

        # 6. decode per group, concatenate channels
        parts = [dec(z_q) for dec in self.decoders]   # (B, g_ch, T') each
        x_hat = torch.cat(parts, dim=1)[..., :T_in]   # (B, 7, T)

        return x_hat, indices, vq_loss, z_e

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 7) or (B, 7, T) → (B, T') token indices."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices


# ── Plain patch VQ-VAE ────────────────────────────────────────────────────────

class FlatPatchVQVAE(nn.Module):
    """All 7 channels fused into one patch → 1 token per time step, single codebook.

    Encoder: GroupPatchEmbed(7) → 1 SA block → VQ
    Decoder: linear patch decoder → (B, 7, T)
    """

    def __init__(
        self,
        patch_size:    int   = 25,
        d_model:       int   = 256,
        n_heads:       int   = 4,
        n_layers:      int   = 3,
        dim_ff:        int   = 512,
        codebook_dim:  int   = 64,
        codebook_size: int   = 512,
        decay:         float = 0.8,
        commitment:    float = 1.0,
        dropout:       float = 0.1,
        max_seq_len:   int   = 1024,
        in_channels:   int   = 7,
        deep_decoder:  bool  = True,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.codebook_size = codebook_size
        self._deep_decoder = deep_decoder

        self.patch_embed = GroupPatchEmbed(in_channels, patch_size, d_model)
        self.sa = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.vq          = VectorQuantize(
            dim                     = d_model,
            codebook_dim            = codebook_dim,
            codebook_size           = codebook_size,
            decay                   = decay,
            commitment_weight       = commitment,
            kmeans_init             = True,
            threshold_ema_dead_code = 2,
        )
        self.decoder = GroupPatchDecoder(in_channels, patch_size, d_model, deep=deep_decoder)

        cos, sin = _precompute_rope(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor):
        """x: (B, 7, T) → (x_hat, indices, vq_loss, z_e)"""
        T_in = x.shape[-1]
        z_e  = self.patch_embed(x)                                # (B, T', d)
        cos, sin = self.rope_cos[:z_e.shape[1]], self.rope_sin[:z_e.shape[1]]
        for blk in self.sa:
            z_e = blk(z_e, cos, sin)
        z_q, indices, vq_loss = self.vq(z_e)
        x_hat = self.decoder(z_q)[..., :T_in]                    # (B, 7, T)
        return x_hat, indices, vq_loss, z_e

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 7) or (B, 7, T) → (B, T') token indices."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices


# ── Grouped patch VQ-VAE ──────────────────────────────────────────────────────

class GroupedPatchVQVAE(nn.Module):
    """acc / gyr / pressure encoded independently — 3 codebooks, no cross-modal.

    Encoder: 3 × (GroupPatchEmbed → SA) → 3 × VQ
    Decoder: 3 × linear patch decoder → cat → (B, 7, T)
    encode() returns (B, 3, T') — one index sequence per group.
    """

    def __init__(
        self,
        patch_size:    int   = 25,
        d_model:       int   = 256,
        n_heads:       int   = 4,
        n_layers:      int   = 3,
        dim_ff:        int   = 512,
        codebook_dim:  int   = 64,
        codebook_size: int   = 512,
        decay:         float = 0.8,
        commitment:    float = 1.0,
        dropout:       float = 0.1,
        max_seq_len:   int   = 1024,
        deep_decoder:  bool  = True,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.codebook_size = codebook_size

        self.patch_embeds = nn.ModuleList([
            GroupPatchEmbed(end - start, patch_size, d_model)
            for start, end in SENSOR_GROUPS
        ])
        # sa_blocks[g] = ModuleList of n_layers blocks for group g
        self.sa_blocks = nn.ModuleList([
            nn.ModuleList([TransformerBlock(d_model, n_heads, dim_ff, dropout)
                           for _ in range(n_layers)])
            for _ in SENSOR_GROUPS
        ])
        self.vqs = nn.ModuleList([
            VectorQuantize(
                dim                     = d_model,
                codebook_dim            = codebook_dim,
                codebook_size           = codebook_size,
                decay                   = decay,
                commitment_weight       = commitment,
                kmeans_init             = True,
                threshold_ema_dead_code = 2,
            )
            for _ in SENSOR_GROUPS
        ])
        self.decoders = nn.ModuleList([
            GroupPatchDecoder(end - start, patch_size, d_model, deep=deep_decoder)
            for start, end in SENSOR_GROUPS
        ])

        cos, sin = _precompute_rope(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor):
        """x: (B, 7, T) → (x_hat, indices, vq_loss, z_e_list)"""
        T_in        = x.shape[-1]
        parts_hat   = []
        all_indices = []
        total_vq    = torch.tensor(0.0, device=x.device)
        all_z_e     = []

        for (start, end), embed, sa, vq, dec in zip(
            SENSOR_GROUPS, self.patch_embeds, self.sa_blocks, self.vqs, self.decoders
        ):
            z_e = embed(x[:, start:end, :])                       # (B, T', d)
            cos, sin = self.rope_cos[:z_e.shape[1]], self.rope_sin[:z_e.shape[1]]
            for blk in sa:
                z_e = blk(z_e, cos, sin)
            z_q, idx, vq_loss = vq(z_e)
            parts_hat.append(dec(z_q)[..., :T_in])
            all_indices.append(idx)
            total_vq = total_vq + vq_loss
            all_z_e.append(z_e)

        x_hat   = torch.cat(parts_hat, dim=1)                     # (B, 7, T)
        indices = torch.stack(all_indices, dim=1)                  # (B, 3, T')
        return x_hat, indices, total_vq / len(SENSOR_GROUPS), all_z_e

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 7) or (B, 7, T) → (B, 3, T') token indices."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices

# ── TCN-based fusion VQ-VAE ───────────────────────────────────────────────────

class TCNFusionVQVAE(nn.Module):
    """
    TCN encoders per sensor group + cross-modal attention fusion + single VQ.

    Encoder pipeline:
      1. Per-group TCN encoder  (acc / gyr / pres) → (N, T', latent)
      2. Cross-modal CA         — acc attends to gyr  → acc_from_gyr
                                — acc attends to pres → acc_from_pres
      3. Fusion                 — cat(acc_from_gyr, acc_from_pres) → Linear → z_e
      4. VQ                     — single shared codebook

    Decoder: per-group TCN decoder receives z_q and reconstructs its channels.
    """

    def __init__(
        self,
        hidden_dim:        int   = HIDDEN_DIM,
        latent_dim:        int   = LATENT_DIM,
        codebook_dim:      int   = CODEBOOK_DIM,
        codebook_size:     int   = CODEBOOK_SIZE,
        decay:             float = 0.8,
        commitment_weight: float = 1.0,
        n_downsample:      int   = 3,
        n_tcn_layers:      int   = 6,
        kernel_size:       int   = 3,
        n_heads:           int   = 4,
        dropout:           float = 0.1,
        max_seq_len:       int   = 1024,
    ):
        super().__init__()
        self.n_downsample  = n_downsample
        self.codebook_size = codebook_size

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for start, end in SENSOR_GROUPS:
            g_ch = end - start
            self.encoders.append(
                Encoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )
            self.decoders.append(
                Decoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )

        self.cross_gyr  = CrossAttn(latent_dim, n_heads, dropout)
        self.cross_pres = CrossAttn(latent_dim, n_heads, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        self.vq = VectorQuantize(
            dim                     = latent_dim,
            codebook_dim            = codebook_dim,
            codebook_size           = codebook_size,
            decay                   = decay,
            commitment_weight       = commitment_weight,
            kmeans_init             = True,
            threshold_ema_dead_code = 2,
        )

        cos, sin = _precompute_rope(latent_dim // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        return F.pad(x, (0, pad)) if pad else x, T_in

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """x: (N, 7, T) → (z_e, T_in)  where z_e is (N, T', latent)."""
        T_in = x.shape[-1]

        enc_feats = []
        for g, (start, end) in enumerate(SENSOR_GROUPS):
            x_g, _ = self._pad(x[:, start:end, :])
            enc_feats.append(self.encoders[g](x_g).transpose(1, 2))   # (N, T', latent)

        acc_f, gyr_f, pres_f = enc_feats
        T_prime = acc_f.shape[1]
        cos = self.rope_cos[:T_prime]
        sin = self.rope_sin[:T_prime]

        acc_from_gyr  = self.cross_gyr(acc_f,  gyr_f,  cos, sin)
        acc_from_pres = self.cross_pres(acc_f, pres_f, cos, sin)

        z_e = self.fusion(torch.cat([acc_from_gyr, acc_from_pres], dim=-1))
        return z_e, T_in

    def forward(self, x: torch.Tensor):
        """x: (N, 7, T) → (x_hat, indices, vq_loss, z_e)"""
        z_e, T_in = self.encode_features(x)

        z_q, indices, vq_loss = self.vq(z_e)   # (N, T', latent)

        z_q_t = z_q.transpose(1, 2)            # (N, latent, T')
        parts = [
            self.decoders[g](z_q_t)[..., :T_in]
            for g, _ in enumerate(SENSOR_GROUPS)
        ]
        x_hat = torch.cat(parts, dim=1)        # (N, 7, T)

        return x_hat, indices, vq_loss, z_e

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, 7) or (N, 7, T) → (N, T') token indices."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices
