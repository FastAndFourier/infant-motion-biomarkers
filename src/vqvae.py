from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from src.transformer import TransformerBlock, _precompute_rope

IN_CHANNELS   = 7
HIDDEN_DIM    = 128
LATENT_DIM    = 256
CODEBOOK_DIM  = 64
CODEBOOK_SIZE = 512


class TCNBlock(nn.Module):
    """Dilated residual block — non-causal, same-length output."""
    def __init__(self, ch: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # symmetric padding → same T
        self.conv  = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.proj  = nn.Conv1d(ch, ch, 1)
        self.norm1 = nn.BatchNorm1d(ch)
        self.norm2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv(x)))
        h = self.norm2(self.proj(h))
        return F.gelu(h + x)


class TCNStack(nn.Module):
    """Stack of TCN blocks with exponentially growing dilation (1, 2, 4, …)."""
    def __init__(self, ch: int, n_layers: int = 6, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            TCNBlock(ch, 2 ** i, kernel_size)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN_DIM, latent=LATENT_DIM,
                 n_downsample=3, n_tcn_layers=6, kernel_size=3):
        super().__init__()
        # 1. project to hidden
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        # 2. TCN — wide receptive field before compression
        self.tcn = TCNStack(hidden, n_tcn_layers, kernel_size)
        # 3. strided downsampling
        downs, ch = [], hidden
        for i in range(n_downsample):
            out = latent if i == n_downsample - 1 else hidden
            downs += [
                nn.Conv1d(ch, out, 4, stride=2, padding=1),
                nn.BatchNorm1d(out), nn.GELU(),
            ]
            ch = out
        self.down = nn.Sequential(*downs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.tcn(self.input_proj(x)))   # (N, latent, T/2^n)


class Decoder(nn.Module):
    def __init__(self, out_channels=IN_CHANNELS, hidden=HIDDEN_DIM, latent=LATENT_DIM,
                 n_downsample=3, n_tcn_layers=6, kernel_size=3):
        super().__init__()
        # 1. strided upsampling
        ups, ch = [], latent
        for _ in range(n_downsample):
            ups += [
                nn.ConvTranspose1d(ch, hidden, 4, stride=2, padding=1),
                nn.BatchNorm1d(hidden), nn.GELU(),
            ]
            ch = hidden
        self.up = nn.Sequential(*ups)
        # 2. TCN — reconstruct detail with wide context
        self.tcn = TCNStack(hidden, n_tcn_layers, kernel_size)
        # 3. output projection
        self.output_proj = nn.Conv1d(hidden, out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.tcn(self.up(x)))   # (N, out_channels, T)


SENSOR_GROUPS = [(0, 3), (3, 6), (6, 7)]   # acc (3ch), gyr (3ch), pressure (1ch)
GROUP_NAMES   = ["acc", "gyr", "pressure"]


# ── Interleaved encoder / decoder blocks ──────────────────────────────────────

class InterleavedBlock(nn.Module):
    """f(x) = stride2( x + TCNStack(x) )  — residual context then compress."""

    def __init__(self, in_ch: int, out_ch: int, n_tcn_layers: int, kernel_size: int):
        super().__init__()
        self.tcn  = TCNStack(in_ch, n_tcn_layers, kernel_size)
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x + self.tcn(x))


class InterleavedUpBlock(nn.Module):
    """f(x) = up(x) + TCNStack( up(x) )  — upsample then residual refinement."""

    def __init__(self, in_ch: int, out_ch: int, n_tcn_layers: int, kernel_size: int):
        super().__init__()
        self.up  = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.tcn = TCNStack(out_ch, n_tcn_layers, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x + self.tcn(x)


class InterleavedEncoder(nn.Module):
    """input_proj → [ TCN residual → stride-2 ] × n_downsample"""

    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN_DIM, latent=LATENT_DIM,
                 n_downsample=3, n_tcn_layers=6, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        blocks, ch = [], hidden
        for i in range(n_downsample):
            out_ch = latent if i == n_downsample - 1 else hidden
            blocks.append(InterleavedBlock(ch, out_ch, n_tcn_layers, kernel_size))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x   # (N, latent, T/2^n)


class InterleavedDecoder(nn.Module):
    """[ upsample → TCN residual ] × n_downsample → output_proj"""

    def __init__(self, out_channels=IN_CHANNELS, hidden=HIDDEN_DIM, latent=LATENT_DIM,
                 n_downsample=3, n_tcn_layers=6, kernel_size=3):
        super().__init__()
        blocks, ch = [], latent
        for _ in range(n_downsample):
            blocks.append(InterleavedUpBlock(ch, hidden, n_tcn_layers, kernel_size))
            ch = hidden
        self.blocks      = nn.ModuleList(blocks)
        self.output_proj = nn.Conv1d(hidden, out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)   # (N, out_channels, T)


class GroupedVQVAE(nn.Module):
    """
    Three independent Encoder→VQ→Decoder paths, one per sensor group.

    Vocab layout in encode() output  (codebook_size = V per group):
      acc   tokens :   0 …   V-1
      gyr   tokens :   V … 2V-1
      pres  tokens :  2V … 3V-1

    encode() returns (N, 3*T') interleaved: [acc_0, gyr_0, pres_0, acc_1, …]
    Total BERT vocab = 3*V  (+ 3 special tokens for PAD/MASK/CLS).
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
    ):
        super().__init__()
        self.n_downsample  = n_downsample
        self.codebook_size = codebook_size
        self.n_groups      = len(SENSOR_GROUPS)

        self.encoders = nn.ModuleList()
        self.vqs      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for start, end in SENSOR_GROUPS:
            g_ch = end - start
            self.encoders.append(
                Encoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )
            self.vqs.append(VectorQuantize(
                dim                     = latent_dim,
                codebook_dim            = codebook_dim,
                codebook_size           = codebook_size,
                decay                   = decay,
                commitment_weight       = commitment_weight,
                kmeans_init             = True,
                threshold_ema_dead_code = 2,
            ))
            self.decoders.append(
                Decoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        return F.pad(x, (0, pad)) if pad else x, T_in

    def forward(self, x: torch.Tensor):
        """x: (N, 7, T)  →  (x_hat, indices_interleaved, vq_loss, z_e_list)"""
        x_hats, indices_list, vq_losses, z_e_list = [], [], [], []

        for g, (start, end) in enumerate(SENSOR_GROUPS):
            x_g, T_in = self._pad(x[:, start:end, :])      # (N, g_ch, T_pad)
            z_e = self.encoders[g](x_g).transpose(1, 2)    # (N, T', latent)
            z_q, idx, vq_loss = self.vqs[g](z_e)           # (N, T', latent), (N, T')
            x_hat_g = self.decoders[g](z_q.transpose(1, 2))[..., :T_in]
            x_hats.append(x_hat_g)
            indices_list.append(idx + g * self.codebook_size)  # vocab offset
            vq_losses.append(vq_loss)
            z_e_list.append(z_e)

        x_hat = torch.cat(x_hats, dim=1)                   # (N, 7, T)
        # interleave: (N, T', 3) → (N, 3*T')
        T_prime = indices_list[0].shape[1]
        indices = torch.stack(indices_list, dim=2).reshape(x.shape[0], -1)

        return x_hat, indices, sum(vq_losses), z_e_list

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, 7) or (N, 7, T) → (N, 3*T') interleaved with vocab offsets."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices   # (N, 3*T')


class InterleavedGroupedVQVAE(nn.Module):
    """
    Same three-group structure as GroupedVQVAE but with interleaved
    TCN-residual encoder/decoder blocks:
        [ x + TCNStack(x) → stride-2 ] × n_downsample  (encoder)
        [ stride-up-2 → x + TCNStack(x) ] × n_downsample  (decoder)

    Vocab layout identical to GroupedVQVAE.
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
    ):
        super().__init__()
        self.n_downsample  = n_downsample
        self.codebook_size = codebook_size
        self.n_groups      = len(SENSOR_GROUPS)

        self.encoders = nn.ModuleList()
        self.vqs      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for start, end in SENSOR_GROUPS:
            g_ch = end - start
            self.encoders.append(
                InterleavedEncoder(g_ch, hidden_dim, latent_dim,
                                   n_downsample, n_tcn_layers, kernel_size)
            )
            self.vqs.append(VectorQuantize(
                dim                     = latent_dim,
                codebook_dim            = codebook_dim,
                codebook_size           = codebook_size,
                decay                   = decay,
                commitment_weight       = commitment_weight,
                kmeans_init             = True,
                threshold_ema_dead_code = 2,
            ))
            self.decoders.append(
                InterleavedDecoder(g_ch, hidden_dim, latent_dim,
                                   n_downsample, n_tcn_layers, kernel_size)
            )

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        return F.pad(x, (0, pad)) if pad else x, T_in

    def forward(self, x: torch.Tensor):
        """x: (N, 7, T)  →  (x_hat, indices_interleaved, vq_loss, z_e_list)"""
        x_hats, indices_list, vq_losses, z_e_list = [], [], [], []

        for g, (start, end) in enumerate(SENSOR_GROUPS):
            x_g, T_in = self._pad(x[:, start:end, :])
            z_e = self.encoders[g](x_g).transpose(1, 2)    # (N, T', latent)
            z_q, idx, vq_loss = self.vqs[g](z_e)
            x_hat_g = self.decoders[g](z_q.transpose(1, 2))[..., :T_in]
            x_hats.append(x_hat_g)
            indices_list.append(idx + g * self.codebook_size)
            vq_losses.append(vq_loss)
            z_e_list.append(z_e)

        x_hat   = torch.cat(x_hats, dim=1)
        indices = torch.stack(indices_list, dim=2).reshape(x.shape[0], -1)
        return x_hat, indices, sum(vq_losses), z_e_list

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, 7) or (N, 7, T) → (N, 3*T') interleaved with vocab offsets."""
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels:       int   = IN_CHANNELS,
        hidden_dim:        int   = HIDDEN_DIM,
        latent_dim:        int   = LATENT_DIM,
        codebook_dim:      int   = CODEBOOK_DIM,
        codebook_size:     int   = CODEBOOK_SIZE,
        decay:             float = 0.8,
        commitment_weight: float = 1.0,
        n_downsample:      int   = 3,
        n_tcn_layers:      int   = 6,
        kernel_size:       int   = 3,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.n_downsample = n_downsample
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim,
                               n_downsample, n_tcn_layers, kernel_size)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim,
                               n_downsample, n_tcn_layers, kernel_size)
        self.vq = VectorQuantize(
            dim                     = latent_dim,
            codebook_dim            = codebook_dim,
            codebook_size           = codebook_size,
            decay                   = decay,
            commitment_weight       = commitment_weight,
            kmeans_init             = True,
            threshold_ema_dead_code = 2,
        )

    def forward(self, x: torch.Tensor):
        # x: (N, C, T)
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        if pad:
            x = F.pad(x, (0, pad))

        z_e = self.encoder(x)                   # (N, latent, T')
        z_e = z_e.transpose(1, 2)               # (N, T', latent)
        z_q, indices, vq_loss = self.vq(z_e)    # (N, T', latent), (N, T'), scalar
        z_q = z_q.transpose(1, 2)               # (N, latent, T')
        x_hat = self.decoder(z_q)[..., :T_in]   # (N, C, T)
        return x_hat, indices, vq_loss, z_e      # z_e (N, T', latent) for smooth loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        if pad:
            x = F.pad(x, (0, pad))
        z_e = self.encoder(x).transpose(1, 2)
        _, indices, _ = self.vq(z_e)
        return indices  # (N, T')


class GroupTimestepAttn(nn.Module):
    """Symmetric self-attention over the 3 sensor groups at each compressed timestep.

    At each time step t in the compressed sequence, the 3 group latents form a
    length-3 sequence. Self-attention over that modality dimension lets every
    group attend to every other group symmetrically. Modality embeddings replace
    RoPE because the 3 groups have no natural ordering.

    Input:  list of 3 tensors, each (N, T', D)
    Output: list of 3 tensors, each (N, T', D), cross-modal-aware
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.mod_embed = nn.Embedding(3, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, z_list: list) -> list:
        N, Tp, D = z_list[0].shape
        ids = torch.arange(3, device=z_list[0].device)

        z = torch.stack(z_list, dim=2) + self.mod_embed(ids)   # (N, T', 3, D)
        x = z.reshape(N * Tp, 3, D)

        xn = self.norm1(x)
        h, _ = self.attn(xn, xn, xn)
        x = x + h
        x = x + self.ffn(self.norm2(x))

        x = x.reshape(N, Tp, 3, D)
        return [x[:, :, g, :] for g in range(3)]


class CrossModalGroupedVQVAE(nn.Module):
    """GroupedVQVAE with symmetric per-timestep cross-modal attention before VQ.

    Pipeline: per-group encode → GroupTimestepAttn → per-group VQ → per-group decode.
    Vocab layout identical to GroupedVQVAE (offsets: 0, V, 2V). encode() output
    and BERT downstream are unchanged.
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
        n_attn_heads:      int   = 4,
    ):
        super().__init__()
        self.n_downsample  = n_downsample
        self.codebook_size = codebook_size
        self.n_groups      = len(SENSOR_GROUPS)

        self.encoders = nn.ModuleList()
        self.vqs      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for start, end in SENSOR_GROUPS:
            g_ch = end - start
            self.encoders.append(
                Encoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )
            self.vqs.append(VectorQuantize(
                dim                     = latent_dim,
                codebook_dim            = codebook_dim,
                codebook_size           = codebook_size,
                decay                   = decay,
                commitment_weight       = commitment_weight,
                kmeans_init             = True,
                threshold_ema_dead_code = 2,
            ))
            self.decoders.append(
                Decoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )

        self.cross_modal_attn = GroupTimestepAttn(latent_dim, n_attn_heads)

    def _pad(self, x: torch.Tensor) -> tuple:
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        return F.pad(x, (0, pad)) if pad else x, T_in

    def forward(self, x: torch.Tensor):
        """x: (N, 7, T) → (x_hat, indices_interleaved, vq_loss, z_e_prime_list)"""
        z_e_list, T_ins = [], []
        for g, (start, end) in enumerate(SENSOR_GROUPS):
            x_g, T_in = self._pad(x[:, start:end, :])
            z_e = self.encoders[g](x_g).transpose(1, 2)    # (N, T', latent)
            z_e_list.append(z_e)
            T_ins.append(T_in)

        z_e_prime = self.cross_modal_attn(z_e_list)         # list of (N, T', latent)

        x_hats, indices_list, vq_losses = [], [], []
        for g, (start, end) in enumerate(SENSOR_GROUPS):
            z_q, idx, vq_loss = self.vqs[g](z_e_prime[g])
            x_hat_g = self.decoders[g](z_q.transpose(1, 2))[..., :T_ins[g]]
            x_hats.append(x_hat_g)
            indices_list.append(idx + g * self.codebook_size)
            vq_losses.append(vq_loss)

        x_hat   = torch.cat(x_hats, dim=1)
        indices = torch.stack(indices_list, dim=2).reshape(x.shape[0], -1)
        return x_hat, indices, sum(vq_losses), z_e_prime

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices


class CrossGroupVQVAE(nn.Module):
    """Per-group TCN encode → interleave → shallow SA transformer → split → per-group VQ → decode.

    All 3 groups attend to each other across both time and modality.
    Group embeddings distinguish acc/gyr/pressure within the same timestep.
    RoPE encodes temporal position (same position for all groups at time t).
    Vocab layout identical to GroupedVQVAE (offsets 0, V, 2V).
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
        n_fusion_layers:   int   = 1,
        n_attn_heads:      int   = 4,
        max_seq_len:       int   = 512,
    ):
        super().__init__()
        self.n_downsample  = n_downsample
        self.codebook_size = codebook_size
        self.n_groups      = len(SENSOR_GROUPS)

        self.encoders = nn.ModuleList()
        self.vqs      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for start, end in SENSOR_GROUPS:
            g_ch = end - start
            self.encoders.append(
                Encoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )
            self.vqs.append(VectorQuantize(
                dim                     = latent_dim,
                codebook_dim            = codebook_dim,
                codebook_size           = codebook_size,
                decay                   = decay,
                commitment_weight       = commitment_weight,
                kmeans_init             = True,
                threshold_ema_dead_code = 2,
            ))
            self.decoders.append(
                Decoder(g_ch, hidden_dim, latent_dim, n_downsample, n_tcn_layers, kernel_size)
            )

        self.group_embed = nn.Embedding(3, latent_dim)
        head_dim = latent_dim // n_attn_heads
        self.fusion = nn.ModuleList([
            TransformerBlock(latent_dim, n_attn_heads, latent_dim * 2)
            for _ in range(n_fusion_layers)
        ])
        cos, sin = _precompute_rope(head_dim, max_seq_len)
        self.register_buffer("_rope_cos", cos)   # (max_seq_len, head_dim//2)
        self.register_buffer("_rope_sin", sin)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T_in   = x.shape[-1]
        factor = 2 ** self.n_downsample
        pad    = (factor - T_in % factor) % factor
        return F.pad(x, (0, pad)) if pad else x, T_in

    def forward(self, x: torch.Tensor):
        """x: (N, 7, T) → (x_hat, indices_interleaved, vq_loss, z_fused_list)"""
        z_e_list, T_ins = [], []
        for g, (start, end) in enumerate(SENSOR_GROUPS):
            x_g, T_in = self._pad(x[:, start:end, :])
            z_e = self.encoders[g](x_g).transpose(1, 2)   # (N, T', D)
            z_e_list.append(z_e)
            T_ins.append(T_in)

        N, Tp, D = z_e_list[0].shape
        ids = torch.arange(3, device=x.device)

        # add group embeddings, interleave: [acc_t, gyr_t, pres_t, acc_{t+1}, ...]
        z = torch.stack([
            z_e_list[g] + self.group_embed(ids[g])
            for g in range(3)
        ], dim=2).reshape(N, Tp * 3, D)                    # (N, T', 3, D) → (N, 3T', D)

        # shared temporal RoPE: same position t for acc_t, gyr_t, pres_t
        cos = self._rope_cos[:Tp].repeat_interleave(3, dim=0)   # (3T', head_dim//2)
        sin = self._rope_sin[:Tp].repeat_interleave(3, dim=0)

        for blk in self.fusion:
            z = blk(z, cos, sin)

        # split back → per-group latents
        z_fused_list = [z.reshape(N, Tp, 3, D)[:, :, g, :] for g in range(3)]

        x_hats, indices_list, vq_losses = [], [], []
        for g in range(3):
            z_q, idx, vq_loss = self.vqs[g](z_fused_list[g])
            x_hat_g = self.decoders[g](z_q.transpose(1, 2))[..., :T_ins[g]]
            x_hats.append(x_hat_g)
            indices_list.append(idx + g * self.codebook_size)
            vq_losses.append(vq_loss)

        x_hat   = torch.cat(x_hats, dim=1)
        indices = torch.stack(indices_list, dim=2).reshape(N, -1)
        return x_hat, indices, sum(vq_losses), z_fused_list

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 7:
            x = x.permute(0, 2, 1)
        _, indices, _, _ = self.forward(x)
        return indices


# ── Model-agnostic helpers ────────────────────────────────────────────────────

def _is_grouped(model: nn.Module) -> bool:
    return isinstance(model, (GroupedVQVAE, InterleavedGroupedVQVAE,
                              CrossModalGroupedVQVAE, CrossGroupVQVAE))


def n_codes_from_model(model: nn.Module, base_codebook_size: int) -> int:
    """Total vocabulary size: 3× for grouped models, 1× for standard / fused."""
    return base_codebook_size * (3 if _is_grouped(model) else 1)


def get_codebook(model: nn.Module) -> np.ndarray:
    """(V, codebook_dim) float32 array — works for standard, grouped, and fused."""
    if _is_grouped(model):
        return torch.cat([
            vq._codebook.embed.squeeze(0) for vq in model.vqs
        ], dim=0).detach().cpu().numpy()
    return model.vq._codebook.embed.squeeze(0).detach().cpu().numpy()
