import torch
import torch.nn as nn
import torch.nn.functional as F


def _precompute_rope(head_dim: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   (B, nhead, T, head_dim)
    # cos: (T, head_dim // 2)
    x1  = x[..., ::2]
    x2  = x[..., 1::2]
    cos = cos[None, None]
    sin = sin[None, None]
    out = torch.empty_like(x)
    out[..., ::2]  = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ── Modality vocabulary ───────────────────────────────────────────────────────

# Semantic IDs are stable across datasets — index by name, not column position.
# Datasets declare which subset they provide; absent channels get missing_token.
MODALITY_VOCAB: dict[str, int] = {
    "acc_x": 0, "acc_y": 1, "acc_z": 2,
    "gyr_x": 3, "gyr_y": 4, "gyr_z": 5,
    "pressure": 6,
}
MAX_MODALITIES = 16   # reserve space for future sensors without breaking checkpoints

# Default: all 7 channels present (our dataset)
DEFAULT_MODALITY_IDS   = list(range(7))
DEFAULT_PRESENT_MASK   = [True] * 7   # converted to tensor in forward

# Sensor groups for grouped patch embedding (indices into MODALITY_VOCAB)
SENSOR_GROUPS: list[list[int]] = [[0, 1, 2], [3, 4, 5], [6]]   # acc / gyr / pressure


# ── Patch embeddings ──────────────────────────────────────────────────────────

class PatchTCN(nn.Module):
    """Small CNN encoder for one patch.

    in_channels=1 (default, per-channel mode): input (N, patch_size).
    in_channels>1 (grouped mode): input (N, in_channels, patch_size).
    Output: (N, d_model) in both cases.
    """
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        hidden = d_model // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden,   kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, dilation=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)             # (N, patch_size) → (N, 1, patch_size)
        return self.net(x).squeeze(-1)     # (N, d_model)


class PatchEmbed1d(nn.Module):
    """Standard patch embedding — fuses all channels into one token.
    Input:  (B, T, in_channels)
    Output: (B, T // patch_size, d_model)
    """
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 7):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, **_) -> tuple[torch.Tensor, int]:
        tokens  = self.proj(x.transpose(1, 2)).transpose(1, 2)  # (B, T', d_model)
        return tokens, tokens.shape[1]


class ChannelPatchEmbed1d(nn.Module):
    """Channel-independent patch embedding with semantic modality embeddings.

    Each channel is patched and projected separately. A learnable modality
    embedding (indexed by MODALITY_VOCAB ID, not column position) distinguishes
    sensor types across datasets. Absent channels are replaced by a shared
    learned missing_token so the transformer sees a signal for every slot.

    Args:
        present_mask: (C,) bool — True for channels present in this batch.
                      None → all channels present (default, backward compat).
        modality_ids: (C,) int — MODALITY_VOCAB index for each column of x.
                      None → arange(C) (default, backward compat).

    Input:  (B, T, C)
    Output: (B, T' * C, d_model), T'
    """
    def __init__(self, patch_size: int, d_model: int, num_channels: int = 7,
                 use_tcn: bool = False):
        super().__init__()
        self.patch_size   = patch_size
        self.num_channels = num_channels
        self.use_tcn      = use_tcn
        self.proj         = PatchTCN(patch_size, d_model) if use_tcn else nn.Linear(patch_size, d_model)
        self.modal_embed  = nn.Embedding(MAX_MODALITIES, d_model)
        self.missing_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.missing_token, std=0.02)
        # stored as buffer so _infer_enc_arch can recover patch_size from TCN checkpoints
        self.register_buffer("patch_size_buf", torch.tensor(patch_size))

    def forward(
        self,
        x:            torch.Tensor,
        present_mask: torch.Tensor | None = None,   # (C,) bool
        modality_ids: torch.Tensor | None = None,   # (C,) int
    ) -> tuple[torch.Tensor, int]:
        B, T, C = x.shape
        T_prime = T // self.patch_size

        # Patchify: (B, T'*P, C) → (B*C, T', P)
        xp = x[:, :T_prime * self.patch_size, :]
        xp = xp.reshape(B, T_prime, self.patch_size, C)
        xp = xp.permute(0, 3, 1, 2).reshape(B * C, T_prime, self.patch_size)

        if self.use_tcn:
            # Each patch processed independently: (B*C*T', P) → (B*C*T', d_model)
            tokens = self.proj(xp.reshape(B * C * T_prime, self.patch_size))
            tokens = tokens.reshape(B * C, T_prime, -1)
        else:
            tokens = self.proj(xp)                                          # (B*C, T', d_model)

        # Semantic modality embedding
        if modality_ids is None:
            modality_ids = torch.arange(C, device=x.device)
        ch_idx = modality_ids.unsqueeze(0).expand(B, -1).reshape(B * C)
        tokens = tokens + self.modal_embed(ch_idx).unsqueeze(1)             # (B*C, T', d_model)

        # Reshape to (B, T', C, d_model) for per-channel operations
        tokens = tokens.reshape(B, C, T_prime, -1).permute(0, 2, 1, 3)     # (B, T', C, d_model)

        # Replace absent channels with missing_token
        if present_mask is not None:
            absent = ~present_mask                                          # (C,)
            tokens[:, :, absent, :] = self.missing_token

        tokens = tokens.reshape(B, T_prime * C, -1)                        # (B, T'*C, d_model)
        return tokens, T_prime


class GroupedPatchEmbed1d(nn.Module):
    """Sensor-group patch embedding: acc / gyr / pressure each get one token per time patch.

    Groups channels by sensor type before projection, so cross-channel correlation
    within a group is captured in the embed rather than requiring attention.

    groups: channel indices per group, defaults to SENSOR_GROUPS (acc/gyr/pressure).
    use_tcn: if True, uses a dilated CNN per group; otherwise linear projection.
    present_mask passed to forward is channel-level (C,); absent groups (all channels
    in a group absent) are replaced with missing_token.

    Input:  (B, T, C)
    Output: (B, T' * n_groups, d_model),  T'
    """
    def __init__(self, patch_size: int, d_model: int,
                 groups: list[list[int]] | None = None,
                 use_tcn: bool = False):
        super().__init__()
        if groups is None:
            groups = SENSOR_GROUPS
        self.patch_size = patch_size
        self.groups     = groups
        self.use_tcn    = use_tcn
        n_groups        = len(groups)

        if use_tcn:
            self.projs = nn.ModuleList([
                PatchTCN(patch_size, d_model, in_channels=len(g)) for g in groups
            ])
        else:
            self.projs = nn.ModuleList([
                nn.Linear(len(g) * patch_size, d_model) for g in groups
            ])

        self.modal_embed   = nn.Embedding(n_groups, d_model)
        self.missing_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.missing_token, std=0.02)
        self.register_buffer("patch_size_buf", torch.tensor(patch_size))

    def forward(
        self,
        x:            torch.Tensor,
        present_mask: torch.Tensor | None = None,   # (C,) bool, channel-level
        **_,
    ) -> tuple[torch.Tensor, int]:
        B, T, C = x.shape
        T_prime  = T // self.patch_size
        n_groups = len(self.groups)

        group_tokens = []
        for gi, g in enumerate(self.groups):
            n_ch = len(g)
            xg = x[:, :T_prime * self.patch_size, :][:, :, g]      # (B, T'*P, n_ch)
            xg = xg.reshape(B, T_prime, self.patch_size, n_ch)

            if self.use_tcn:
                # (B*T', n_ch, P) → PatchTCN → (B*T', d_model)
                xg  = xg.permute(0, 1, 3, 2).reshape(B * T_prime, n_ch, self.patch_size)
                tok = self.projs[gi](xg)                            # (B*T', d_model)
                tok = tok.reshape(B, T_prime, -1)                   # (B, T', d_model)
            else:
                xg  = xg.reshape(B, T_prime, n_ch * self.patch_size)
                tok = self.projs[gi](xg)                            # (B, T', d_model)

            tok = tok + self.modal_embed(
                torch.tensor(gi, device=x.device)
            )
            group_tokens.append(tok)                                # (B, T', d_model)

        # (n_groups, B, T', d_model) → (B, T', n_groups, d_model)
        tokens = torch.stack(group_tokens, dim=0).permute(1, 2, 0, 3)

        if present_mask is not None:
            # group absent if ALL its channels are absent
            group_present = torch.stack([
                present_mask[torch.tensor(g, device=x.device)].all()
                for g in self.groups
            ])                                                      # (n_groups,) bool
            tokens[:, :, ~group_present, :] = self.missing_token

        return tokens.reshape(B, T_prime * n_groups, -1), T_prime


# ── Adaptive LayerNorm (DiT-style) ────────────────────────────────────────────

class AdaptiveLN(nn.Module):
    """
    DiT-style adaptive LayerNorm: y = (1 + gamma(c)) * LayerNorm(x) + beta(c)

    gamma and beta are linear projections of the conditioning vector c.
    Projection weights initialised to zero so the block is identical to a
    standard pre-LN transformer at initialisation — safe drop-in replacement.

    Args:
        d_model:  token dimension
        cond_dim: conditioning vector dimension
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)   c: (B, cond_dim)
        gamma, beta = self.proj(c).unsqueeze(1).chunk(2, dim=-1)   # (B, 1, d_model) each
        return (1.0 + gamma) * self.norm(x) + beta


# ── Transformer block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with RoPE and FlashAttention.

    Args:
        cond_dim: if > 0, replaces LayerNorm with AdaptiveLN conditioned on a
                  vector c passed to forward(). Zero preserves standard behaviour.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1,
                 cond_dim: int = 0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.cond_dim = cond_dim

        self.norm1    = AdaptiveLN(d_model, cond_dim) if cond_dim > 0 else nn.LayerNorm(d_model)
        self.norm2    = AdaptiveLN(d_model, cond_dim) if cond_dim > 0 else nn.LayerNorm(d_model)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                c: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        h   = self.norm1(x, c) if self.cond_dim > 0 else self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        dp  = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, T, -1)
        x   = x + self.drop(self.out_proj(out))
        h2  = self.norm2(x, c) if self.cond_dim > 0 else self.norm2(x)
        x   = x + self.drop(self.ff(h2))
        return x


class TransformerBlockSWA(nn.Module):
    """Pre-LayerNorm transformer block with RoPE and sliding window attention.

    Args:
        cond_dim: if > 0, replaces LayerNorm with AdaptiveLN (see TransformerBlock).
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1,
                 window_size: int = None, cond_dim: int = 0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead       = nhead
        self.head_dim    = d_model // nhead
        self.window_size = window_size
        self.cond_dim    = cond_dim

        self.norm1    = AdaptiveLN(d_model, cond_dim) if cond_dim > 0 else nn.LayerNorm(d_model)
        self.norm2    = AdaptiveLN(d_model, cond_dim) if cond_dim > 0 else nn.LayerNorm(d_model)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _sliding_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns additive attention mask (0 = attend, -inf = ignore)."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        mask = (j - i).abs() > self.window_size
        return mask.float().masked_fill(mask, float('-inf'))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                c: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        h   = self.norm1(x, c) if self.cond_dim > 0 else self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        dp   = self.drop.p if self.training else 0.0
        mask = self._sliding_window_mask(T, x.device) if self.window_size is not None else None
        out  = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)

        out = out.transpose(1, 2).reshape(B, T, -1)
        x   = x + self.drop(self.out_proj(out))
        h2  = self.norm2(x, c) if self.cond_dim > 0 else self.norm2(x)
        x   = x + self.drop(self.ff(h2))
        return x


class GroupedCrossmodalPatchEmbed1d(GroupedPatchEmbed1d):
    """GroupedPatchEmbed1d + cross-group self-attention after per-group encoding.

    After each group (acc / gyr / pressure) is embedded independently, a small
    Transformer block lets the three group tokens at each time step attend to each
    other.  This bakes cross-sensor information into the tokens before the main
    encoder Transformer.

    n_cross_layers: number of cross-group attention blocks (default 1).
    n_cross_heads:  attention heads for those blocks (must divide d_model).

    Input/Output: same shapes as GroupedPatchEmbed1d.
    """
    def __init__(self, patch_size: int, d_model: int,
                 groups: list[list[int]] | None = None,
                 use_tcn: bool = False,
                 n_cross_layers: int = 1,
                 n_cross_heads: int = 4):
        super().__init__(patch_size, d_model, groups, use_tcn)
        n_groups = len(self.groups)
        head_dim = d_model // n_cross_heads
        self.cross_layers = nn.ModuleList([
            TransformerBlock(d_model, n_cross_heads, d_model * 2, dropout=0.1)
            for _ in range(n_cross_layers)
        ])
        cos, sin = _precompute_rope(head_dim, n_groups)
        self.register_buffer("cross_cos", cos)   # (n_groups, head_dim//2)
        self.register_buffer("cross_sin", sin)

    def forward(self, x: torch.Tensor,
                present_mask: torch.Tensor | None = None, **_) -> tuple[torch.Tensor, int]:
        tokens, T_prime = super().forward(x, present_mask)   # (B, T'*G, d)
        B, _, d = tokens.shape
        G = len(self.groups)

        # cross-group attention per time step: reshape to (B*T', G, d)
        t = tokens.reshape(B, T_prime, G, d).reshape(B * T_prime, G, d)
        for layer in self.cross_layers:
            t = layer(t, self.cross_cos, self.cross_sin)
        return t.reshape(B, T_prime * G, d), T_prime


# ── Encoder ───────────────────────────────────────────────────────────────────

class IMUTransformerEncoder(nn.Module):
    """
    Patch-based Transformer encoder for multivariate IMU sequences.

    channel_independent=True  (default): each sensor channel gets its own
        token stream (ChannelPatchEmbed1d). Cross-channel interaction happens
        via self-attention. Channel tokens are mean-pooled before output.

        Supports cross-dataset pretraining via present_mask / modality_ids:
          present_mask: (C,) bool — absent channels replaced by missing_token,
                        excluded from the pooling average.
          modality_ids: (C,) int — MODALITY_VOCAB index per column, so "acc_x"
                        carries the same embedding regardless of column position.

    channel_independent=False: all channels fused into one token (PatchEmbed1d).
        Legacy mode for loading old checkpoints.

    Output: (B, T', d_model)  where T' = T // patch_size
    """

    def __init__(
        self,
        patch_size:          int              = 20,
        d_model:             int              = 256,
        nhead:               int              = 8,
        num_layers:          int              = 8,
        dim_ff:              int              = 1024,
        dropout:             float            = 0.1,
        max_seq_len:         int              = 2048,
        channel_independent: bool             = True,
        num_channels:        int              = 7,
        use_swa:             bool             = False,
        swa_window:          int              = 10,
        cond_dim:            int              = 0,
        cond_layers:         list[int] | str  = "all",
        instance_norm:       bool             = False,
        use_tcn:             bool             = False,
        patch_mode:          str              = "channel",
    ):
        super().__init__()
        self.channel_independent = channel_independent
        self.num_channels        = num_channels
        self.instance_norm       = instance_norm
        self.d_model             = d_model
        self.cond_dim            = cond_dim
        self.patch_mode          = patch_mode
        self._cond_set  = (
            set(range(num_layers)) if cond_layers == "all"
            else set(cond_layers)
        )

        if patch_mode == "grouped":
            self.patch_embed = GroupedPatchEmbed1d(patch_size, d_model,
                                                   SENSOR_GROUPS, use_tcn=use_tcn)
            self._n_groups   = len(SENSOR_GROUPS)          # 3
        elif patch_mode == "crossmodal":
            self.patch_embed = GroupedCrossmodalPatchEmbed1d(patch_size, d_model,
                                                             SENSOR_GROUPS, use_tcn=use_tcn)
            self._n_groups   = len(SENSOR_GROUPS)          # 3
        elif patch_mode == "plain" or not channel_independent:
            self.patch_embed = PatchEmbed1d(patch_size, d_model, num_channels)
            self._n_groups   = 1
        else:  # "channel" (default)
            self.patch_embed = ChannelPatchEmbed1d(patch_size, d_model, num_channels,
                                                   use_tcn=use_tcn)
            self._n_groups   = num_channels                # 7

        def _make_block(i: int):
            cd = cond_dim if i in self._cond_set else 0
            if use_swa:
                return TransformerBlockSWA(d_model, nhead, dim_ff, dropout, swa_window, cond_dim=cd)
            return TransformerBlock(d_model, nhead, dim_ff, dropout, cond_dim=cd)

        self.layers = nn.ModuleList([_make_block(i) for i in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

        cos, sin = _precompute_rope(d_model // nhead, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _time_positions(self, T_prime: int, device) -> torch.Tensor:
        """RoPE position indices: same time index repeated _n_groups times (time-first)."""
        return torch.arange(T_prime, device=device).unsqueeze(1).expand(-1, self._n_groups).reshape(-1)

    def embed(
        self,
        x:            torch.Tensor,
        present_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        """Embed x → (tokens, T_prime, positions)."""
        if self.instance_norm:
            # Normalise each channel independently across the time dimension.
            # Removes per-window sensor drift / DC offset; improves cross-infant transfer.
            mean = x.mean(dim=1, keepdim=True)          # (B, 1, C)
            std  = x.std(dim=1, keepdim=True) + 1e-8    # (B, 1, C)
            x    = (x - mean) / std

        if self.patch_mode in ("grouped", "crossmodal"):
            tokens, T_prime = self.patch_embed(x, present_mask)
            positions = self._time_positions(T_prime, x.device)
        elif self.patch_mode == "plain" or not self.channel_independent:
            tokens, T_prime = self.patch_embed(x)
            positions = None
        else:  # "channel"
            tokens, T_prime = self.patch_embed(x, present_mask, modality_ids)
            positions = self._time_positions(T_prime, x.device)
        return tokens, T_prime, positions

    def forward_tokens(self, tokens: torch.Tensor, positions: torch.Tensor | None = None,
                       c: torch.Tensor | None = None) -> torch.Tensor:
        """Process pre-embedded tokens through transformer layers."""
        if positions is None:
            cos = self.rope_cos[:tokens.shape[1]]
            sin = self.rope_sin[:tokens.shape[1]]
        else:
            cos = self.rope_cos[positions]
            sin = self.rope_sin[positions]
        for i, layer in enumerate(self.layers):
            layer_c = c if (self.cond_dim > 0 and i in self._cond_set) else None
            tokens = layer(tokens, cos, sin, c=layer_c)
        return self.norm(tokens)

    def pool_channels(
        self,
        encoded:      torch.Tensor,
        T_prime:      int,
        present_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, T'*_n_groups, d_model) → mean over present groups → (B, T', d_model)."""
        B, _, d = encoded.shape
        tokens = encoded.reshape(B, T_prime, self._n_groups, d)             # (B, T', G, d)
        if present_mask is None:
            return tokens.mean(dim=2)
        if self.patch_mode in ("grouped", "crossmodal"):
            # convert channel-level mask → group-level: group present iff all its channels present
            group_mask = torch.stack([
                present_mask[torch.tensor(g, device=encoded.device)].all()
                for g in SENSOR_GROUPS
            ])                                                               # (n_groups,) bool
        else:
            group_mask = present_mask                                        # (C,) bool
        mask = group_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)   # (1, 1, G, 1)
        n    = group_mask.float().sum().clamp(min=1)
        return (tokens * mask).sum(dim=2) / n

    def forward(
        self,
        x:            torch.Tensor,
        present_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        c:            torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, T, C)
            c:            (B, cond_dim) conditioning vector for AdaptiveLN layers; ignored when cond_dim=0
        Returns:
            (B, T', d_model)
        """
        tokens, T_prime, positions = self.embed(x, present_mask, modality_ids)
        encoded = self.forward_tokens(tokens, positions, c=c)
        if self.patch_mode in ("grouped", "crossmodal") or \
                (self.patch_mode == "channel" and self.channel_independent):
            return self.pool_channels(encoded, T_prime, present_mask)
        return encoded


class HARTransformer(nn.Module):

    def __init__(
        self, 
        input_dim:int=7, 
        d_model:int=256, 
        dim_ff:int=1024,
         
    ):
        super().__init__()