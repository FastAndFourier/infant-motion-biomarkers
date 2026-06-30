"""
VQ-VAE training, reconstruction visualization, and codebook analysis.

Prerequisites
-------------
This script requires two sliding-window dataset classes that are NOT included
in this repository because they depend on your local action-label annotation
format. You must implement them in src/dataset.py:

  class UnlabeledDataset(Dataset):
      '''Sliding-window dataset from OUTPUT_UNLABELED_DIR (no frame labels).'''
      def __init__(self, win: float, hop: float, norm: str = "infant"): ...
      # Each __getitem__ returns (x, info, metadata, y) where
      #   x        : (T, 7) float32  — acc (3) + gyr (3) + pressure (1) at 100 Hz
      #   y        : (T,)   int64    — all -100 (ignored by loss)
      #   info     : dict with keys "infant", "session", "outcome"
      #   metadata : (D,)   float32  — optional session-level features

  class LabeledDataset(Dataset):
      '''Sliding-window dataset from OUTPUT_ALIGN_DIR with per-frame action labels.'''
      def __init__(self, win: float, hop: float, norm: str = "infant",
                   label_scheme: str = "full"): ...

Usage
-----
  # Train the grouped VQ-VAE (one codebook per sensor group: acc / gyr / pres)
  uv run python train_vqvae.py train --grouped --ckpt models/vqvae_grouped.pt

  # Analyze codebook usage and reconstruction quality
  uv run python train_vqvae.py analyze --ckpt models/vqvae_grouped.pt

  # Visualize original vs. reconstructed signal
  uv run python train_vqvae.py reconstruct --ckpt models/vqvae_grouped.pt --grouped
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from src.dataset import UnlabeledDataset, LabeledDataset
from src.vqvae import VQVAE, GroupedVQVAE, InterleavedGroupedVQVAE, CrossModalGroupedVQVAE, CrossGroupVQVAE, IN_CHANNELS
from src.fused_vqvae import ModalFusionVQVAE, TCNFusionVQVAE, FlatPatchVQVAE, GroupedPatchVQVAE
from src.utils import print_param_count


def run_epoch(model, loader, device, optimizer=None, decimation=1, smooth_weight=0.0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_recon, total_vq, total_smooth, total_n = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, _, _, _ in loader:
            x = x.permute(0, 2, 1).to(device)   # (N, T, C) → (N, C, T)
            if decimation > 1:
                x = x[:, :, ::decimation]
            x_hat, _, vq_loss, z_e = model(x)

            T = min(x.shape[-1], x_hat.shape[-1])
            recon_loss = F.mse_loss(x_hat[..., :T], x[..., :T])
            if isinstance(z_e, list):
                smooth_loss = torch.stack(
                    [((z[:, 1:] - z[:, :-1]) ** 2).mean() for z in z_e]
                ).mean()
            else:
                smooth_loss = ((z_e[:, 1:] - z_e[:, :-1]) ** 2).mean()
            loss = recon_loss + vq_loss + smooth_weight * smooth_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = x.shape[0]
            total_recon  += recon_loss.item()  * n
            total_vq     += vq_loss.item()     * n
            total_smooth += smooth_loss.item() * n
            total_n      += n

    return total_recon / total_n, total_vq / total_n, total_smooth / total_n


def codebook_usage(model, loader, device, codebook_size, decimation=1):
    model.eval()
    used = set()
    with torch.no_grad():
        for x, _, _, _ in loader:
            x = x.permute(0, 2, 1).to(device)
            if decimation > 1:
                x = x[:, :, ::decimation]
            idx = model.encode(x)
            used.update(idx.cpu().numpy().flatten().tolist())
    from src.vqvae import _is_grouped
    n_books = 3 if (_is_grouped(model) or isinstance(model, GroupedPatchVQVAE)) else 1
    return len(used) / (codebook_size * n_books)


def train(args):
    device = torch.device(args.device)

    unlab_ds = UnlabeledDataset(args.win, args.hop, norm=args.norm)
    if args.use_labeled:
        lab_ds  = LabeledDataset(args.win, args.hop, norm=args.norm)
        dataset = ConcatDataset([unlab_ds, lab_ds])
        print(f"Dataset: {len(unlab_ds)} unlabeled + {len(lab_ds)} labeled = {len(dataset)} windows")
    else:
        dataset = unlab_ds
        print(f"Dataset: {len(dataset)} unlabeled windows")

    val_len  = max(1, int(len(dataset) * args.val_frac))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    from src.dataset import SR as DATASET_SR
    decimation = DATASET_SR // args.target_sr

    arch_kwargs = dict(
        hidden_dim    = args.hidden_dim,
        latent_dim    = args.latent_dim,
        codebook_dim  = args.codebook_dim,
        codebook_size = args.codebook_size,
        n_downsample  = args.n_downsample,
        n_tcn_layers  = args.n_tcn_layers,
        kernel_size   = args.kernel_size,
    )
    patch_kwargs = dict(
        patch_size    = args.patch_size,
        d_model       = args.latent_dim,
        n_heads       = args.n_heads,
        n_layers      = args.n_layers,
        dim_ff        = args.latent_dim * 2,
        codebook_dim  = args.codebook_dim,
        codebook_size = args.codebook_size,
        commitment    = args.commitment,
    )
    if args.patch_mode == "plain":
        model = FlatPatchVQVAE(**patch_kwargs).to(device)
    elif args.patch_mode == "grouped":
        model = GroupedPatchVQVAE(**patch_kwargs).to(device)
    elif args.patch_mode == "crossmodal":
        model = ModalFusionVQVAE(**patch_kwargs).to(device)
    elif args.fused:
        patch_size = 2 ** args.n_downsample
        model = ModalFusionVQVAE(
            patch_size    = patch_size,
            d_model       = args.latent_dim,
            n_heads       = args.n_heads,
            dim_ff        = args.latent_dim * 2,
            codebook_dim  = args.codebook_dim,
            codebook_size = args.codebook_size,
        ).to(device)
    elif args.tcn_fused:
        model = TCNFusionVQVAE(
            hidden_dim    = args.hidden_dim,
            latent_dim    = args.latent_dim,
            codebook_dim  = args.codebook_dim,
            codebook_size = args.codebook_size,
            n_downsample  = args.n_downsample,
            n_tcn_layers  = args.n_tcn_layers,
            kernel_size   = args.kernel_size,
            n_heads       = args.n_heads,
        ).to(device)
    elif args.cross_group:
        model = CrossGroupVQVAE(**arch_kwargs, n_fusion_layers=args.n_layers,
                                n_attn_heads=args.n_heads).to(device)
    elif args.cross_modal_grouped:
        model = CrossModalGroupedVQVAE(**arch_kwargs, n_attn_heads=args.n_heads).to(device)
    elif args.interleaved:
        model = InterleavedGroupedVQVAE(**arch_kwargs).to(device)
    elif args.grouped:
        model = GroupedVQVAE(**arch_kwargs).to(device)
    else:
        model = VQVAE(in_channels=IN_CHANNELS, **arch_kwargs).to(device)

    T_eff    = int(args.win * args.target_sr)
    if args.patch_mode in ("plain", "grouped", "crossmodal"):
        n_tokens = T_eff // args.patch_size
        ms = round(args.patch_size / args.target_sr * 1000)
        n_cb = 3 if args.patch_mode == "grouped" else 1
        flavor_str = (f"  patch_mode={args.patch_mode}  patch_size={args.patch_size} "
                      f"({ms} ms/token)  {n_tokens} tokens/win  {n_cb} codebook(s)")
    else:
        n_tokens = T_eff // (2 ** args.n_downsample)
    if args.patch_mode in ("plain", "grouped", "crossmodal"):
        pass   # flavor_str already set above
    elif args.fused:
        flavor_str = f"  fused=True (patch={2**args.n_downsample}, {n_tokens} tokens/win, 1 codebook)"
    elif args.tcn_fused:
        flavor_str = f"  tcn_fused=True ({n_tokens} tokens/win, 1 codebook)"
    elif args.cross_group:
        flavor_str = (f"  cross_group=True (3×{n_tokens} tokens/win, "
                      f"fusion_layers={args.n_layers}, heads={args.n_heads})")
    elif args.cross_modal_grouped:
        flavor_str = f"  cross_modal_grouped=True (3×{n_tokens} tokens/win, heads={args.n_heads})"
    elif args.interleaved:
        flavor_str = f"  interleaved=True (3×{n_tokens} tokens/win)"
    elif args.grouped:
        flavor_str = f"  grouped=True (3×{n_tokens} tokens/win)"
    else:
        flavor_str = ""
    print(f"SR: {DATASET_SR}→{args.target_sr} Hz  |  tokens/window: {n_tokens}{flavor_str}")
    print_param_count(model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_recon, tr_vq, tr_sm = run_epoch(model, train_loader, device, optimizer,
                                            decimation, args.smooth_latent)
        vl_recon, vl_vq, vl_sm = run_epoch(model, val_loader,   device,
                                            decimation=decimation)
        scheduler.step()

        flag = ""
        if vl_recon < best_val:
            best_val = vl_recon
            torch.save(model.state_dict(), args.ckpt)
            flag = " *"

        smooth_str = f" | smooth {tr_sm:.4f}" if args.smooth_latent > 0 else ""
        if epoch % 5 == 0 or epoch == 1:
            usage = codebook_usage(model, val_loader, device, args.codebook_size, decimation)
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"recon {tr_recon:.4f}/{vl_recon:.4f} | "
                  f"vq {tr_vq:.4f}/{vl_vq:.4f}{smooth_str} | "
                  f"codebook {usage*100:.1f}%{flag}", flush=True)
        else:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"recon {tr_recon:.4f}/{vl_recon:.4f} | "
                  f"vq {tr_vq:.4f}/{vl_vq:.4f}{smooth_str}{flag}", flush=True)

    print(f"\nBest val recon: {best_val:.4f}  →  {args.ckpt}")


def load_and_encode(ckpt_path, win, hop, batch_size=64, labeled=False, device=None,
                    hidden_dim=128, latent_dim=256, codebook_dim=64, codebook_size=512):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(hidden_dim=hidden_dim, latent_dim=latent_dim,
                  codebook_dim=codebook_dim, codebook_size=codebook_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    if labeled:
        from src.dataset import HARDataset
        dataset = HARDataset(win, hop)
        labels  = dataset.label
        data    = dataset.data
    else:
        dataset = UnlabeledDataset(win, hop)
        labels  = None
        data    = dataset.data

    all_indices = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            x = torch.tensor(data[i:i + batch_size], dtype=torch.float32).to(device)
            all_indices.append(model.encode(x).cpu())   # encode handles (N,T,C) or (N,C,T)

    return torch.cat(all_indices, dim=0), labels


def _load_model(ckpt_path, device, hidden_dim=128, latent_dim=256,
                codebook_dim=64, codebook_size=512, n_downsample=3,
                n_tcn_layers=6, kernel_size=3):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    def _vq_shape(prefix="vq"):
        """Read codebook_size and codebook_dim from checkpoint VQ embedding."""
        embed = state[f"{prefix}._codebook.embed"]   # (1, codebook_size, codebook_dim)
        return int(embed.shape[1]), int(embed.shape[2])

    def _embed_info(embed_prefix, g_ch_for_old_compat):
        """Return (d_model, patch_size) from either CNN or old linear embed keys."""
        if f"{embed_prefix}.cnn.2.weight" in state:          # new CNN embed
            d_model    = state[f"{embed_prefix}.cnn.2.weight"].shape[0]
            patch_size = int(state[f"{embed_prefix}._patch_size"])
        else:                                                  # old linear proj
            pw         = state[f"{embed_prefix}.proj.weight"]
            d_model    = pw.shape[0]
            patch_size = pw.shape[1] // g_ch_for_old_compat
        return d_model, patch_size

    if "patch_embed.cnn.2.weight" in state or "patch_embed.proj.weight" in state:
        # FlatPatchVQVAE — singular patch_embed, all 7 channels fused
        d_model, patch_size = _embed_info("patch_embed", 7)
        deep_decoder = "decoder.proj.3.weight" in state
        if "sa.norm1.weight" in state:
            n_layers = 1
        else:
            n_layers = sum(1 for k in state if k.startswith("sa.") and k.endswith(".norm1.weight"))
        codebook_size, codebook_dim = _vq_shape("vq")
        model = FlatPatchVQVAE(
            patch_size    = patch_size,
            d_model       = d_model,
            n_layers      = n_layers,
            codebook_dim  = codebook_dim,
            codebook_size = codebook_size,
            deep_decoder  = deep_decoder,
        ).to(device)
    elif any(k.startswith("patch_embeds.") for k in state) and any(k.startswith("sa_blocks.") for k in state):
        # GroupedPatchVQVAE — grouped patch embeds, no cross-modal fusion
        d_model, patch_size = _embed_info("patch_embeds.0", 3)
        deep_decoder = "decoders.0.proj.3.weight" in state
        if "sa_blocks.0.norm1.weight" in state:
            n_layers = 1
        else:
            n_layers = sum(1 for k in state if k.startswith("sa_blocks.0.") and k.endswith(".norm1.weight"))
        codebook_size, codebook_dim = _vq_shape("vqs.0")
        model = GroupedPatchVQVAE(
            patch_size    = patch_size,
            d_model       = d_model,
            n_layers      = n_layers,
            codebook_dim  = codebook_dim,
            codebook_size = codebook_size,
            deep_decoder  = deep_decoder,
        ).to(device)
    elif any(k.startswith("patch_embeds.") for k in state):
        # ModalFusionVQVAE — grouped patch embeds + cross-modal fusion
        d_model, patch_size = _embed_info("patch_embeds.0", 3)
        deep_decoder = "decoders.0.proj.3.weight" in state
        codebook_size, codebook_dim = _vq_shape("vq")
        # detect n_layers: old ckpts have flat intra_attn.0.norm1, new have intra_attn.0.0.norm1
        if "intra_attn.0.norm1.weight" in state:
            n_layers = 1
        else:
            n_layers = sum(1 for k in state
                           if k.startswith("intra_attn.0.") and k.endswith(".norm1.weight"))
        model = ModalFusionVQVAE(
            patch_size    = patch_size,
            d_model       = d_model,
            n_layers      = n_layers,
            codebook_dim  = codebook_dim,
            codebook_size = codebook_size,
            deep_decoder  = deep_decoder,
        ).to(device)
    # ── auto-detect hidden_dim / latent_dim / codebook for TCN-based models ─
    if "encoders.0.down.0.weight" in state:
        hidden_dim = int(state["encoders.0.down.0.weight"].shape[0])
    if "vqs.0.project_in.weight" in state:
        latent_dim = int(state["vqs.0.project_in.weight"].shape[1])
    elif "group_embed.weight" in state:
        latent_dim = int(state["group_embed.weight"].shape[1])
    elif "decoders.0.up.0.weight" in state:
        latent_dim = int(state["decoders.0.up.0.weight"].shape[0])
    if "vqs.0._codebook.embed" in state:
        codebook_size, codebook_dim = int(state["vqs.0._codebook.embed"].shape[1]), int(state["vqs.0._codebook.embed"].shape[2])
    elif "vq._codebook.embed" in state:
        codebook_size, codebook_dim = int(state["vq._codebook.embed"].shape[1]), int(state["vq._codebook.embed"].shape[2])
    n_tcn_layers = sum(1 for k in state if k.startswith("encoders.0.tcn.layers.") and k.endswith(".conv.weight"))
    if n_tcn_layers == 0:
        n_tcn_layers = 6

    if any(k.startswith("encoders.") and ".blocks." in k for k in state):
        model = InterleavedGroupedVQVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
        ).to(device)
    elif any(k.startswith("cross_gyr.") for k in state):
        model = TCNFusionVQVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
        ).to(device)
    elif any(k.startswith("cross_modal_attn.") for k in state):
        model = CrossModalGroupedVQVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
        ).to(device)
    elif "group_embed.weight" in state:
        n_fusion = sum(1 for k in state if k.startswith("fusion.") and k.endswith(".norm1.weight"))
        model = CrossGroupVQVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
            n_fusion_layers=n_fusion,
        ).to(device)
    elif any(k.startswith("encoders.") for k in state):
        model = GroupedVQVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
        ).to(device)
    else:
        model = VQVAE(
            in_channels=IN_CHANNELS,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            codebook_dim=codebook_dim, codebook_size=codebook_size,
            n_downsample=n_downsample, n_tcn_layers=n_tcn_layers,
            kernel_size=kernel_size,
        ).to(device)

    # Remap old single-layer FlatPatchVQVAE sa keys (sa.X) to nested (sa.0.X)
    if isinstance(model, FlatPatchVQVAE) and "sa.norm1.weight" in state:
        state = {(f"sa.0.{k[3:]}" if k.startswith("sa.") else k): v
                 for k, v in state.items()}
    # Remap old single-layer GroupedPatchVQVAE sa_blocks keys (sa_blocks.G.X) to (sa_blocks.G.0.X)
    if isinstance(model, GroupedPatchVQVAE) and "sa_blocks.0.norm1.weight" in state:
        new_state = {}
        import re
        for k, v in state.items():
            m = re.match(r"^(sa_blocks\.\d+)\.(.+)$", k)
            if m:
                new_state[f"{m.group(1)}.0.{m.group(2)}"] = v
            else:
                new_state[k] = v
        state = new_state
    # Remap old single-layer ModalFusionVQVAE intra_attn keys (intra_attn.G.X) to (intra_attn.G.0.X)
    if isinstance(model, ModalFusionVQVAE) and "intra_attn.0.norm1.weight" in state:
        import re
        new_state = {}
        for k, v in state.items():
            m = re.match(r"^(intra_attn\.\d+)\.(.+)$", k)
            if m:
                new_state[f"{m.group(1)}.0.{m.group(2)}"] = v
            else:
                new_state[k] = v
        state = new_state

    model.load_state_dict(state)
    model.eval()
    return model


def _encode_sessions(model, dataset, device, decimation, batch_size=128):
    """Encode each session in temporal order.
    Returns {session_id: (n_windows, n_tokens) int array}."""
    session_ids = np.array(dataset.session)
    data        = dataset.data
    result      = {}
    for session in np.unique(session_ids):
        idx     = np.where(session_ids == session)[0]   # already in temporal order
        chunks  = []
        for i in range(0, len(idx), batch_size):
            x = torch.tensor(data[idx[i:i + batch_size]], dtype=torch.float32)
            x = x.permute(0, 2, 1)          # (N, T, C) → (N, C, T)
            if decimation > 1:
                x = x[:, :, ::decimation]
            with torch.no_grad():
                tok = model.encode(x.to(device)).cpu().numpy()
            if tok.ndim == 3:          # grouped: (N, G, T') → (N, G*T')
                tok = tok.reshape(tok.shape[0], -1)
            chunks.append(tok)
        result[session] = np.concatenate(chunks, axis=0)
    return result


def analyze(args):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    from collections import Counter

    device = torch.device(args.device)
    model  = _load_model(args.ckpt, device, args.hidden_dim, args.latent_dim,
                         args.codebook_dim, args.codebook_size, args.n_downsample,
                         args.n_tcn_layers, args.kernel_size)

    from src.dataset import SR as DATASET_SR
    decimation = DATASET_SR // args.target_sr

    dataset    = UnlabeledDataset(args.win, args.hop, norm=args.norm)
    sessions   = _encode_sessions(model, dataset, device, decimation)
    print(f"Encoded {len(sessions)} sessions")

    # ── per-window majority token ─────────────────────────────────────────────
    # sessions: {id: (n_win, n_tok)}  →  majority_tok: {id: (n_win,)}
    majority = {s: np.array([np.bincount(row).argmax() for row in toks])
                for s, toks in sessions.items()}

    all_tokens = np.concatenate(list(majority.values()))

    # ── 1. usage distribution ─────────────────────────────────────────────────
    usage = Counter(all_tokens.tolist())
    codes      = np.arange(args.codebook_size)
    freqs      = np.array([usage.get(c, 0) for c in codes])
    sorted_idx = np.argsort(-freqs)

    # ── 2. transition matrix (top-K tokens) ───────────────────────────────────
    K = min(args.top_k, args.codebook_size)
    top_k = sorted_idx[:K]
    top_k_set = set(top_k.tolist())
    trans = np.zeros((K, K), dtype=np.float32)
    for tok_seq in majority.values():
        for a, b in zip(tok_seq[:-1], tok_seq[1:]):
            if a in top_k_set and b in top_k_set:
                i = np.where(top_k == a)[0][0]
                j = np.where(top_k == b)[0][0]
                trans[i, j] += 1
    row_sum = trans.sum(1, keepdims=True).clip(1)
    trans   = trans / row_sum   # row-normalize → P(next | current)

    # ── 3. segment durations per token ────────────────────────────────────────
    seg_lens = []
    for tok_seq in majority.values():
        i = 0
        while i < len(tok_seq):
            j = i + 1
            while j < len(tok_seq) and tok_seq[j] == tok_seq[i]:
                j += 1
            seg_lens.append(j - i)
            i = j
    seg_lens = np.array(seg_lens)

    # ── 4. colormap: top-20 tokens get distinct colors, rest = grey ───────────
    cmap20  = plt.get_cmap("tab20")
    top20   = set(sorted_idx[:20].tolist())
    tok_color = {}
    for rank, c in enumerate(sorted_idx[:20]):
        tok_color[c] = cmap20(rank / 20)
    default_color = (0.8, 0.8, 0.8, 1.0)

    # ── 5. token × infant and token × session specificity ────────────────────
    session_to_infant = {s: inf for s, inf in zip(dataset.session, dataset.infant)}
    unique_infants    = sorted(set(session_to_infant.values()))
    unique_sessions   = sorted(sessions.keys())
    inf_to_idx        = {inf: i for i, inf in enumerate(unique_infants)}
    ses_to_idx        = {s:   i for i, s   in enumerate(unique_sessions)}

    tok_inf_mat = np.zeros((K, len(unique_infants)),  dtype=np.float32)
    tok_ses_mat = np.zeros((K, len(unique_sessions)), dtype=np.float32)
    for sid, tok_seq in majority.items():
        ii = inf_to_idx[session_to_infant[sid]]
        si = ses_to_idx[sid]
        for tok in tok_seq:
            rank = np.where(top_k == tok)[0]
            if len(rank):
                tok_inf_mat[rank[0], ii] += 1
                tok_ses_mat[rank[0], si] += 1

    # row-normalise → fraction of each token's usage per infant/session
    tok_inf_norm = tok_inf_mat / tok_inf_mat.sum(1, keepdims=True).clip(1)
    tok_ses_norm = tok_ses_mat / tok_ses_mat.sum(1, keepdims=True).clip(1)
    # subtract expected uniform distribution so deviations are visible
    tok_inf_dev  = tok_inf_norm - (1 / len(unique_infants))
    tok_ses_dev  = tok_ses_norm - (1 / len(unique_sessions))

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 20))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

    # panel 1: usage (sorted)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(np.arange(args.codebook_size), freqs[sorted_idx], width=1,
            color="steelblue", linewidth=0)
    ax1.set_xlabel("Code rank"); ax1.set_ylabel("Window count")
    ax1.set_title("Codebook usage (sorted by frequency)")
    used = (freqs > 0).sum()
    ax1.text(0.98, 0.95, f"{used}/{args.codebook_size} codes used",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9)

    # panel 2: segment duration histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(seg_lens * args.win, bins=40, color="coral", edgecolor="none", density=True)
    ax2.set_xlabel("Segment duration (s)"); ax2.set_ylabel("Density")
    ax2.set_title("Segment duration distribution")
    ax2.text(0.98, 0.95, f"median {np.median(seg_lens)*args.win:.1f}s",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9)

    # panel 3: transition matrix
    ax3 = fig.add_subplot(gs[1, :2])
    im  = ax3.imshow(trans, aspect="auto", cmap="hot", vmin=0, vmax=trans.max())
    ax3.set_xlabel("Next token (rank)"); ax3.set_ylabel("Current token (rank)")
    ax3.set_title(f"Transition matrix — top-{K} tokens (row-normalised)")
    plt.colorbar(im, ax=ax3, fraction=0.03)

    # panel 4: UMAP of codebook vectors
    ax4 = fig.add_subplot(gs[1, 2])
    try:
        from umap import UMAP
        from src.vqvae import get_codebook
        cpu_model = _load_model(args.ckpt, torch.device("cpu"),
                                args.hidden_dim, args.latent_dim,
                                args.codebook_dim, args.codebook_size,
                                args.n_downsample, args.n_tcn_layers, args.kernel_size)
        cb = get_codebook(cpu_model)   # (total_vocab, codebook_dim)
        n_cb = len(cb)
        all_tok_flat = np.concatenate([toks.flatten() for toks in sessions.values()])
        cb_usage = Counter(all_tok_flat.tolist())
        cb_freqs = np.array([cb_usage.get(i, 0) for i in range(n_cb)], dtype=float)
        cb_sorted = np.argsort(-cb_freqs)
        cb_colors = {c: cmap20(r / 20) for r, c in enumerate(cb_sorted[:20])}
        colors_cb = [cb_colors.get(i, default_color) for i in range(n_cb)]
        proj = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(cb)
        sizes = 12 + cb_freqs / cb_freqs.max().clip(1) * 60
        ax4.scatter(proj[:, 0], proj[:, 1], c=colors_cb, s=sizes,
                    alpha=0.7, linewidths=0)
        ax4.set_title("UMAP — codebook entries\n(size ∝ usage, top-20 colored)")
    except ImportError:
        ax4.text(0.5, 0.5, "umap-learn not installed", ha="center", va="center",
                 transform=ax4.transAxes)
        ax4.set_title("UMAP (unavailable)")
    ax4.set_xticks([]); ax4.set_yticks([])

    # panel 5: token timelines (sample sessions)
    ax5 = fig.add_subplot(gs[2, :])
    session_sample = list(sessions.keys())[:args.n_sessions]
    n_shown = len(session_sample)
    for row, sid in enumerate(session_sample):
        tok_seq = majority[sid]
        colors_row = [tok_color.get(t, default_color) for t in tok_seq]
        for col, c in enumerate(colors_row):
            ax5.barh(row, 1, left=col, height=0.8, color=c, linewidth=0)
    ax5.set_yticks(range(n_shown))
    ax5.set_yticklabels([s[:16] for s in session_sample], fontsize=7)
    ax5.set_xlabel("Window index (time →)")
    ax5.set_title(f"Token timelines — {n_shown} sessions (top-20 codes colored, grey = other)")
    ax5.invert_yaxis()

    # panel 6: token × infant deviation heatmap
    ax6 = fig.add_subplot(gs[3, :2])
    vmax = np.abs(tok_inf_dev).max()
    im6  = ax6.imshow(tok_inf_dev.T, aspect="auto", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax)
    ax6.set_xlabel("Token rank"); ax6.set_ylabel("Infant")
    ax6.set_yticks(range(len(unique_infants)))
    ax6.set_yticklabels(unique_infants, fontsize=6)
    ax6.set_title(f"Token usage deviation from uniform — by infant\n"
                  f"(red = over-represented, blue = under-represented)")
    plt.colorbar(im6, ax=ax6, fraction=0.02)

    # panel 7: token × session deviation heatmap
    ax7 = fig.add_subplot(gs[3, 2])
    vmax7 = np.abs(tok_ses_dev).max()
    im7   = ax7.imshow(tok_ses_dev.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax7, vmax=vmax7)
    ax7.set_xlabel("Token rank"); ax7.set_ylabel("Session")
    ax7.set_yticks([])
    ax7.set_title("Deviation — by session")
    plt.colorbar(im7, ax=ax7, fraction=0.04)

    plt.suptitle("VQ-VAE action tokenization analysis", fontsize=13, y=1.01)
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
    plt.show()


def reconstruct(args):
    """5-column plot: 2×low-MSE (Q25) | median | 2×high-MSE (Q75), shared y per channel."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "pressure"]
    device = torch.device(args.device)
    decimation = 100 // args.target_sr

    model = _load_model(args.ckpt, device,
                        hidden_dim=128, latent_dim=256,
                        codebook_dim=args.codebook_dim, codebook_size=args.codebook_size,
                        n_downsample=args.n_downsample,
                        n_tcn_layers=6, kernel_size=3)

    dataset = UnlabeledDataset(args.win, args.hop, norm=args.norm)
    data    = dataset.data   # (N, T, C)

    # ── sample pool and forward pass ──────────────────────────────────────────
    rng       = np.random.default_rng(args.seed)
    pool_idx  = rng.choice(len(data), size=min(args.n_pool, len(data)), replace=False)
    pool_idx  = np.sort(pool_idx)

    x_pool = torch.tensor(data[pool_idx], dtype=torch.float32).permute(0, 2, 1)  # (P, C, T)
    if decimation > 1:
        x_pool = x_pool[:, :, ::decimation]

    all_x_in, all_x_hat, all_toks = [], [], []
    with torch.no_grad():
        for i in range(0, len(pool_idx), 128):
            xb = x_pool[i:i + 128].to(device)
            xh, idx_b, _, _ = model(xb)
            T_out = xh.shape[-1]           # may be < T_in if T not divisible by patch_size
            all_x_in.append(xb[..., :T_out].cpu().numpy())
            all_x_hat.append(xh.cpu().numpy())
            all_toks.append(idx_b.cpu().numpy())

    x_in_pool  = np.concatenate(all_x_in,  axis=0)   # (P, C, T)
    x_hat_pool = np.concatenate(all_x_hat, axis=0)
    toks_pool  = np.concatenate(all_toks,  axis=0)   # (P, n_tok)

    # ── select 5 representative windows ──────────────────────────────────────
    mse_pool   = np.mean((x_in_pool - x_hat_pool) ** 2, axis=(1, 2))   # (P,)
    order      = np.argsort(mse_pool)
    P          = len(order)
    q25        = max(1, int(0.25 * P))
    q75        = int(0.75 * P)

    low_grp    = order[:q25]
    high_grp   = order[q75:]
    med_i      = order[P // 2]


    if args.index is not None:
        sel = [args.index]
    else:
        sel = [med_i]

    x_in  = x_in_pool[sel]    # (1, C, T)
    x_hat = x_hat_pool[sel]
    mses  = mse_pool[sel]

    n_ch = x_in.shape[1]
    T    = x_in.shape[2]
    t    = np.arange(T) / args.target_sr

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    ax_grid = []
    # row 0: acc_x, acc_y, acc_z
    for c in range(3):
        ax_grid.append(fig.add_subplot(gs[0, c]))
    # row 1: gyr_x, gyr_y, gyr_z
    for c in range(3):
        ax_grid.append(fig.add_subplot(gs[1, c]))
    # row 2: pressure spanning all 3 columns
    ax_grid.append(fig.add_subplot(gs[2, :]))

    ch_map = [0, 1, 2, 3, 4, 5, 6]
    for idx, ch in enumerate(ch_map):
        ax = ax_grid[idx]
        ax.plot(t, x_in[0, ch],  color="#2166ac", lw=0.8)
        ax.plot(t, x_hat[0, ch], color="#d73027", lw=0.8, linestyle="--")
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (s)", fontsize=9)

    # ── shared y-scale per sensor group ───────────────────────────────────────
    for start, end in [(0, 3), (3, 6)]:
        ymin = min(ax_grid[i].get_ylim()[0] for i in range(start, end))
        ymax = max(ax_grid[i].get_ylim()[1] for i in range(start, end))
        for i in range(start, end):
            ax_grid[i].set_ylim(ymin, ymax)

    handles = [
        mpatches.Patch(color="#2166ac", label="original"),
        mpatches.Patch(color="#d73027", label="reconstruction"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    fig.suptitle(f"VQ-VAE Reconstruction (MSE = {mses[0]:.4f})", fontsize=13, y=0.98)
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
    plt.show()


def plot_representations(args):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    indices, labels = load_and_encode(args.ckpt, args.win, args.hop, labeled=args.labeled)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    codebook = model.vq.codebook.detach().numpy()

    embeddings = codebook[indices.numpy()].mean(axis=1)

    if args.method == "umap":
        from umap import UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 8))
    if labels is not None:
        classes = np.unique(labels)
        colors  = cm.tab10(np.linspace(0, 1, len(classes)))
        for cls, col in zip(classes, colors):
            mask = labels == cls
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[col], label=str(cls), s=8, alpha=0.6)
        ax.legend(title="Action", markerscale=2, fontsize=8)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.4, c="steelblue")

    ax.set_title(f"VQ-VAE representations — {args.method.upper()}")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    plt.show()
    print(f"Saved → {args.save}")


def _default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VQ-VAE for IMU sequences")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="train the VQ-VAE")
    t.add_argument("--win",           type=float, default=3.0)
    t.add_argument("--hop",           type=float, default=1.5)
    t.add_argument("--norm",          type=str,   default="infant",
                   choices=["infant", "session", "none"])
    t.add_argument("--use-labeled",   action="store_true",
                   help="include labeled dataset windows in training")
    t.add_argument("--val-frac",      type=float, default=0.1)
    # architecture
    t.add_argument("--hidden-dim",    type=int,   default=128)
    t.add_argument("--latent-dim",    type=int,   default=256)
    t.add_argument("--codebook-dim",  type=int,   default=64)
    t.add_argument("--codebook-size", type=int,   default=512)
    t.add_argument("--n-downsample",  type=int,   default=3,
                   help="encoder stride-2 layers: 2=4x, 3=8x, 4=16x")
    t.add_argument("--n-tcn-layers",  type=int,   default=6,
                   help="TCN dilated layers in encoder and decoder")
    t.add_argument("--kernel-size",   type=int,   default=3,
                   help="TCN kernel size (odd)")
    t.add_argument("--target-sr",     type=int,   default=50,
                   help="target sample rate (Hz); decimates from dataset SR=100")
    # training
    t.add_argument("--epochs",        type=int,   default=100)
    t.add_argument("--batch-size",    type=int,   default=64)
    t.add_argument("--lr",             type=float, default=3e-4)
    t.add_argument("--weight-decay",   type=float, default=1e-5)
    t.add_argument("--smooth-latent",  type=float, default=0.1,
                   help="weight on temporal smoothness loss for z_e")
    t.add_argument("--cross-group",    action="store_true",
                   help="use CrossGroupVQVAE (grouped TCN + sequence-level cross-group SA before VQ)")
    t.add_argument("--grouped",        action="store_true",
                   help="use GroupedVQVAE (one codebook per sensor group: acc/gyr/pressure)")
    t.add_argument("--interleaved",    action="store_true",
                   help="use InterleavedGroupedVQVAE (interleaved TCN residual blocks)")
    t.add_argument("--cross-modal-grouped", action="store_true",
                   help="use CrossModalGroupedVQVAE (grouped + symmetric cross-modal attn before VQ)")
    t.add_argument("--fused",          action="store_true",
                   help="use ModalFusionVQVAE (patch + cross-modal attn + single codebook)")
    t.add_argument("--tcn-fused",      action="store_true",
                   help="use TCNFusionVQVAE (TCN encoders + cross-modal attn + single codebook)")
    t.add_argument("--n-heads",        type=int,   default=4,
                   help="attention heads (fused / patch models)")
    t.add_argument("--n-layers",       type=int,   default=3,
                   help="transformer layers in patch encoder (plain/grouped)")
    t.add_argument("--patch-mode",    type=str,   default="none",
                   choices=["none", "plain", "grouped", "crossmodal"],
                   help="patch-encoder VQ-VAE: plain=flat 7ch, grouped=acc/gyr/pres×3 codebooks, "
                        "crossmodal=grouped+cross-attn (ModalFusionVQVAE)")
    t.add_argument("--patch-size",    type=int,   default=25,
                   help="frames per token for patch models (at --target-sr Hz); "
                        "100 Hz × 12→120ms, 25→250ms, 50→500ms")
    t.add_argument("--commitment",    type=float, default=1.0,
                   help="VQ commitment loss weight (default 1.0; try 0.1 to escape EMA fixed-point)")
    t.add_argument("--ckpt",          type=str,   default="models/best_vqvae.pt")
    t.add_argument("--device",        type=str,   default=_default_device())

    # ── plot ──────────────────────────────────────────────────────────────────
    pl = sub.add_parser("plot", help="plot VQ-VAE representations")
    pl.add_argument("--ckpt",    type=str,   default="models/best_vqvae.pt")
    pl.add_argument("--win",     type=float, default=3.0)
    pl.add_argument("--hop",     type=float, default=1.5)
    pl.add_argument("--labeled", action="store_true")
    pl.add_argument("--method",  choices=["umap", "tsne"], default="umap")
    pl.add_argument("--save",    type=str,   default="vqvae_repr.png")

    # ── reconstruct ───────────────────────────────────────────────────────────
    rc = sub.add_parser("reconstruct", help="plot original vs reconstructed signal")
    rc.add_argument("--ckpt",          type=str,   default="models/best_vqvae.pt")
    rc.add_argument("--win",           type=float, default=5.12)
    rc.add_argument("--hop",           type=float, default=2.56)
    rc.add_argument("--norm",          type=str,   default="infant",
                    choices=["infant", "session", "none"])
    rc.add_argument("--target-sr",     type=int,   default=50)
    rc.add_argument("--n-downsample",  type=int,   default=3)
    rc.add_argument("--codebook-size", type=int,   default=512)
    rc.add_argument("--codebook-dim",  type=int,   default=64)
    rc.add_argument("--grouped",        action="store_true")
    rc.add_argument("--n-pool",         type=int,   default=500,
                    help="pool size for MSE quantile selection")
    rc.add_argument("--seed",          type=int,   default=42)
    rc.add_argument("--index",         type=int,   default=None,
                    help="pool index to plot (default: median MSE)")
    rc.add_argument("--save",          type=str,   default="figs/vqvae_recon.png")
    rc.add_argument("--device",        type=str,   default=_default_device())

    # ── analyze ───────────────────────────────────────────────────────────────
    an = sub.add_parser("analyze", help="visualize tokenization quality")
    an.add_argument("--ckpt",          type=str,   default="models/best_vqvae.pt")
    an.add_argument("--win",           type=float, default=5.12)
    an.add_argument("--hop",           type=float, default=2.56)
    an.add_argument("--norm",          type=str,   default="infant",
                    choices=["infant", "session", "none"])
    an.add_argument("--target-sr",     type=int,   default=50)
    an.add_argument("--n-downsample",  type=int,   default=3)
    an.add_argument("--n-tcn-layers",  type=int,   default=6)
    an.add_argument("--kernel-size",   type=int,   default=3)
    an.add_argument("--hidden-dim",    type=int,   default=128)
    an.add_argument("--latent-dim",    type=int,   default=256)
    an.add_argument("--codebook-dim",  type=int,   default=64)
    an.add_argument("--codebook-size", type=int,   default=512)
    an.add_argument("--top-k",         type=int,   default=30,
                    help="top-K tokens shown in transition matrix")
    an.add_argument("--n-sessions",    type=int,   default=10,
                    help="number of sessions shown in timeline panel")
    an.add_argument("--save",          type=str,   default="vqvae_analysis.png")
    an.add_argument("--device",        type=str,   default=_default_device())

    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "analyze":
        analyze(args)
    elif args.cmd == "reconstruct":
        reconstruct(args)
    else:
        plot_representations(args)
