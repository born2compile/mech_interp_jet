"""
Activation cache, PatchableParT, corruption strategies, and path-patching sweep.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import CFG, DEVICE
from .model import SmallParT


# ── Activation cache ──────────────────────────────────────────────────────────

class ActivationCache:
    def __init__(self):
        self.cache = {}

    def store(self, key, tensor):
        self.cache[key] = tensor.detach().clone()

    def get(self, key):
        return self.cache[key]

    def clear(self):
        self.cache = {}


# ── Corruption strategies ─────────────────────────────────────────────────────

def make_corrupt_input(x_top, v_top, strategy="within_batch"):
    """
    Within-batch particle replacement (default): replace each top jet's
    particles with those of another jet in the batch, offset by half the
    batch size.  Preserves the per-particle feature distribution while
    breaking jet-level kinematic correlations.

    Parameters
    ----------
    x_top  : (B, 7, P) float tensor — particle features of top jets
    v_top  : (B, 4, P) float tensor — four-vectors of top jets
    strategy : "within_batch" (default) | "permutation"
               "permutation" permutes intact jets across the batch.

    Returns
    -------
    x_c, v_c : corrupted versions, same shape
    """
    B = x_top.size(0)
    if strategy == "permutation":
        perm = torch.randperm(B, device=x_top.device)
        return x_top[perm], v_top[perm]
    else:  # within_batch
        shift  = B // 2
        offset = torch.arange(B, device=x_top.device)
        offset = (offset + shift) % B
        return x_top[:, :, offset], v_top[:, :, offset]


# ── PatchableParT ─────────────────────────────────────────────────────────────

class PatchableParT(SmallParT):
    """
    Extends SmallParT with activation caching and per-head patching.

    Usage
    -----
    model_p.run_and_cache(x, v, mask, cache_name='clean')
    model_p.run_and_cache(x_c, v_c, mask, cache_name='corrupt')
    logits = model_p.run_with_patch(x_c, v_c, mask,
                                     patch_layer=l, patch_head=h,
                                     source_cache='clean')
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.caches       = {}
        self.patch_config = None

    def run_and_cache(self, x, v, mask, cache_name):
        cache = ActivationCache()
        self.caches[cache_name] = cache
        self.patch_config = None
        with torch.no_grad():
            logits = self._instrumented_forward(x, v, mask, cache=cache)
        return logits

    @torch.no_grad()
    def run_with_patch(self, x, v, mask,
                       patch_layer, patch_head, source_cache="clean"):
        self.patch_config = dict(layer=patch_layer, head=patch_head,
                                 source_cache=source_cache)
        logits = self._instrumented_forward(x, v, mask, cache=None)
        self.patch_config = None
        return logits

    def _instrumented_forward(self, x, v, mask, cache=None):
        self.residual_stream  = []
        self.attn_weights     = []
        self.cls_attn_weights = []

        batch_size = x.size(0)
        P          = x.size(2)
        num_heads  = self.num_heads

        real_mask = mask.bool().squeeze(1)
        pad_mask  = ~real_mask

        pair_bias = self.pair_embed(v)
        col_mask  = (pad_mask.float() * -1e4)[:, None, None, :]
        row_mask  = (pad_mask.float() * -1e4)[:, None, :, None]
        attn_bias = pair_bias + col_mask + row_mask
        attn_mask = attn_bias.reshape(batch_size * num_heads, P, P)

        x = self.embed(x)
        x = x.masked_fill(pad_mask.T.unsqueeze(-1), 0.)
        self.residual_stream.append(x.permute(1, 0, 2).detach().cpu())

        for layer_idx, block in enumerate(self.blocks):
            x = self._block_forward_with_hooks(
                block, x, layer_idx, attn_mask, pad_mask, cache)
            x = x.masked_fill(pad_mask.T.unsqueeze(-1), 0.)
            self.residual_stream.append(x.permute(1, 0, 2).detach().cpu())
            self.attn_weights.append(block.last_attn_weights_per_head)

        cls_tokens = self.cls_token.expand(1, batch_size, -1).clone()
        for cls_block in self.cls_blocks:
            cls_tokens = cls_block(x, x_cls=cls_tokens, padding_mask=pad_mask)
            self.cls_attn_weights.append(cls_block.last_attn_weights_per_head)

        x_cls = self.norm(cls_tokens).squeeze(0)
        return self.fc(x_cls)

    def _block_forward_with_hooks(self, block, x, layer_idx,
                                   attn_mask, pad_mask, cache):
        """Run block forward; optionally cache or patch head output."""
        # Run block
        x_out = block(x, x_cls=None, padding_mask=None, attn_mask=attn_mask)

        # Cache the per-head output
        key = f"l{layer_idx}_h"
        if cache is not None:
            for h in range(self.num_heads):
                cache.store(f"l{layer_idx}_h{h}", x_out)

        # Apply patch if configured
        if (self.patch_config is not None and
                self.patch_config["layer"] == layer_idx):
            h_patch     = self.patch_config["head"]
            src_cache   = self.caches[self.patch_config["source_cache"]]
            clean_out   = src_cache.get(f"l{layer_idx}_h{h_patch}")
            x_out = clean_out.to(x_out.device)

        return x_out


# ── Path patching sweep ───────────────────────────────────────────────────────

def path_patch_sweep(model_p, dataset, indices, important_heads,
                     batch_size=128, strategy="within_batch"):
    """
    Measures:
      direct_effects : dict (l, h) -> recovery score
      path_effects   : dict (src, tgt) -> representational delta

    Parameters
    ----------
    model_p        : PatchableParT instance (weights loaded)
    dataset        : TopTagDataset
    indices        : np.ndarray  jet indices to use
    important_heads: list of (l, h) tuples
    strategy       : corruption strategy passed to make_corrupt_input
    """
    # ── Step 1: baseline clean and corrupt LDs ───────────────────────────────
    clean_lds, corrupt_lds = [], []
    print("Collecting clean and corrupt baseline LDs ...")
    for start in tqdm(range(0, len(indices), batch_size), desc="  Baseline runs"):
        idx  = indices[start:start + batch_size]
        x    = dataset.x[idx].to(DEVICE)
        v    = dataset.v[idx].to(DEVICE)
        mask = dataset.mask[idx].to(DEVICE)
        lbls = dataset.labels[idx]
        top  = lbls == 1
        if top.sum() == 0:
            continue
        x_t, v_t, m_t = x[top], v[top], mask[top]
        x_c, v_c      = make_corrupt_input(x_t, v_t, strategy=strategy)
        lc = model_p.run_and_cache(x_t, v_t, m_t, cache_name="clean")
        lk = model_p.run_and_cache(x_c, v_c, m_t, cache_name="corrupt")
        clean_lds.append((lc[:, 1] - lc[:, 0]).cpu())
        corrupt_lds.append((lk[:, 1] - lk[:, 0]).cpu())

    clean_ld_mean   = torch.cat(clean_lds).mean().item()
    corrupt_ld_mean = torch.cat(corrupt_lds).mean().item()
    denominator     = clean_ld_mean - corrupt_ld_mean

    print(f"  Clean LD    : {clean_ld_mean:.4f}")
    print(f"  Corrupt LD  : {corrupt_ld_mean:.4f}")
    print(f"  Denominator : {denominator:.4f}")
    if abs(denominator) < 0.01:
        print("  WARNING: denominator too small — scores unreliable.")

    # ── Step 2: direct effects ────────────────────────────────────────────────
    print("\nMeasuring direct effects ...")
    direct_effects = {}

    for (l, h) in tqdm(important_heads, desc="  Direct effects"):
        patched_lds = []
        for start in range(0, len(indices), batch_size):
            idx  = indices[start:start + batch_size]
            x    = dataset.x[idx].to(DEVICE)
            v    = dataset.v[idx].to(DEVICE)
            mask = dataset.mask[idx].to(DEVICE)
            lbls = dataset.labels[idx]
            top  = lbls == 1
            if top.sum() == 0:
                continue
            x_t, v_t, m_t = x[top], v[top], mask[top]
            x_c, v_c      = make_corrupt_input(x_t, v_t, strategy=strategy)
            model_p.run_and_cache(x_t, v_t, m_t, cache_name="clean")
            lp = model_p.run_with_patch(x_c, v_c, m_t,
                                         patch_layer=l, patch_head=h,
                                         source_cache="clean")
            patched_lds.append((lp[:, 1] - lp[:, 0]).cpu())

        patched_mean          = torch.cat(patched_lds).mean().item()
        recovery              = (patched_mean - corrupt_ld_mean) / (denominator + 1e-8)
        direct_effects[(l, h)] = recovery
        print(f"  Layer {l} Head {h}: patched_LD={patched_mean:.4f}  recovery={recovery:.4f}")

    # ── Step 3: path effects ──────────────────────────────────────────────────
    print("\nMeasuring path effects (src → tgt) ...")
    path_effects = {}

    for src in tqdm(important_heads, desc="  Path effects"):
        for tgt in important_heads:
            if src == tgt:
                continue
            l_src, h_src = src
            l_tgt, h_tgt = tgt
            if l_src >= l_tgt:
                continue

            key_tgt          = f"l{l_tgt}_h{h_tgt}"
            tgt_corrupt_list = []
            tgt_patched_list = []

            for start in range(0, min(len(indices), 1000), batch_size):
                idx  = indices[start:start + batch_size]
                x    = dataset.x[idx].to(DEVICE)
                v    = dataset.v[idx].to(DEVICE)
                mask = dataset.mask[idx].to(DEVICE)
                lbls = dataset.labels[idx]
                top  = lbls == 1
                if top.sum() == 0:
                    continue
                x_t, v_t, m_t = x[top], v[top], mask[top]
                x_c, v_c      = make_corrupt_input(x_t, v_t, strategy=strategy)

                model_p.run_and_cache(x_c, v_c, m_t, cache_name="corrupt_run")
                tgt_corrupt_list.append(
                    model_p.caches["corrupt_run"].get(key_tgt).cpu())

                model_p.run_and_cache(x_t, v_t, m_t, cache_name="clean")
                patched_cache        = ActivationCache()
                model_p.patch_config = dict(layer=l_src, head=h_src,
                                            source_cache="clean")
                model_p._instrumented_forward(x_c, v_c, m_t, cache=patched_cache)
                model_p.patch_config       = None
                model_p.caches["patched"]  = patched_cache
                tgt_patched_list.append(patched_cache.get(key_tgt).cpu())

            if len(tgt_corrupt_list) == 0:
                continue

            tgt_corrupt_all = torch.cat(tgt_corrupt_list, dim=0)
            tgt_patched_all = torch.cat(tgt_patched_list, dim=0)
            delta           = (tgt_patched_all - tgt_corrupt_all).norm(dim=-1).mean().item()
            path_effects[(src, tgt)] = delta
            print(f"  {src} → {tgt} : Δ = {delta:.6f}")

    return direct_effects, path_effects


# ── Bootstrap confidence intervals for path effects ──────────────────────────

def bootstrap_path_effects(model_p, dataset, indices, important_heads,
                             n_boot=500, batch_size=128, strategy="within_batch"):
    """
    Re-run path_patch_sweep with bootstrap resampling to obtain 95% CIs
    on path effect magnitudes.

    Returns
    -------
    pe_mean : dict (src, tgt) -> float
    pe_ci   : dict (src, tgt) -> (lo, hi)
    """
    all_deltas = {(src, tgt): [] for src in important_heads
                  for tgt in important_heads
                  if src != tgt and src[0] < tgt[0]}

    rng = np.random.default_rng(42)

    for boot in tqdm(range(n_boot), desc="Bootstrap path effects"):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        _, pe = path_patch_sweep(
            model_p, dataset, boot_idx[:200], important_heads,
            batch_size=batch_size, strategy=strategy)
        for k, v in pe.items():
            if k in all_deltas:
                all_deltas[k].append(v)

    pe_mean = {}
    pe_ci   = {}
    for k, vals in all_deltas.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        pe_mean[k] = float(arr.mean())
        pe_ci[k]   = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))

    print("\n── Path effects with 95% bootstrap CI ──")
    print(f"  {'Path':20s}  {'Mean Δ':>10s}  {'95% CI':>22s}")
    print("─" * 58)
    for (src, tgt) in sorted(pe_mean.keys()):
        lo, hi = pe_ci[(src, tgt)]
        print(f"  L{src[0]}H{src[1]} → L{tgt[0]}H{tgt[1]}   "
              f"  {pe_mean[(src,tgt)]:10.6f}  [{lo:.6f}, {hi:.6f}]")

    return pe_mean, pe_ci
