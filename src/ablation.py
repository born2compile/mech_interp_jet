"""
Zero ablation and mean ablation of attention heads, with bootstrap uncertainty.
Includes BlockWithCapture for mean-ablation experiments.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CFG, DEVICE


# ── Logit difference utility ──────────────────────────────────────────────────

@torch.no_grad()
def mean_ld_top(model, dataset, indices, batch_size=512):
    """Mean logit difference over top jets."""
    model.eval()
    diffs = []
    for start in range(0, len(indices), batch_size):
        idx    = indices[start:start + batch_size]
        x      = dataset.x[idx].to(DEVICE)
        v      = dataset.v[idx].to(DEVICE)
        mask   = dataset.mask[idx].to(DEVICE)
        labels = dataset.labels[idx]
        logits = model(x, v, mask).cpu()
        ld     = logits[:, 1] - logits[:, 0]
        diffs.append(ld[labels == 1])
    return torch.cat(diffs).mean().item()


def logit_diff_with_uncertainty(model, dataset, indices,
                                 batch_size=512, n_bootstrap=1000):
    """
    Returns (mean, std) of logit difference over top jets
    using bootstrap resampling for uncertainty quantification.
    """
    model.eval()
    all_ld = []

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx    = indices[start:start + batch_size]
            x      = dataset.x[idx].to(DEVICE)
            v      = dataset.v[idx].to(DEVICE)
            mask   = dataset.mask[idx].to(DEVICE)
            labels = dataset.labels[idx]
            logits = model(x, v, mask).cpu()
            ld     = logits[:, 1] - logits[:, 0]
            all_ld.append(ld[labels == 1])

    all_ld     = torch.cat(all_ld).numpy()
    boot_means = [
        all_ld[np.random.choice(len(all_ld), len(all_ld), replace=True)].mean()
        for _ in range(n_bootstrap)
    ]
    return float(all_ld.mean()), float(np.std(boot_means))


# ── Zero ablation sweep ───────────────────────────────────────────────────────

def zero_ablation_sweep(model, dataset, ablation_idx,
                         n_layers=None, n_heads=None, n_bootstrap=1000):
    """
    Ablate each head by setting c_attn[h] = 0.
    Returns imp_mean (n_layers, n_heads) and imp_std (n_layers, n_heads).
    """
    if n_layers is None:
        n_layers = CFG["num_layers"]
    if n_heads is None:
        n_heads  = CFG["num_heads"]

    baseline_mean, baseline_std = logit_diff_with_uncertainty(
        model, dataset, ablation_idx, n_bootstrap=n_bootstrap)
    print(f"Baseline LD: {baseline_mean:.4f} ± {baseline_std:.4f}")

    imp_mean = np.zeros((n_layers, n_heads), dtype=np.float32)
    imp_std  = np.zeros((n_layers, n_heads), dtype=np.float32)

    pbar = tqdm(total=n_layers * n_heads, desc="Zero ablation sweep")
    for l in range(n_layers):
        block        = model.blocks[l]
        orig_c_attn  = block.c_attn.data.clone()
        for h in range(n_heads):
            ablated        = orig_c_attn.clone()
            ablated[h]     = 0.0
            block.c_attn.data = ablated

            abl_mean, abl_std = logit_diff_with_uncertainty(
                model, dataset, ablation_idx, n_bootstrap=n_bootstrap)
            imp_mean[l, h] = baseline_mean - abl_mean
            imp_std[l, h]  = np.sqrt(baseline_std**2 + abl_std**2)

            pbar.set_postfix(l=l, h=h,
                             imp=f"{imp_mean[l,h]:.3f}±{imp_std[l,h]:.3f}")
            pbar.update(1)

        block.c_attn.data = orig_c_attn

    pbar.close()
    return imp_mean, imp_std


# ── BlockWithCapture (mean ablation) ─────────────────────────────────────────

class BlockWithCapture(nn.Module):
    """
    Extends Block with per-head output capture and mean ablation support.
    """
    def __init__(self, embed_dim=128, num_heads=4, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation="gelu",
                 scale_fc=True, scale_attn=True,
                 scale_heads=True, scale_resids=True):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.ffn_dim   = embed_dim * ffn_ratio

        self.pre_attn_norm  = nn.LayerNorm(embed_dim)
        self.attn           = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout        = nn.Dropout(dropout)
        self.pre_fc_norm    = nn.LayerNorm(embed_dim)
        self.fc1            = nn.Linear(embed_dim, self.ffn_dim)
        self.act            = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout    = nn.Dropout(activation_dropout)
        self.post_fc_norm   = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2            = nn.Linear(self.ffn_dim, embed_dim)
        self.c_attn  = nn.Parameter(torch.ones(num_heads), requires_grad=True) \
                       if scale_heads  else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) \
                       if scale_resids else None

        self.last_head_out_raw = None
        self.last_attn_weights_per_head = None
        self.ablate_head    = None
        self.mean_head_out  = None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        if x_cls is not None:
            with torch.no_grad():
                cls_pad      = torch.zeros(padding_mask.size(0), 1,
                                           dtype=padding_mask.dtype,
                                           device=padding_mask.device)
                padding_mask = torch.cat((cls_pad, padding_mask), dim=1)
            residual = x_cls
            u        = torch.cat((x_cls, x), dim=0)
            u        = self.pre_attn_norm(u)
            x        = self.attn(x_cls, u, u,
                                  key_padding_mask=padding_mask,
                                  need_weights=False)[0]
        else:
            residual = x
            x_norm   = self.pre_attn_norm(x)
            attn_out, attn_w = self.attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=None,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False)
            x = attn_out
            self.last_attn_weights_per_head = attn_w.detach().cpu()

        # Per-head split and optional ablation
        tgt_len = x.size(0)
        x_heads = x.view(tgt_len, -1, self.num_heads, self.head_dim)

        if self.ablate_head is not None and self.mean_head_out is not None:
            x_heads = x_heads.clone()
            x_heads[:, :, self.ablate_head, :] = self.mean_head_out

        self.last_head_out_raw = x_heads.detach().clone()

        if self.c_attn is not None:
            x_heads = torch.einsum("tbhd,h->tbdh", x_heads, self.c_attn)
        x = x_heads.reshape(tgt_len, -1, self.embed_dim)

        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x        = self.pre_fc_norm(x)
        x        = self.act(self.fc1(x))
        x        = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual
        return x


# ── Mean ablation sweep ───────────────────────────────────────────────────────

def mean_ablation_sweep(model_capture, dataset, ablation_idx,
                         n_layers=None, n_heads=None):
    """
    Replace each head's output with its dataset mean and measure importance.
    Requires model blocks to be BlockWithCapture instances.

    Returns imp_mean (n_layers, n_heads).
    """
    if n_layers is None:
        n_layers = CFG["num_layers"]
    if n_heads is None:
        n_heads  = CFG["num_heads"]

    baseline_ld = mean_ld_top(model_capture, dataset, ablation_idx)
    print(f"Baseline LD (mean ablation): {baseline_ld:.4f}")

    # ── Step 1: collect mean head outputs ────────────────────────────────────
    print("Collecting mean head outputs ...")
    mean_outputs = {l: {h: [] for h in range(n_heads)}
                    for l in range(n_layers)}

    with torch.no_grad():
        for start in tqdm(range(0, len(ablation_idx), 256),
                          desc="  Collecting heads", leave=False):
            idx  = ablation_idx[start:start + 256]
            x    = dataset.x[idx].to(DEVICE)
            v    = dataset.v[idx].to(DEVICE)
            mask = dataset.mask[idx].to(DEVICE)
            model_capture(x, v, mask)
            for l in range(n_layers):
                raw = model_capture.blocks[l].last_head_out_raw
                if raw is not None:
                    for h in range(n_heads):
                        mean_outputs[l][h].append(
                            raw[:, :, h, :].mean(dim=(0, 1)).detach().cpu())

    head_means = {}
    for l in range(n_layers):
        for h in range(n_heads):
            if mean_outputs[l][h]:
                head_means[(l, h)] = torch.stack(
                    mean_outputs[l][h]).mean(dim=0).to(DEVICE)

    # ── Step 2: ablation sweep ────────────────────────────────────────────────
    imp_mean = np.zeros((n_layers, n_heads), dtype=np.float32)
    pbar = tqdm(total=n_layers * n_heads, desc="Mean ablation sweep")

    for l in range(n_layers):
        for h in range(n_heads):
            model_capture.blocks[l].ablate_head   = h
            model_capture.blocks[l].mean_head_out = head_means.get((l, h))
            abl_ld = mean_ld_top(model_capture, dataset, ablation_idx)
            imp_mean[l, h] = baseline_ld - abl_ld
            model_capture.blocks[l].ablate_head   = None
            model_capture.blocks[l].mean_head_out = None
            pbar.update(1)

    pbar.close()
    return imp_mean


# ── Head complementarity: cosine similarity + pairwise superadditivity ────────

@torch.no_grad()
def head_complementarity(model, dataset, indices, important_heads,
                          batch_size=256):
    """
    Returns
    -------
    cos_sim        : (n_imp, n_imp) cosine similarity matrix
    individual     : dict (l, h) -> individual ablation drop
    pair_results   : dict ((l1,h1),(l2,h2)) -> superadditivity score
    head_labels    : list of strings
    """
    model.eval()

    # ── Representational similarity ───────────────────────────────────────────
    head_reps = {(l, h): [] for (l, h) in important_heads}

    for start in range(0, len(indices), batch_size):
        idx  = indices[start:start + batch_size]
        x    = dataset.x[idx].to(DEVICE)
        v    = dataset.v[idx].to(DEVICE)
        mask = dataset.mask[idx].to(DEVICE)
        _    = model(x, v, mask)

        m = mask[:, 0, :].cpu().float()

        for (l, h) in important_heads:
            aw  = model.attn_weights[l][:, h, :, :]    # (batch, P, P) CPU
            rs  = model.residual_stream[l]              # (batch, P, 128) CPU
            m_  = m.unsqueeze(-1)
            rep = torch.einsum("bqk,bkd->bqd", aw, rs)
            rep = (rep * m_).sum(1) / m_.sum(1).clamp(min=1)
            head_reps[(l, h)].append(rep.numpy())

    for (l, h) in important_heads:
        head_reps[(l, h)] = np.concatenate(head_reps[(l, h)], axis=0)

    n_imp   = len(important_heads)
    cos_sim = np.zeros((n_imp, n_imp))
    for i, h1 in enumerate(important_heads):
        for j, h2 in enumerate(important_heads):
            r1    = head_reps[h1]
            r2    = head_reps[h2]
            norms = (np.linalg.norm(r1, axis=1, keepdims=True) *
                     np.linalg.norm(r2, axis=1, keepdims=True)).clip(min=1e-8)
            cos_sim[i, j] = ((r1 * r2).sum(axis=1) / norms.squeeze()).mean()

    head_labels = [f"L{l}H{h}" for l, h in important_heads]
    print("── Head representational similarity (cosine) ──")
    print("          " + "  ".join(f"{lb:>8s}" for lb in head_labels))
    for i, lb in enumerate(head_labels):
        print(f"{lb:8s}  " + "  ".join(f"{cos_sim[i,j]:8.3f}" for j in range(n_imp)))

    # ── Pairwise ablation superadditivity ─────────────────────────────────────
    baseline_ld, _ = logit_diff_with_uncertainty(
        model, dataset, indices[:2000])

    individual = {}
    for (l, h) in important_heads:
        block  = model.blocks[l]
        orig   = block.c_attn.data.clone()
        ab     = orig.clone(); ab[h] = 0.
        block.c_attn.data = ab
        abl_ld, _ = logit_diff_with_uncertainty(model, dataset, indices[:2000])
        individual[(l, h)] = baseline_ld - abl_ld
        block.c_attn.data  = orig

    print("\n── Pairwise ablation superadditivity ──")
    print("  (positive = complementary, negative = redundant)")
    pair_results = {}

    for i, h1 in enumerate(important_heads):
        for j, h2 in enumerate(important_heads):
            if j <= i:
                continue
            l1, hh1 = h1
            l2, hh2 = h2

            b1 = model.blocks[l1]; b2 = model.blocks[l2]
            o1 = b1.c_attn.data.clone(); o2 = b2.c_attn.data.clone()

            a1 = o1.clone(); a1[hh1] = 0.
            a2 = o2.clone(); a2[hh2] = 0.
            b1.c_attn.data = a1
            b2.c_attn.data = a2

            joint_ld, _ = logit_diff_with_uncertainty(
                model, dataset, indices[:2000])
            joint_drop = baseline_ld - joint_ld
            sum_drops  = individual[h1] + individual[h2]
            superadd   = joint_drop - sum_drops

            b1.c_attn.data = o1
            b2.c_attn.data = o2

            pair_results[(h1, h2)] = superadd
            label = ("complementary" if superadd >  0.05 else
                     "redundant"     if superadd < -0.05 else "independent")
            print(f"  L{l1}H{hh1} + L{l2}H{hh2}: "
                  f"joint={joint_drop:.3f}  sum={sum_drops:.3f}  "
                  f"superadd={superadd:+.3f}  ({label})")

    return cos_sim, individual, pair_results, head_labels


# ── Circuit AUC with bootstrap CI ────────────────────────────────────────────

@torch.no_grad()
def circuit_auc(model, dataset, circuit_set, n_layers, n_heads,
                batch_size=512):
    """
    Measure test AUC with all heads outside circuit_set masked.
    Saves and unconditionally restores c_attn via try/finally.
    """
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    model.eval()
    orig = {l: model.blocks[l].c_attn.data.clone() for l in range(n_layers)}
    for l in range(n_layers):
        ab = torch.zeros_like(model.blocks[l].c_attn.data)
        for h in range(n_heads):
            if (l, h) in circuit_set:
                ab[h] = orig[l][h]
        model.blocks[l].c_attn.data = ab

    all_probs, all_labels = [], []
    try:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        for x, v, mask, labels in loader:
            x, v, mask = x.to(DEVICE), v.to(DEVICE), mask.to(DEVICE)
            logits = model(x, v, mask).cpu()
            all_probs.append(torch.softmax(logits.float(), dim=1)[:, 1].numpy())
            all_labels.append(labels.numpy())
    finally:
        for l in range(n_layers):
            model.blocks[l].c_attn.data = orig[l]

    return roc_auc_score(np.concatenate(all_labels),
                         np.concatenate(all_probs))


def bootstrap_auc_ci(probs, labels, n_boot=1000, seed=42):
    """Return (ci_lo, ci_hi) 95% bootstrap confidence interval on AUC."""
    from sklearn.metrics import roc_auc_score
    rng   = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], probs[idx]))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


@torch.no_grad()
def auc_with_circuit_bootstrap(model, dataset, circuit_set,
                                n_layers, n_heads,
                                n_boot=1000, batch_size=512):
    """Returns (auc, ci_lo, ci_hi)."""
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    model.eval()
    orig = {l: model.blocks[l].c_attn.data.clone() for l in range(n_layers)}
    for l in range(n_layers):
        ab = torch.zeros_like(model.blocks[l].c_attn.data)
        for h in range(n_heads):
            if (l, h) in circuit_set:
                ab[h] = orig[l][h]
        model.blocks[l].c_attn.data = ab

    all_probs, all_labels = [], []
    try:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        for x, v, mask, labels in loader:
            x, v, mask = x.to(DEVICE), v.to(DEVICE), mask.to(DEVICE)
            logits = model(x, v, mask).cpu()
            all_probs.append(torch.softmax(logits.float(), dim=1)[:, 1].numpy())
            all_labels.append(labels.numpy())
    finally:
        for l in range(n_layers):
            model.blocks[l].c_attn.data = orig[l]

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc    = roc_auc_score(labels, probs)
    ci_lo, ci_hi = bootstrap_auc_ci(probs, labels, n_boot=n_boot)
    return auc, ci_lo, ci_hi


def minimality_test(model, dataset, heads_ordered, direct_effects,
                    n_layers, n_heads, n_boot=1000):
    """
    Add heads one at a time in order of recovery score.
    Returns list of dicts: head, auc, ci_lo, ci_hi.
    """
    from sklearn.metrics import roc_auc_score

    growing_circuit = set()
    results = []

    print("\nRunning minimality test...")
    print(f"{'Circuit':42s}  {'AUC':>7s}  {'95% CI':>22s}")
    print("─" * 76)

    for head in heads_ordered:
        growing_circuit.add(head)

        auc, lo, hi = auc_with_circuit_bootstrap(
            model, dataset, growing_circuit, n_layers, n_heads, n_boot=n_boot)
        results.append(dict(head=head, auc=auc, ci_lo=lo, ci_hi=hi))

        heads_str = " + ".join(f"L{l}H{h}" for l, h in sorted(growing_circuit))
        print(f"  {heads_str:40s}  {auc:.4f}  [{lo:.4f}, {hi:.4f}]")

    return results


def random_circuit_baseline(model, dataset, n_layers, n_heads,
                              circuit_size, n_random=200, seed=42):
    """
    Sample n_random circuits of circuit_size heads and evaluate AUC.
    Returns array of AUCs.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)

    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    random_aucs = []

    for _ in tqdm(range(n_random), desc="Random circuits"):
        rand_circuit = set(_random.sample(all_heads, circuit_size))
        auc = circuit_auc(model, dataset, rand_circuit, n_layers, n_heads)
        random_aucs.append(auc)

    return np.array(random_aucs)


def kinematic_regime_performance(model, dataset, indices,
                                  circuit_heads, obs_tau32, obs_mass,
                                  n_layers, n_heads, batch_size=512):
    """
    Compares full model AUC vs circuit-only AUC in kinematic regime bins.
    Returns dict of regime -> {n, full_auc, circuit_auc, retention}.
    """
    from sklearn.metrics import roc_auc_score
    model.eval()

    all_probs_full    = []
    all_probs_circuit = []
    all_labels        = []
    all_mass          = []
    all_npart         = []

    # Full model
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="Full model eval"):
            idx    = indices[start:start + batch_size]
            x      = dataset.x[idx].to(DEVICE)
            v      = dataset.v[idx].to(DEVICE)
            mask   = dataset.mask[idx].to(DEVICE)
            labels = dataset.labels[idx]
            logits = model(x, v, mask).cpu()
            all_probs_full.append(torch.softmax(logits, dim=1)[:, 1])
            all_labels.append(labels)

            v_np   = v.cpu().numpy()
            m_np   = mask[:, 0, :].cpu().numpy()
            px, py, pz, E = v_np[:,0,:], v_np[:,1,:], v_np[:,2,:], v_np[:,3,:]
            pt     = np.sqrt(px**2 + py**2).clip(min=1e-8) * m_np
            jet_E  = (E * m_np).sum(1); jet_px = (px * m_np).sum(1)
            jet_py = (py * m_np).sum(1); jet_pz = (pz * m_np).sum(1)
            jet_m  = np.sqrt((jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2).clip(0))
            all_mass.append(jet_m)
            all_npart.append(m_np.sum(1))

    # Circuit-only
    orig_vals = {}
    with torch.no_grad():
        for l in range(n_layers):
            block         = model.blocks[l]
            orig_vals[l]  = block.c_attn.data.clone()
            ablated       = torch.zeros_like(block.c_attn.data)
            for h in range(n_heads):
                if (l, h) in circuit_heads:
                    ablated[h] = orig_vals[l][h]
            block.c_attn.data = ablated

        for start in tqdm(range(0, len(indices), batch_size), desc="Circuit eval"):
            idx    = indices[start:start + batch_size]
            x      = dataset.x[idx].to(DEVICE)
            v      = dataset.v[idx].to(DEVICE)
            mask   = dataset.mask[idx].to(DEVICE)
            logits = model(x, v, mask).cpu()
            all_probs_circuit.append(torch.softmax(logits, dim=1)[:, 1])

    for l in range(n_layers):
        model.blocks[l].c_attn.data = orig_vals[l]

    probs_full    = torch.cat(all_probs_full).numpy()
    probs_circuit = torch.cat(all_probs_circuit).numpy()
    labels        = torch.cat(all_labels).numpy()
    mass          = np.concatenate(all_mass)
    npart         = np.concatenate(all_npart)
    tau32         = obs_tau32[:len(labels)]

    print("\n── Circuit performance by kinematic regime ──")
    print(f"{'Regime':30s}  {'N':>6s}  {'Full AUC':>9s}  "
          f"{'Circuit AUC':>11s}  {'Retention':>9s}")
    print("─" * 75)

    regime_results = {}

    def eval_regime(mask_regime, name):
        if mask_regime.sum() < 50:
            return
        y  = labels[mask_regime]
        pf = probs_full[mask_regime]
        pc = probs_circuit[mask_regime]
        if len(np.unique(y)) < 2:
            return
        auc_full    = roc_auc_score(y, pf)
        auc_circuit = roc_auc_score(y, pc)
        retention   = auc_circuit / auc_full
        regime_results[name] = dict(n=int(mask_regime.sum()),
                                    full_auc=auc_full,
                                    circuit_auc=auc_circuit,
                                    retention=retention)
        print(f"  {name:30s}  {mask_regime.sum():6d}  "
              f"{auc_full:9.4f}  {auc_circuit:11.4f}  {retention:9.4f}")

    q33, q67 = np.percentile(npart, [33, 67])
    eval_regime(npart <= q33,               "Sparse jets (few particles)")
    eval_regime((npart > q33) & (npart <= q67), "Medium multiplicity")
    eval_regime(npart > q67,                "Dense jets (many particles)")

    mq33, mq67 = np.percentile(mass, [33, 67])
    eval_regime(mass <= mq33,               "Jet mass (low third)")
    eval_regime((mass > mq33) & (mass <= mq67), "Jet mass (middle third)")
    eval_regime(mass > mq67,                "Jet mass (high third)")

    return regime_results
