"""
Residual stream extraction, logit lens, jet-level and per-layer linear probes,
class-token state extraction, and the basis-rotation comparison.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .config import CFG, DEVICE


# ── Residual stream extraction ────────────────────────────────────────────────

@torch.no_grad()
def extract_residual_streams(model, dataset, indices, batch_size=256):
    """
    Returns
    -------
    jet_streams  : list[np.array(N, 128)]      length = num_layers + 1
    part_streams : list[np.array(N, P, 128)]   length = num_layers + 1
    """
    model.eval()
    n_layers     = CFG["num_layers"]
    jet_streams_ = [[] for _ in range(n_layers + 1)]
    part_streams_= [[] for _ in range(n_layers + 1)]

    pbar = tqdm(range(0, len(indices), batch_size),
                desc="Extracting residual streams")
    for start in pbar:
        idx  = indices[start:start + batch_size]
        x    = dataset.x[idx].to(DEVICE)
        v    = dataset.v[idx].to(DEVICE)
        mask = dataset.mask[idx].to(DEVICE)
        _    = model(x, v, mask)

        m = mask[:, 0, :].cpu().float()
        for l in range(n_layers + 1):
            rs      = model.residual_stream[l]
            jet_rep = ((rs * m.unsqueeze(-1)).sum(1) /
                       m.sum(1, keepdim=True).clamp(min=1))
            jet_streams_[l].append(jet_rep.numpy())
            part_streams_[l].append(rs.numpy())

        pbar.set_postfix(processed=min(start + batch_size, len(indices)))

    jet_streams  = [np.concatenate(s, axis=0) for s in jet_streams_]
    part_streams = [np.concatenate(s, axis=0) for s in part_streams_]
    return jet_streams, part_streams


# ── Class-token state extraction ──────────────────────────────────────────────

@torch.no_grad()
def extract_cls_states(model, dataset, indices, batch_size=256):
    """
    Returns a list of (N, embed_dim) arrays, one per cls_block,
    representing the class token state immediately after each
    class attention block.
    """
    model.eval()
    n_cls    = CFG["num_cls_layers"]
    collected = [[] for _ in range(n_cls)]
    hooks    = []

    for i, cls_block in enumerate(model.cls_blocks):
        def make_hook(idx):
            def hook(module, inp, output):
                # output: (1, batch, embed_dim)
                collected[idx].append(output.squeeze(0).detach().cpu())
            return hook
        hooks.append(cls_block.register_forward_hook(make_hook(i)))

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="Extracting cls states", leave=False):
        idx  = indices[start:start + batch_size]
        x    = dataset.x[idx].to(DEVICE)
        v    = dataset.v[idx].to(DEVICE)
        mask = dataset.mask[idx].to(DEVICE)
        _    = model(x, v, mask)

    for h in hooks:
        h.remove()

    return [torch.cat(c, dim=0).numpy() for c in collected]


# ── Logit lens ────────────────────────────────────────────────────────────────

@torch.no_grad()
def logit_lens(model, dataset, indices, batch_size=256):
    """
    Projects the mean-pooled residual stream at each particle-attention layer
    through the final LayerNorm + fc head, returning AUC at each depth.
    """
    model.eval()
    n_layers   = CFG["num_layers"] + 1
    all_logits = [[] for _ in range(n_layers)]
    all_labels = []

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="Logit lens", leave=False):
        idx    = indices[start:start + batch_size]
        x      = dataset.x[idx].to(DEVICE)
        v      = dataset.v[idx].to(DEVICE)
        mask   = dataset.mask[idx].to(DEVICE)
        labels = dataset.labels[idx]
        _      = model(x, v, mask)

        m = mask[:, 0, :].cpu().float()
        for l in range(n_layers):
            rs     = model.residual_stream[l].to(DEVICE)
            pooled = ((rs * m.unsqueeze(-1).to(DEVICE)).sum(1) /
                      m.sum(1, keepdim=True).to(DEVICE).clamp(min=1))
            pseudo = model.fc(model.norm(pooled))
            all_logits[l].append(pseudo.cpu())
        all_labels.append(labels)

    all_labels = torch.cat(all_labels).numpy()
    lens_aucs  = []
    for l in range(n_layers):
        logits = torch.cat(all_logits[l])
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        auc    = roc_auc_score(all_labels, probs)
        lens_aucs.append(auc)
        print(f"  Logit lens layer {l}: AUC = {auc:.4f}")

    return lens_aucs


@torch.no_grad()
def compute_logit_lens_full(model, dataset, indices, batch_size=256):
    """
    Extends logit_lens to include class attention blocks as well.
    Returns (lens_aucs_all, lens_ld_top, lens_ld_qcd, all_labels, layer_labels).
    """
    model.eval()
    n_part  = CFG["num_layers"] + 1
    n_cls   = CFG["num_cls_layers"]

    all_logits_part = [[] for _ in range(n_part)]
    cls_logits      = [[] for _ in range(n_cls)]
    cls_collected   = [[] for _ in range(n_cls)]
    all_labels      = []

    hooks = []
    for i, cls_block in enumerate(model.cls_blocks):
        def make_hook(idx):
            def hook(module, inp, output):
                cls_collected[idx].append(output.squeeze(0).detach().cpu())
            return hook
        hooks.append(cls_block.register_forward_hook(make_hook(i)))

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="Logit lens (full)", leave=False):
        idx    = indices[start:start + batch_size]
        x      = dataset.x[idx].to(DEVICE)
        v      = dataset.v[idx].to(DEVICE)
        mask   = dataset.mask[idx].to(DEVICE)
        labels = dataset.labels[idx]
        _      = model(x, v, mask)

        m = mask[:, 0, :].cpu().float()
        for l in range(n_part):
            rs     = model.residual_stream[l].to(DEVICE)
            pooled = ((rs * m.unsqueeze(-1).to(DEVICE)).sum(1) /
                      m.sum(1, keepdim=True).to(DEVICE).clamp(min=1))
            pseudo = model.fc(model.norm(pooled))
            all_logits_part[l].append(pseudo.cpu())
        all_labels.append(labels)

    for h in hooks:
        h.remove()

    for i in range(n_cls):
        rep = torch.cat(cls_collected[i], dim=0).to(DEVICE)
        pseudo = model.fc(model.norm(rep))
        cls_logits[i] = pseudo.cpu()

    all_labels = torch.cat(all_labels).numpy()
    layer_labels = ([f"L{l}" for l in range(n_part)] +
                    [f"Cls{i}" for i in range(n_cls)])

    lens_aucs = []
    lens_ld_top = []
    lens_ld_qcd = []

    # Particle attention layers
    for l in range(n_part):
        logits = torch.cat(all_logits_part[l])
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        ld     = (logits[:, 1] - logits[:, 0]).numpy()
        auc    = roc_auc_score(all_labels, probs)
        lens_aucs.append(auc)
        lens_ld_top.append(ld[all_labels == 1].mean())
        lens_ld_qcd.append(ld[all_labels == 0].mean())

    # Class attention layers
    for i in range(n_cls):
        logits = cls_logits[i]
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        ld     = (logits[:, 1] - logits[:, 0]).numpy()
        auc    = roc_auc_score(all_labels, probs)
        lens_aucs.append(auc)
        lens_ld_top.append(ld[all_labels == 1].mean())
        lens_ld_qcd.append(ld[all_labels == 0].mean())

    return lens_aucs, lens_ld_top, lens_ld_qcd, all_labels, layer_labels


# ── Jet-level linear probes ───────────────────────────────────────────────────

def run_jet_level_probes(jet_streams, probe_targets, split=7000):
    """
    For each layer in jet_streams, fit a Ridge (continuous) or Logistic
    (binary) probe to predict each observable.

    Parameters
    ----------
    jet_streams    : list[np.array(N, 128)]
    probe_targets  : dict name -> np.array(N,)
    split          : train/test split index

    Returns
    -------
    results : dict name -> list of scores (one per layer)
    """
    results = {name: [] for name in probe_targets}

    for name, target in probe_targets.items():
        y_tr = target[:split]
        y_te = target[split:]
        pbar = tqdm(range(len(jet_streams)),
                    desc=f"  {name:15s}", leave=False)
        for l in pbar:
            X_tr = jet_streams[l][:split]
            X_te = jet_streams[l][split:]

            if name == "is_top":
                pipe = Pipeline([("sc",  StandardScaler()),
                                 ("clf", LogisticRegression(max_iter=1000, C=1.))])
            else:
                pipe = Pipeline([("sc",  StandardScaler()),
                                 ("reg", Ridge(alpha=1.))])

            pipe.fit(X_tr, y_tr)
            score = pipe.score(X_te, y_te)
            results[name].append(score)
            pbar.set_postfix(layer=l, score=f"{score:.4f}")

    return results


# ── Per-layer trained logistic probe (basis-rotation analysis) ────────────────

def run_per_layer_probes(all_reps, labels, split=7000):
    """
    Fit one logistic regression probe per representation layer.
    Returns list of AUC values, one per entry in all_reps.

    Parameters
    ----------
    all_reps : list[np.array(N, embed_dim)]
    labels   : np.array(N,)  binary
    """
    probe_aucs = []
    y_tr = labels[:split].astype(np.float32)
    y_te = labels[split:]

    for rep in tqdm(all_reps, desc="Per-layer probe"):
        X_tr = rep[:split]
        X_te = rep[split:]

        pipe = Pipeline([
            ('sc',  StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, C=1.,
                solver='lbfgs', random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, probs)
        probe_aucs.append(auc)

    return probe_aucs


# ── Particle-level probes ─────────────────────────────────────────────────────

def compute_particle_level_targets(v_batch, mask_batch):
    """
    Returns per-particle targets, each (N, P):
      pt_rank_norm : pT rank normalised to [0, 1]
      delta_r_norm : ΔR from jet axis normalised to [0, 1]
      is_leading   : 1 if hardest particle in jet
      is_top3      : 1 if in top-3 by pT
    """
    import math as _math
    N, _, P = v_batch.shape
    mask    = mask_batch[:, 0, :]

    px = v_batch[:, 0, :]
    py = v_batch[:, 1, :]
    pz = v_batch[:, 2, :]
    E  = v_batch[:, 3, :]

    pt  = np.sqrt(px**2 + py**2).clip(min=1e-8) * mask
    eta = np.arcsinh(pz / pt.clip(min=1e-8))
    phi = np.arctan2(py, px)

    jet_pt  = pt.sum(axis=1, keepdims=True).clip(min=1e-8)
    jet_eta = (eta * pt).sum(axis=1, keepdims=True) / jet_pt
    jet_phi = np.arctan2(
        (np.sin(phi) * pt).sum(axis=1, keepdims=True),
        (np.cos(phi) * pt).sum(axis=1, keepdims=True))

    deta    = eta - jet_eta
    dphi    = (phi - jet_phi + _math.pi) % (2 * _math.pi) - _math.pi
    dr      = np.sqrt(deta**2 + dphi**2) * mask

    pt_for_rank   = pt + 1e9 * (1 - mask)
    ranks         = np.argsort(np.argsort(-pt_for_rank, axis=1), axis=1)
    n_real        = mask.sum(axis=1, keepdims=True).clip(min=1)
    pt_rank_norm  = (ranks / n_real).astype(np.float32) * mask

    dr_max   = dr.max(axis=1, keepdims=True).clip(min=1e-8)
    dr_norm  = (dr / dr_max).astype(np.float32)

    is_leading = (ranks == 0).astype(np.float32) * mask
    is_top3    = (ranks <= 2).astype(np.float32) * mask

    return {
        "pt_rank_norm": pt_rank_norm,
        "delta_r_norm": dr_norm,
        "is_leading"  : is_leading,
        "is_top3"     : is_top3,
    }


def probe_particle_level(part_streams, particle_targets, mask_np, split=7000):
    """
    For each layer, probe (N*P_real, 128) features for per-particle targets.
    Uses only real particles (mask = 1).
    """
    results  = {k: [] for k in particle_targets}
    n_layers = len(part_streams)

    for name, target in particle_targets.items():
        pbar = tqdm(range(n_layers),
                    desc=f"  Particle probe: {name:15s}", leave=False)
        for l in pbar:
            rs       = part_streams[l]           # (N, P, 128)
            m        = mask_np[:, 0, :]           # (N, P)
            idx_n, idx_p = np.where(m > 0)
            X_flat   = rs[idx_n, idx_p, :]
            y_flat   = target[idx_n, idx_p]

            train_jet = idx_n < split
            test_jet  = idx_n >= split
            X_tr, y_tr = X_flat[train_jet], y_flat[train_jet]
            X_te, y_te = X_flat[test_jet],  y_flat[test_jet]

            if name in ["is_leading", "is_top3"]:
                pipe = Pipeline([("sc",  StandardScaler()),
                                 ("clf", LogisticRegression(
                                     max_iter=500, C=1.,
                                     class_weight="balanced"))])
            else:
                pipe = Pipeline([("sc",  StandardScaler()),
                                 ("reg", Ridge(alpha=1.))])

            pipe.fit(X_tr, y_tr)
            score = pipe.score(X_te, y_te)
            results[name].append(score)
            pbar.set_postfix(layer=l, score=f"{score:.4f}")

    return results


# ── Mass-residualization control ──────────────────────────────────────────────

def residualize_against_mass(y, x_mass, split, degree=3):
    """
    Fit f(mass) -> y on training split using polynomial regression.
    Return residuals y - f(mass) on full array.
    """
    from sklearn.preprocessing import PolynomialFeatures

    x_tr = x_mass[:split].reshape(-1, 1)
    y_tr = y[:split]

    p1, p99  = np.percentile(y_tr[np.isfinite(y_tr)], [1, 99])
    y_tr_c   = np.clip(y_tr, p1, p99)

    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('sc',   StandardScaler()),
        ('reg',  Ridge(alpha=1.)),
    ])
    pipe.fit(x_tr, y_tr_c)

    y_pred    = pipe.predict(x_mass.reshape(-1, 1))
    residuals = y - y_pred
    r2_mass   = pipe.score(x_mass[split:].reshape(-1, 1),
                           y[split:])
    return residuals, r2_mass


def run_mass_residualized_probes(all_reps, obs_vals, mass_vals,
                                  split=7000, degree=3):
    """
    For each layer, probe the residual of obs_vals after removing mass.

    Returns
    -------
    r2_raw    : list of R2 for raw observable
    r2_resid  : list of R2 for mass-residualized observable
    """
    residuals, _ = residualize_against_mass(obs_vals, mass_vals, split, degree)

    r2_raw   = []
    r2_resid = []

    for rep in tqdm(all_reps, desc="Mass-residualized probe", leave=False):
        X_tr = rep[:split]
        X_te = rep[split:]

        for y_all, store in [(obs_vals, r2_raw), (residuals, r2_resid)]:
            y_tr = y_all[:split]
            y_te = y_all[split:]
            pipe = Pipeline([("sc",  StandardScaler()),
                              ("reg", Ridge(alpha=1.))])
            pipe.fit(X_tr, y_tr)
            store.append(pipe.score(X_te, y_te))

    return r2_raw, r2_resid
