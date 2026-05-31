"""
Interaction feature attribution (Pearson correlations of attention vs pairwise
features), causal feature ablation, and ECF–attention correlation analyses
(2-prong and 3-prong discriminating power per circuit head).
"""

import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DEVICE
from .model import pairwise_lv_fts

EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
#  Interaction feature attribution
# ═══════════════════════════════════════════════════════════════════════════════

def interaction_feature_attribution(model, dataset, indices,
                                     important_heads, batch_size=64):
    """
    For each important head (l, h), measures Pearson correlation between
    each of the 4 raw interaction features and the attention weights.

    Returns
    -------
    correlations : dict (l, h) -> list of 4 Pearson r values
                   order: [ln kT, ln z, ln Δ, ln m²]
    feat_names   : list of 4 strings
    """
    model.eval()
    FEAT_NAMES = ["ln kT", "ln z", "ln Δ", "ln m²"]

    head_pairs = {(l, h): {f: {"feat": [], "aw": []} for f in FEAT_NAMES}
                  for (l, h) in important_heads}

    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size),
                          desc="Interaction attribution", leave=False):
            idx  = indices[start:start + batch_size]
            x    = dataset.x[idx].to(DEVICE)
            v    = dataset.v[idx].to(DEVICE)
            mask = dataset.mask[idx].to(DEVICE)
            _    = model(x, v, mask)

            B    = v.size(0)
            P    = v.size(2)
            real = mask[:, 0, :].bool()

            v_i     = v.unsqueeze(-1).expand(-1, -1, -1, P)
            v_j     = v.unsqueeze(-2).expand(-1, -1, P, -1)
            vi_flat = v_i.permute(0, 2, 3, 1).reshape(-1, 4)
            vj_flat = v_j.permute(0, 2, 3, 1).reshape(-1, 4)
            feats   = pairwise_lv_fts(vi_flat, vj_flat, num_outputs=4)
            feats   = feats.reshape(B, P, P, 4).cpu().numpy()

            for (l, h) in important_heads:
                aw = model.attn_weights[l][:, h, :, :].numpy()

                for bi in range(B):
                    m_bi     = real[bi].cpu().numpy()
                    idx_real = np.where(m_bi)[0]
                    n        = len(idx_real)
                    if n < 2:
                        continue

                    aw_sub   = aw[bi][np.ix_(idx_real, idx_real)]
                    feat_sub = feats[bi][np.ix_(idx_real, idx_real)]
                    off      = ~np.eye(n, dtype=bool)
                    aw_flat  = aw_sub[off]

                    for fi, fname in enumerate(FEAT_NAMES):
                        f_flat = feat_sub[:, :, fi][off]
                        valid  = np.isfinite(f_flat) & np.isfinite(aw_flat)
                        if valid.sum() < 10:
                            continue
                        head_pairs[(l, h)][fname]["feat"].append(f_flat[valid])
                        head_pairs[(l, h)][fname]["aw"].append(aw_flat[valid])

    # ── Compute Pearson r ─────────────────────────────────────────────────────
    correlations = {}
    print(f"\n── Interaction feature attribution ──")
    print(f"  {'Head':10s}" + "".join(f"  {n:>10s}" for n in FEAT_NAMES))
    print(f"  {'-'*55}")

    for (l, h) in important_heads:
        corrs = []
        for fname in FEAT_NAMES:
            f_all  = np.concatenate(head_pairs[(l, h)][fname]["feat"])
            aw_all = np.concatenate(head_pairs[(l, h)][fname]["aw"])
            assert len(f_all) == len(aw_all)
            r = np.corrcoef(f_all, aw_all)[0, 1]
            corrs.append(float(r))
        correlations[(l, h)] = corrs
        print(f"  L{l}H{h}       " +
              "".join(f"  {r:+10.4f}" for r in corrs))

    return correlations, FEAT_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
#  Causal feature ablation
# ═══════════════════════════════════════════════════════════════════════════════

FEAT_COL = {'ln_delta': 2, 'ln_kT': 0, 'ln_z': 1, 'ln_m2': 3}


def make_zeroed_pair_embed_forward(pair_embed, zero_col):
    """
    Returns a patched pairwise_lv_fts that zeros column zero_col
    of the feature tensor before it enters the CNN.
    """
    original_plv = pair_embed.pairwise_lv_fts

    def patched_plv(xi, xj):
        result = original_plv(xi, xj).clone()
        if result.dim() == 3:
            result[:, zero_col, :] = 0.0   # (B, 4, n_pairs)
        else:
            result[:, zero_col]    = 0.0   # (N, 4)
        return result

    return patched_plv, original_plv


@torch.no_grad()
def collect_profiles_circuit(model, dataset, indices, tracked_heads,
                              zero_col=None, batch_size=64):
    """
    Run model on dataset[indices].
    If zero_col is not None, zero that column of pairwise_lv_fts output.

    Returns (ld_mean, profiles).
    profiles: dict (l,h) -> {'top': (N_BINS,), 'qcd': (N_BINS,)}
    """
    DR_BINS = np.array([0, .05, .1, .15, .2, .25, .3, .35, .4,
                         .5, .6, .8, 1.0, 1.2, 1.6])
    BC      = 0.5 * (DR_BINS[:-1] + DR_BINS[1:])
    N_BINS  = len(DR_BINS) - 1

    model.eval()
    original_plv = model.pair_embed.pairwise_lv_fts

    if zero_col is not None:
        patched_plv, _ = make_zeroed_pair_embed_forward(model.pair_embed, zero_col)
        model.pair_embed.pairwise_lv_fts = patched_plv

    acc = {(l, h): {
        tag: {'tot': np.zeros(N_BINS), 'cnt': np.zeros(N_BINS)}
        for tag in ['top', 'qcd']
    } for (l, h) in tracked_heads}

    all_ld = []

    try:
        for start in tqdm(range(0, len(indices), batch_size),
                          desc=f"  col={zero_col}", leave=False):
            idx    = indices[start:start + batch_size]
            x      = dataset.x[idx].to(DEVICE)
            v      = dataset.v[idx].to(DEVICE)
            mask_  = dataset.mask[idx].to(DEVICE)
            labels = dataset.labels[idx].numpy()

            logits = model(x, v, mask_).cpu()
            ld     = logits[:, 1] - logits[:, 0]
            all_ld.append(ld[labels == 1])

            v_np    = v.cpu().numpy()
            m_np    = mask_[:, 0, :].cpu().numpy()
            px, py, pz = v_np[:,0,:], v_np[:,1,:], v_np[:,2,:]
            pt   = np.sqrt(px**2 + py**2).clip(min=EPS)
            eta  = np.arcsinh(pz / pt)
            phi  = np.arctan2(py, px)
            deta = eta[:, :, None] - eta[:, None, :]
            dphi = ((phi[:, :, None] - phi[:, None, :] + math.pi)
                    % (2 * math.pi) - math.pi)
            dr   = np.sqrt(deta**2 + dphi**2)

            for (l, h) in tracked_heads:
                aw = model.attn_weights[l][:, h, :, :].numpy()
                for bi in range(len(labels)):
                    m_bi = m_np[bi].astype(bool)
                    ir   = np.where(m_bi)[0]
                    n    = len(ir)
                    if n < 2:
                        continue
                    aw_s = aw[bi][np.ix_(ir, ir)]
                    dr_s = dr[bi][np.ix_(ir, ir)]
                    off  = ~np.eye(n, dtype=bool)
                    af   = aw_s[off]
                    df   = dr_s[off]
                    tag  = 'top' if labels[bi] == 1 else 'qcd'
                    for b in range(N_BINS):
                        ib = (df >= DR_BINS[b]) & (df < DR_BINS[b+1])
                        if ib.sum() > 0:
                            acc[(l,h)][tag]['tot'][b] += af[ib].sum()
                            acc[(l,h)][tag]['cnt'][b] += ib.sum()
    finally:
        model.pair_embed.pairwise_lv_fts = original_plv

    profiles = {}
    for (l, h) in tracked_heads:
        profiles[(l, h)] = {}
        for tag in ['top', 'qcd']:
            t = acc[(l,h)][tag]['tot']
            c = acc[(l,h)][tag]['cnt']
            profiles[(l,h)][tag] = np.where(c > 0, t / c, np.nan)

    ld_mean = (torch.cat(all_ld).mean().item()
               if len(all_ld) > 0 else np.nan)
    return ld_mean, profiles, BC


# ═══════════════════════════════════════════════════════════════════════════════
#  ECF–attention correlation (2-prong and 3-prong discriminating power)
# ═══════════════════════════════════════════════════════════════════════════════

def _pairwise_dr(eta, phi):
    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    return np.sqrt(deta**2 + dphi**2).clip(min=EPS)


def compute_pair_ecf2_weight_norm(pt, eta, phi, beta=1.0):
    """Normalised pairwise ECF2 contribution: c~_{ij} = pT_i pT_j DR^b / ECF1^2"""
    n   = len(pt)
    e1  = pt.sum()
    if e1 < EPS or n < 2:
        return np.zeros((n, n))
    dr   = _pairwise_dr(eta, phi)
    w2   = pt[:, None] * pt[None, :] * dr**beta / (e1**2 + EPS)
    np.fill_diagonal(w2, 0.0)
    return w2


def compute_pair_ecf3_weight(pt, eta, phi, beta=1.0):
    """
    Pairwise ECF3 marginal weight:
      w3_{ij} = (pT_i pT_j DR_ij^b / ECF1^3)
                * sum_{k != i,j} pT_k DR_ik^b DR_jk^b
    """
    n    = len(pt)
    e1   = pt.sum()
    if e1 < EPS or n < 3:
        return np.zeros((n, n))

    dr   = _pairwise_dr(eta, phi)
    dr_b = dr ** beta

    dr_b_offdiag = dr_b.copy()
    np.fill_diagonal(dr_b_offdiag, 0.0)

    weighted = dr_b_offdiag * pt[None, :]
    S        = weighted @ dr_b_offdiag.T

    w3 = (pt[:, None] * pt[None, :] *
          dr_b_offdiag * S) / (e1**3 + EPS)
    w3 = (w3 + w3.T) / 2.0
    np.fill_diagonal(w3, 0.0)
    return w3


@torch.no_grad()
def attn_vs_normalised_ecf2(model, dataset, indices, important_heads,
                              beta=1.0, batch_size=64):
    """
    For each circuit head, compute Pearson correlation between attention
    weight A_{ij} and normalised pairwise ECF2 contribution c~_{ij}.

    Returns
    -------
    results : dict (l,h) -> dict with r_top, r_qcd, delta, aw_top/qcd, ec_top/qcd
    """
    model.eval()

    store = {(l, h): {
        "top": {"aw": [], "ec": []},
        "qcd": {"aw": [], "ec": []},
    } for (l, h) in important_heads}

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="Building attn vs ECF2 arrays",
                      leave=False):
        idx    = indices[start:start + batch_size]
        x      = dataset.x[idx].to(DEVICE)
        v      = dataset.v[idx].to(DEVICE)
        mask   = dataset.mask[idx].to(DEVICE)
        labels = dataset.labels[idx].numpy()
        _      = model(x, v, mask)

        v_np    = v.cpu().numpy()
        mask_np = mask[:, 0, :].cpu().numpy()

        for bi in range(len(labels)):
            m_bi     = mask_np[bi].astype(bool)
            idx_real = np.where(m_bi)[0]
            n        = len(idx_real)
            if n < 2:
                continue

            px  = v_np[bi, 0, idx_real]
            py  = v_np[bi, 1, idx_real]
            pz  = v_np[bi, 2, idx_real]
            pt  = np.sqrt(px**2 + py**2).clip(min=EPS)
            eta = np.arcsinh(pz / pt)
            phi = np.arctan2(py, px)

            ec_norm = compute_pair_ecf2_weight_norm(pt, eta, phi, beta=beta)
            off     = ~np.eye(n, dtype=bool)
            ec_flat = ec_norm[off]
            tag     = "top" if labels[bi] == 1 else "qcd"

            for (l, h) in important_heads:
                aw_full = model.attn_weights[l][bi, h, :, :].numpy()
                aw_sub  = aw_full[np.ix_(idx_real, idx_real)]
                aw_flat = aw_sub[off]

                valid = np.isfinite(aw_flat) & np.isfinite(ec_flat)
                if valid.sum() < 10:
                    continue
                store[(l, h)][tag]["aw"].append(aw_flat[valid])
                store[(l, h)][tag]["ec"].append(ec_flat[valid])

    results = {}
    print(f"\n── Attention vs normalised ECF2 contribution  (beta={beta:.1f}) ──")
    print(f"  {'Head':8s}  {'r_top':>8s}  {'r_QCD':>8s}  "
          f"{'delta':>8s}  {'interpretation':}")
    print("─" * 68)

    for (l, h) in important_heads:
        r_vals = {}
        raw    = {}

        for tag in ["top", "qcd"]:
            aw_all = (np.concatenate(store[(l,h)][tag]["aw"])
                     if store[(l,h)][tag]["aw"] else np.array([]))
            ec_all = (np.concatenate(store[(l,h)][tag]["ec"])
                     if store[(l,h)][tag]["ec"] else np.array([]))

            r_vals[tag] = (float(np.corrcoef(aw_all, ec_all)[0, 1])
                           if len(aw_all) >= 20 else np.nan)
            raw[f"aw_{tag}"] = aw_all
            raw[f"ec_{tag}"] = ec_all

        r_top = r_vals.get("top", np.nan)
        r_qcd = r_vals.get("qcd", np.nan)
        delta = (r_top - r_qcd
                 if not (np.isnan(r_top) or np.isnan(r_qcd))
                 else np.nan)

        if np.isnan(delta):
            interp = "insufficient data"
        elif delta > 0.05:
            interp = "discriminating  (attends to ECF2 more in top)"
        elif delta < -0.05:
            interp = "anti-discriminating  (attends more in QCD)"
        else:
            interp = "class-agnostic"

        print(f"  L{l}H{h}      "
              f"  {r_top:8.4f}"
              f"  {r_qcd:8.4f}"
              f"  {delta:8.4f}"
              f"  {interp}")

        results[(l, h)] = dict(r_top=r_top, r_qcd=r_qcd, delta=delta, **raw)

    return results


@torch.no_grad()
def attn_vs_ecf_3prong(model, dataset, indices, important_heads,
                        beta=1.0, batch_size=64):
    """
    For each circuit head, compute Pearson correlation between attention
    weight A_{ij} and both the 2-prong proxy (c~2_{ij}) and the 3-prong
    proxy (w3_{ij}), separately for top and QCD jets.

    Returns
    -------
    results : dict (l,h) -> dict with
                r2_top, r2_qcd, delta2,
                r3_top, r3_qcd, delta3
    """
    model.eval()

    store = {(l, h): {
        "top": {"aw": [], "w2": [], "w3": []},
        "qcd": {"aw": [], "w2": [], "w3": []},
    } for (l, h) in important_heads}

    for start in tqdm(range(0, len(indices), batch_size),
                      desc="ECF3 attention correlation",
                      leave=False):
        idx    = indices[start:start + batch_size]
        x      = dataset.x[idx].to(DEVICE)
        v      = dataset.v[idx].to(DEVICE)
        mask   = dataset.mask[idx].to(DEVICE)
        labels = dataset.labels[idx].numpy()
        _      = model(x, v, mask)

        v_np    = v.cpu().numpy()
        mask_np = mask[:, 0, :].cpu().numpy()

        for bi in range(len(labels)):
            m_bi     = mask_np[bi].astype(bool)
            idx_real = np.where(m_bi)[0]
            n        = len(idx_real)
            if n < 3:
                continue

            px  = v_np[bi, 0, idx_real]
            py  = v_np[bi, 1, idx_real]
            pz  = v_np[bi, 2, idx_real]
            pt  = np.sqrt(px**2 + py**2).clip(min=EPS)
            eta = np.arcsinh(pz / pt)
            phi = np.arctan2(py, px)

            w2_mat = compute_pair_ecf2_weight_norm(pt, eta, phi, beta=beta)
            w3_mat = compute_pair_ecf3_weight(pt, eta, phi, beta=beta)

            off    = ~np.eye(n, dtype=bool)
            w2_flat= w2_mat[off]
            w3_flat= w3_mat[off]

            tag = "top" if labels[bi] == 1 else "qcd"

            for (l, h) in important_heads:
                aw_full = model.attn_weights[l][bi, h, :, :].numpy()
                aw_sub  = aw_full[np.ix_(idx_real, idx_real)]
                aw_flat = aw_sub[off]

                valid2  = np.isfinite(aw_flat) & np.isfinite(w2_flat)
                valid3  = np.isfinite(aw_flat) & np.isfinite(w3_flat)
                valid   = valid2 & valid3

                if valid.sum() < 10:
                    continue

                store[(l,h)][tag]["aw"].append(aw_flat[valid])
                store[(l,h)][tag]["w2"].append(w2_flat[valid])
                store[(l,h)][tag]["w3"].append(w3_flat[valid])

    results = {}
    print(f"\n── Attention vs ECF2 and ECF3 pairwise weights  (beta={beta:.1f}) ──\n")
    print(f"  {'Head':8s}  {'r2_top':>8s}  {'r2_qcd':>8s}  {'d2':>7s}  "
          f"{'r3_top':>8s}  {'r3_qcd':>8s}  {'d3':>7s}  "
          f"{'3-prong > 2-prong?':}")
    print("─" * 85)

    for (l, h) in important_heads:
        raw = {}
        for tag in ["top", "qcd"]:
            s = store[(l,h)][tag]
            if len(s["aw"]) == 0:
                raw[tag] = dict(r2=np.nan, r3=np.nan,
                                aw=np.array([]),
                                w2=np.array([]),
                                w3=np.array([]))
                continue
            aw = np.concatenate(s["aw"])
            w2 = np.concatenate(s["w2"])
            w3 = np.concatenate(s["w3"])
            r2 = float(np.corrcoef(aw, w2)[0, 1])
            r3 = float(np.corrcoef(aw, w3)[0, 1])
            raw[tag] = dict(r2=r2, r3=r3, aw=aw, w2=w2, w3=w3)

        r2t = raw["top"]["r2"];  r2q = raw["qcd"]["r2"]
        r3t = raw["top"]["r3"];  r3q = raw["qcd"]["r3"]
        d2  = (r2t - r2q if not (np.isnan(r2t) or np.isnan(r2q)) else np.nan)
        d3  = (r3t - r3q if not (np.isnan(r3t) or np.isnan(r3q)) else np.nan)

        three_wins = (not np.isnan(d3) and not np.isnan(d2) and
                      abs(d3) > abs(d2))
        label = ("YES — 3-prong more discriminating" if three_wins else
                 ("EQUAL" if abs(d3 - d2) < 0.005 else
                  "NO  — 2-prong more discriminating"))

        print(f"  L{l}H{h}      "
              f"  {r2t:8.4f}  {r2q:8.4f}  {d2:+7.4f}"
              f"  {r3t:8.4f}  {r3q:8.4f}  {d3:+7.4f}"
              f"  {label}")

        results[(l,h)] = dict(
            r2_top=r2t, r2_qcd=r2q, delta2=d2,
            r3_top=r3t, r3_qcd=r3q, delta3=d3,
            **{f"aw_{tag}": raw[tag]["aw"]  for tag in ["top","qcd"]},
            **{f"w2_{tag}": raw[tag]["w2"]  for tag in ["top","qcd"]},
            **{f"w3_{tag}": raw[tag]["w3"]  for tag in ["top","qcd"]},
        )

    return results
