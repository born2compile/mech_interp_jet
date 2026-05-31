"""
Jet substructure observables: N-subjettiness, jet mass, leading-pT fraction,
and the full energy correlator function suite (ECF1–4, C1, C2, C3, D2, N3).
"""

import math

import numpy as np
import fastjet
import awkward as ak
from tqdm.auto import tqdm

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
#  N-subjettiness (exact, via FastJet kT clustering)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_nsubjettiness_exact(v_batch, mask_batch, beta=1.0):
    """
    Exact n-subjettiness using exclusive kT clustering for axes.

    Parameters
    ----------
    v_batch    : np.ndarray (N, 4, P)  columns: px, py, pz, E
    mask_batch : np.ndarray (N, 1, P)  float, 1 = real particle
    beta       : float  angular exponent (IRC-safe standard = 1.0)

    Returns
    -------
    dict of np.ndarray (N,): tau1, tau2, tau3, tau21, tau32
    """
    N    = v_batch.shape[0]
    tau1 = np.zeros(N, dtype=np.float32)
    tau2 = np.zeros(N, dtype=np.float32)
    tau3 = np.zeros(N, dtype=np.float32)

    jet_def = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)

    for i in tqdm(range(N), desc="  n-subjettiness", leave=False):
        m   = mask_batch[i, 0, :].astype(bool)
        px  = v_batch[i, 0, m]
        py  = v_batch[i, 1, m]
        pz  = v_batch[i, 2, m]
        E   = v_batch[i, 3, m]

        if m.sum() < 2:
            continue

        pt   = np.sqrt(px**2 + py**2).clip(min=1e-8)
        eta  = np.arcsinh(pz / pt)
        phi  = np.arctan2(py, px)

        array   = ak.Array({"px": px.tolist(), "py": py.tolist(),
                             "pz": pz.tolist(), "E":  E.tolist()})
        cluster = fastjet.ClusterSequence(array, jet_def)

        def get_tau(n_axes):
            try:
                axes_ak  = cluster.exclusive_jets(n_jets=n_axes)
            except Exception:
                return 0.
            axes_px  = np.asarray(ak.to_numpy(axes_ak.px))
            axes_py  = np.asarray(ak.to_numpy(axes_ak.py))
            axes_pz  = np.asarray(ak.to_numpy(axes_ak.pz))
            axes_pt  = np.sqrt(axes_px**2 + axes_py**2).clip(min=1e-8)
            axes_eta = np.arcsinh(axes_pz / axes_pt)
            axes_phi = np.arctan2(axes_py, axes_px)

            d0 = pt.sum()
            if d0 < 1e-8:
                return 0.

            dr_matrix = np.stack([
                np.sqrt(
                    (eta - ae)**2 +
                    ((phi - ap + math.pi) % (2 * math.pi) - math.pi)**2
                )
                for ae, ap in zip(axes_eta, axes_phi)
            ], axis=1)

            return float((pt * dr_matrix.min(axis=1) ** beta).sum() / d0)

        tau1[i] = get_tau(1)
        tau2[i] = get_tau(2)
        tau3[i] = get_tau(3)

    tau21 = (tau2 / tau1.clip(min=1e-8)).astype(np.float32)
    tau32 = (tau3 / tau2.clip(min=1e-8)).astype(np.float32)

    return dict(tau1=tau1, tau2=tau2, tau3=tau3, tau21=tau21, tau32=tau32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Jet mass and leading-pT fraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jet_mass_and_lead(v_batch, mask_batch):
    mask       = mask_batch[:, 0, :]
    px         = v_batch[:, 0, :] * mask
    py         = v_batch[:, 1, :] * mask
    pz         = v_batch[:, 2, :] * mask
    E          = v_batch[:, 3, :] * mask

    jet_E      = E.sum(axis=1).clip(min=1e-8)
    jet_px     = px.sum(axis=1)
    jet_py     = py.sum(axis=1)
    jet_pz     = pz.sum(axis=1)
    jet_m2     = jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2
    jet_mass   = np.sqrt(jet_m2.clip(min=0)).astype(np.float32)

    pt         = np.sqrt(px**2 + py**2)
    lead_pt    = pt.max(axis=1)
    lead_pt_frac = (lead_pt / pt.sum(axis=1).clip(min=1e-8)).astype(np.float32)
    n_particles  = mask.sum(axis=1).astype(np.float32)

    return jet_mass, lead_pt_frac, n_particles


# ═══════════════════════════════════════════════════════════════════════════════
#  Energy Correlation Functions — single-jet helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ecf1(pt, beta=1.0):
    """ECF(1, beta) = sum_i pt_i.  Independent of beta."""
    return float(pt.sum())


def ecf2(pt, eta, phi, beta=1.0):
    """ECF(2, beta) = sum_{i<j} pt_i * pt_j * DeltaR_ij^beta"""
    n = len(pt)
    if n < 2:
        return 0.0
    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    dr   = np.sqrt(deta**2 + dphi**2).clip(min=EPS)
    i_idx, j_idx = np.triu_indices(n, k=1)
    return float(np.sum(pt[i_idx] * pt[j_idx] *
                         dr[i_idx, j_idx] ** beta))


def ecf3(pt, eta, phi, beta=1.0):
    """ECF(3, beta) via triple loop, with vectorised fast path for n <= 120."""
    n = len(pt)
    if n < 3:
        return 0.0
    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    dr   = np.sqrt(deta**2 + dphi**2).clip(min=EPS)
    dr_b = dr ** beta

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total += (pt[i] * pt[j] * pt[k] *
                          dr_b[i, j] * dr_b[i, k] * dr_b[j, k])
    return float(total)


def ecf3_fast(pt, eta, phi, beta=1.0):
    """
    Vectorised ECF(3, beta) — avoids the Python triple loop.
    Memory: O(n^3) floats, feasible for n <= ~150.
    Falls back to the loop version for large n.
    """
    n = len(pt)
    if n < 3:
        return 0.0
    if n > 120:
        return ecf3(pt, eta, phi, beta=beta)

    deta  = eta[:, None] - eta[None, :]
    dphi  = ((phi[:, None] - phi[None, :] + math.pi)
             % (2 * math.pi) - math.pi)
    dr    = np.sqrt(deta**2 + dphi**2).clip(min=EPS)
    dr_b  = dr ** beta

    pt3   = pt[:, None, None] * pt[None, :, None] * pt[None, None, :]  # (n,n,n)
    dr3   = (dr_b[:, :, None] *          # dr(i,j)^b
             dr_b[:, None, :] *          # dr(i,k)^b
             dr_b[None, :, :])           # dr(j,k)^b    (n,n,n)

    prod  = pt3 * dr3                    # (n, n, n)

    mask  = np.zeros((n, n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            mask[i, j, j + 1:] = True

    return float(prod[mask].sum())


def ecf4(pt, eta, phi, beta=1.0):
    """
    ECF(4, beta) = sum_{i<j<k<l} pT_i pT_j pT_k pT_l
                  * (DR_ij DR_ik DR_il DR_jk DR_jl DR_kl)^beta

    Capped at 50 leading-pT particles for tractability.
    """
    n = len(pt)
    if n < 4:
        return 0.0

    N_MAX = 50
    if n > N_MAX:
        order = np.argsort(-pt)[:N_MAX]
        pt    = pt[order]
        eta   = eta[order]
        phi   = phi[order]
        n     = N_MAX

    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    dr   = np.sqrt(deta**2 + dphi**2).clip(min=EPS)
    dr_b = dr ** beta

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dr_ij_b = dr_b[i, j]
            pt_ij   = pt[i] * pt[j]
            for k in range(j + 1, n):
                dr_ik_b  = dr_b[i, k]
                dr_jk_b  = dr_b[j, k]
                pt_ijk   = pt_ij * pt[k]
                dr_ijk_b = dr_ij_b * dr_ik_b * dr_jk_b
                for l in range(k + 1, n):
                    total += (pt_ijk * pt[l] *
                              dr_ijk_b *
                              dr_b[i, l] * dr_b[j, l] * dr_b[k, l])
    return float(total)


def ecfg_1_3(pt, eta, phi, beta=1.0):
    """
    ECFG(1, 3, beta) = ECFN(3, beta) = ECF(3,beta) / ECF(1,beta)^3
    Uses the single smallest angle among the 3 pairs per triplet.
    """
    n = len(pt)
    if n < 3:
        return 0.0
    e1 = ecf1(pt)
    if e1 < EPS:
        return 0.0

    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    dr   = np.sqrt(deta**2 + dphi**2).clip(min=EPS)

    z    = pt / e1

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                min_dr = min(dr[i, j], dr[i, k], dr[j, k])
                total += z[i] * z[j] * z[k] * (min_dr ** beta)
    return float(total)


def ecfg_2_4(pt, eta, phi, beta=1.0):
    """
    ECFG(2, 4, beta): 2-angle, 4-point generalised correlator.
    Uses product of 2 smallest angles among C(4,2)=6 pairs per quadruplet.
    Capped at 40 leading-pT particles for tractability.
    """
    n = len(pt)
    if n < 4:
        return 0.0

    N_MAX = 40
    if n > N_MAX:
        order = np.argsort(-pt)[:N_MAX]
        pt    = pt[order]
        eta   = eta[order]
        phi   = phi[order]
        n     = N_MAX

    e1 = ecf1(pt)
    if e1 < EPS:
        return 0.0

    deta = eta[:, None] - eta[None, :]
    dphi = ((phi[:, None] - phi[None, :] + math.pi)
            % (2 * math.pi) - math.pi)
    dr   = np.sqrt(deta**2 + dphi**2).clip(min=EPS)
    z    = pt / e1

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    pairs = sorted([
                        dr[i, j], dr[i, k], dr[i, l],
                        dr[j, k], dr[j, l], dr[k, l]
                    ])
                    total += (z[i] * z[j] * z[k] * z[l] *
                               (pairs[0] ** beta) * (pairs[1] ** beta))
    return float(total)


# ── Per-jet derived observables ───────────────────────────────────────────────

def compute_ecf_observables_single(px, py, pz, E, beta=1.0):
    """
    Compute ECF(1–3), C1, C2, D2 for one jet (2-prong suite).

    Parameters
    ----------
    px, py, pz, E : 1D float arrays (real particles only, no padding)
    beta           : angular exponent

    Returns
    -------
    dict with keys: ecf1, ecf2, ecf3, C1, C2, D2
    """
    pt   = np.sqrt(px**2 + py**2).clip(min=EPS)
    eta  = np.arcsinh(pz / pt)
    phi  = np.arctan2(py, px)

    e1 = ecf1(pt,           beta=beta)
    e2 = ecf2(pt, eta, phi, beta=beta)
    e3 = ecf3_fast(pt, eta, phi, beta=beta)

    denom_c1 = max(e1 ** 2,   EPS)
    denom_c2 = max(e2 ** 2,   EPS)
    denom_d2 = max(e2 ** 3,   EPS)

    C1 = e2 / denom_c1
    C2 = (e3 * e1)       / denom_c2
    D2 = (e3 * e1 ** 3)  / denom_d2

    return dict(ecf1=e1, ecf2=e2, ecf3=e3, C1=C1, C2=C2, D2=D2)


def compute_ecf_observables_full(px, py, pz, E, beta=1.0):
    """
    Compute the full ECF suite (2-prong and 3-prong) for one jet.

    Returns dict with keys:
      ecf1, ecf2, ecf3, ecf4
      C1, C2, C3              (double ratios)
      D2                      (optimal 2-prong)
      N3, ECFG_1_3, ECFG_2_4 (generalised 3-prong)
    """
    pt   = np.sqrt(px**2 + py**2).clip(min=EPS)
    eta  = np.arcsinh(pz / pt)
    phi  = np.arctan2(py, px)

    e1 = ecf1(pt,           beta=beta)
    e2 = ecf2(pt, eta, phi, beta=beta)
    e3 = ecf3(pt, eta, phi, beta=beta)
    e4 = ecf4(pt, eta, phi, beta=beta)

    denom_c1 = max(e1**2,  EPS)
    denom_c2 = max(e2**2,  EPS)
    denom_c3 = max(e3**2,  EPS)
    denom_d2 = max(e2**3,  EPS)

    C1 = e2 / denom_c1
    C2 = (e3 * e1)     / denom_c2
    C3 = (e4 * e2)     / denom_c3
    D2 = (e3 * e1**3)  / denom_d2

    g13 = ecfg_1_3(pt, eta, phi, beta=beta)
    g24 = ecfg_2_4(pt, eta, phi, beta=beta)
    N3  = g24 / max(g13**2, EPS)

    return dict(
        ecf1=e1, ecf2=e2, ecf3=e3, ecf4=e4,
        C1=C1, C2=C2, C3=C3, D2=D2,
        N3=N3, ECFG_1_3=g13, ECFG_2_4=g24,
    )


# ── Batch computation ─────────────────────────────────────────────────────────

def compute_ecf_batch(v_batch, mask_batch, beta=1.0, desc=""):
    """
    Compute 2-prong ECF suite (ecf1, ecf2, ecf3, C1, C2, D2) for a batch.

    Parameters
    ----------
    v_batch    : np.ndarray (N, 4, P)  columns: px, py, pz, E
    mask_batch : np.ndarray (N, 1, P)  float, 1 = real particle
    """
    N      = v_batch.shape[0]
    keys   = ["ecf1", "ecf2", "ecf3", "C1", "C2", "D2"]
    out    = {k: np.zeros(N, dtype=np.float32) for k in keys}

    for i in tqdm(range(N), desc=f"  ECF batch {desc}", leave=False):
        m    = mask_batch[i, 0, :].astype(bool)
        if m.sum() < 2:
            continue
        px   = v_batch[i, 0, m]
        py   = v_batch[i, 1, m]
        pz   = v_batch[i, 2, m]
        E    = v_batch[i, 3, m]

        res  = compute_ecf_observables_single(px, py, pz, E, beta=beta)
        for k in keys:
            out[k][i] = res[k]

    return out


def compute_ecf_batch_full(v_batch, mask_batch, beta=1.0, desc=""):
    """
    Compute full ECF suite (including C3, N3) for a batch.
    ECF(4) capped at 50 particles; ECFG(2,4) capped at 40.
    """
    N    = v_batch.shape[0]
    keys = ["ecf1", "ecf2", "ecf3", "ecf4",
            "C1", "C2", "C3", "D2",
            "N3", "ECFG_1_3", "ECFG_2_4"]
    out  = {k: np.zeros(N, dtype=np.float32) for k in keys}

    for i in tqdm(range(N), desc=f"  ECF full {desc}", leave=False):
        m  = mask_batch[i, 0, :].astype(bool)
        if m.sum() < 4:
            continue
        px = v_batch[i, 0, m]
        py = v_batch[i, 1, m]
        pz = v_batch[i, 2, m]
        E  = v_batch[i, 3, m]
        res = compute_ecf_observables_full(px, py, pz, E, beta=beta)
        for k in keys:
            out[k][i] = res[k]

    return out
