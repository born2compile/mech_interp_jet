"""
Interaction feature attribution and ECF–attention correlation analysis.
Produces Figures 13, 14 (causal feature ablation, 2-prong vs 3-prong δ).

Usage
-----
python scripts/07_interaction_features.py --checkpoint small_part_seed0.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.config               import CFG, DEVICE, DATA_DIR
from src.dataset              import build_dataloaders
from src.model                import SmallParT
from src.interaction_features import (interaction_feature_attribution,
                                       collect_profiles_circuit,
                                       attn_vs_normalised_ecf2,
                                       attn_vs_ecf_3prong,
                                       FEAT_COL)
from src.plotting             import (plot_feature_ablation,
                                       plot_ecf_attn_bars,
                                       plot_ecf_2v3_discriminating)


# Hard-coded circuit heads (source of truth from the paper)
CIRCUIT_HEADS = [(0, 1), (1, 3), (0, 2), (1, 0), (1, 1), (3, 3)]
FEAT_NAMES    = ['ln_delta', 'ln_kT', 'ln_z', 'ln_m2']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="small_part_seed0.pt")
    parser.add_argument("--data",        default=DATA_DIR)
    parser.add_argument("--n_jets",      type=int, default=2_000)
    parser.add_argument("--n_feat",      type=int, default=10_000)
    args = parser.parse_args()

    _, _, test_ds, _, _, _ = build_dataloaders(args.data)

    model = SmallParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    # ── 1. Pearson correlation: attention vs pairwise features ────────────────
    print("=== Interaction feature attribution ===")
    correlations, feat_names = interaction_feature_attribution(
        model, test_ds, np.arange(args.n_jets), CIRCUIT_HEADS[:4])

    # ── 2. Causal feature ablation ────────────────────────────────────────────
    print("\n=== Causal feature ablation ===")
    probe_feat_idx = np.arange(args.n_feat)

    print("Baseline ...")
    baseline_ld, baseline_profiles, BC = collect_profiles_circuit(
        model, test_ds, probe_feat_idx, CIRCUIT_HEADS, zero_col=None)
    print(f"Baseline LD: {baseline_ld:.4f}")

    feat_results = {}
    for fname in FEAT_NAMES:
        col = FEAT_COL[fname]
        print(f"Ablating {fname} (col {col}) ...")
        abl_ld, abl_profiles, _ = collect_profiles_circuit(
            model, test_ds, probe_feat_idx, CIRCUIT_HEADS, zero_col=col)
        ld_drop = baseline_ld - abl_ld
        feat_results[fname] = {
            'ld_drop' : ld_drop,
            'abl_ld'  : abl_ld,
            'profiles': abl_profiles,
        }
        print(f"  LD drop: {ld_drop:+.4f}")

    # We need imp_mean for the figure title; use zeros if not available
    try:
        from src.ablation import zero_ablation_sweep
        imp_mean, _ = zero_ablation_sweep(
            model, test_ds, np.arange(2000),
            CFG["num_layers"], CFG["num_heads"], n_bootstrap=100)
    except Exception:
        imp_mean = np.zeros((CFG["num_layers"], CFG["num_heads"]))

    plot_feature_ablation(baseline_profiles, feat_results, CIRCUIT_HEADS,
                           imp_mean, BC)

    # ── 3. ECF2 attention correlation ─────────────────────────────────────────
    print("\n=== ECF2 attention correlation ===")
    ecf2_attn = attn_vs_normalised_ecf2(
        model, test_ds, np.arange(args.n_jets), CIRCUIT_HEADS, beta=1.0)
    plot_ecf_attn_bars(ecf2_attn, CIRCUIT_HEADS)

    # ── 4. ECF3 (3-prong) attention correlation ───────────────────────────────
    print("\n=== ECF3 (3-prong) attention correlation ===")
    ecf3_attn = attn_vs_ecf_3prong(
        model, test_ds, np.arange(args.n_jets), CIRCUIT_HEADS, beta=1.0)
    plot_ecf_2v3_discriminating(ecf3_attn, CIRCUIT_HEADS)

    print("\nAll interaction-feature figures saved.")


if __name__ == "__main__":
    main()
