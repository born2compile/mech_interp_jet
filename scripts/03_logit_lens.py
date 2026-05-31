"""
Logit lens and per-layer trained logistic probe analysis.
Produces Figures 2, 3, 4 (logit lens AUC, mean LD trajectory, basis-rotation).

Usage
-----
python scripts/03_logit_lens.py --checkpoint small_part_seed0.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.config   import CFG, DEVICE, DATA_DIR
from src.dataset  import build_dataloaders
from src.model    import SmallParT
from src.probing  import (extract_residual_streams, extract_cls_states,
                           compute_logit_lens_full, run_per_layer_probes)
from src.plotting import (plot_logit_lens_auc, plot_logit_lens_ld,
                           plot_lens_vs_probe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="small_part_seed0.pt")
    parser.add_argument("--data",       default=DATA_DIR)
    parser.add_argument("--n_jets",     type=int, default=10_000)
    args = parser.parse_args()

    _, _, test_ds, _, _, test_loader = build_dataloaders(args.data)

    model = SmallParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    # Evaluate full-model AUC
    from src.training import evaluate
    _, _, te_auc, _, _ = evaluate(model, test_loader, desc="test")
    print(f"Full model test AUC: {te_auc:.4f}")

    n = min(args.n_jets, len(test_ds))
    probe_idx = np.arange(n)
    labels    = test_ds.labels[:n].numpy()

    # ── Logit lens (full, including class attention) ──────────────────────────
    print("\nRunning logit lens ...")
    (lens_aucs, lens_ld_top, lens_ld_qcd,
     _, layer_labels) = compute_logit_lens_full(model, test_ds, probe_idx)

    print(f"\nLogit lens AUC by layer:")
    for lbl, auc in zip(layer_labels, lens_aucs):
        print(f"  {lbl:6s}: {auc:.4f}")

    # ── Per-layer trained probes ──────────────────────────────────────────────
    print("\nExtracting residual streams ...")
    jet_streams, _ = extract_residual_streams(model, test_ds, probe_idx)

    print("\nExtracting class token states ...")
    cls_states = extract_cls_states(model, test_ds, probe_idx)

    all_reps = jet_streams + cls_states

    print("\nTraining per-layer logistic probes ...")
    probe_aucs = run_per_layer_probes(all_reps, labels, split=7000)

    print(f"\n── Logit lens vs per-layer probe AUC ──")
    print(f"  {'Layer':6s}  {'Logit lens':>12s}  {'Probe AUC':>10s}  {'Diff':>8s}")
    print("─" * 45)
    for lbl, ll, pr in zip(layer_labels, lens_aucs, probe_aucs):
        print(f"  {lbl:6s}  {ll:12.4f}  {pr:10.4f}  {pr-ll:+8.4f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    n_part = CFG["num_layers"] + 1
    plot_logit_lens_auc(lens_aucs, te_auc)
    plot_logit_lens_ld(lens_ld_top, lens_ld_qcd)
    plot_lens_vs_probe(lens_aucs, probe_aucs, te_auc, layer_labels)

    print("\nFigures saved.")


if __name__ == "__main__":
    main()
