"""
Circuit identification pipeline:
  1. Zero-ablation importance ranking
  2. Path patching (direct effects + path effects)
  3. Minimality test with bootstrap CI
  4. Random-circuit baseline comparison

Usage
-----
python scripts/04_circuit_identification.py --checkpoint small_part_seed0.pt
python scripts/04_circuit_identification.py --checkpoint small_part_seed0.pt \
    --strategy permutation
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from src.config    import CFG, DEVICE, DATA_DIR
from src.dataset   import build_dataloaders
from src.model     import SmallParT
from src.ablation  import (zero_ablation_sweep, head_complementarity,
                            minimality_test, random_circuit_baseline,
                            kinematic_regime_performance)
from src.patching  import PatchableParT, path_patch_sweep
from src.plotting  import (plot_importance_heatmap, plot_direct_effect_matrix,
                            plot_circuit, plot_minimality, plot_random_baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="small_part_seed0.pt")
    parser.add_argument("--data",        default=DATA_DIR)
    parser.add_argument("--n_ablation",  type=int, default=5_000)
    parser.add_argument("--n_patch",     type=int, default=2_000)
    parser.add_argument("--n_random",    type=int, default=200)
    parser.add_argument("--strategy",    default="within_batch",
                        choices=["within_batch", "permutation"])
    args = parser.parse_args()

    _, _, test_ds, _, _, test_loader = build_dataloaders(args.data)

    model = SmallParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    from src.training import evaluate
    _, _, te_auc, _, _ = evaluate(model, test_loader, desc="test")
    print(f"Full model test AUC: {te_auc:.4f}")

    n_layers = CFG["num_layers"]
    n_heads  = CFG["num_heads"]

    # ── 1. Zero-ablation importance ────────────────────────────────────────────
    print("\n=== Zero ablation sweep ===")
    ablation_idx = np.arange(args.n_ablation)
    imp_mean, imp_std = zero_ablation_sweep(
        model, test_ds, ablation_idx, n_layers, n_heads)

    sig_mask = imp_mean > 2 * imp_std
    flat_idx  = np.argsort(imp_mean.ravel())[::-1]
    top_heads = [(int(i) // n_heads, int(i) % n_heads) for i in flat_idx[:6]]
    print(f"\nTop 6 heads: {top_heads}")

    plot_importance_heatmap(imp_mean, imp_std, sig_mask,
                             filename="pdf/fig3_head_importance.pdf")

    # ── 2. Path patching ───────────────────────────────────────────────────────
    print(f"\n=== Path patching ({args.strategy}) ===")
    model_p = PatchableParT(CFG).to(DEVICE)
    model_p.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model_p.eval()

    patch_idx = np.arange(args.n_patch)
    direct_effects, path_effects = path_patch_sweep(
        model_p, test_ds, patch_idx, important_heads=top_heads[:6],
        strategy=args.strategy)

    print("\nDirect effects (recovery scores):")
    for lh, de in sorted(direct_effects.items(), key=lambda x: -x[1]):
        print(f"  L{lh[0]}H{lh[1]}: {de:+.4f}")

    plot_direct_effect_matrix(direct_effects, n_layers, n_heads)

    # Circuit graph
    de_hardcoded = {lh: direct_effects.get(lh, 0.) for lh in top_heads[:6]}
    pe_hardcoded = {(s, t): v for (s, t), v in path_effects.items()
                    if s in top_heads[:6] and t in top_heads[:6]}

    pos = {
        (0,1): (0.0, 0.5), (0,2): (0.0, -0.5),
        (1,0): (2.5, 1.2), (1,1): (2.5, 0.0), (1,3): (2.5, -1.2),
        (3,3): (5.0, 0.0),
    }
    plot_circuit(top_heads[:6], de_hardcoded, pe_hardcoded, pos=pos)

    # ── 3. Head complementarity ────────────────────────────────────────────────
    print("\n=== Head complementarity ===")
    cos_sim, individual_drops, pair_results, head_labels = head_complementarity(
        model, test_ds, np.arange(3000), top_heads[:6])

    # ── 4. Minimality test ────────────────────────────────────────────────────
    print("\n=== Minimality test ===")
    heads_ordered = sorted(direct_effects.keys(),
                           key=lambda lh: direct_effects[lh], reverse=True)
    min_results = minimality_test(
        model, test_ds, heads_ordered, direct_effects, n_layers, n_heads)
    plot_minimality(min_results, te_auc)

    # ── 5. Random baseline ─────────────────────────────────────────────────────
    print(f"\n=== Random baseline ({args.n_random} samples) ===")
    circuit_set = set(top_heads[:6])
    from src.ablation import circuit_auc
    our_auc = circuit_auc(model, test_ds, circuit_set, n_layers, n_heads)
    print(f"Our circuit AUC: {our_auc:.4f}")

    random_aucs = random_circuit_baseline(
        model, test_ds, n_layers, n_heads,
        circuit_size=6, n_random=args.n_random)
    percentile = float((random_aucs < our_auc).mean() * 100)
    print(f"Percentile: {percentile:.1f}th  "
          f"(mean={random_aucs.mean():.4f}, std={random_aucs.std():.4f})")
    plot_random_baseline(random_aucs, our_auc, te_auc, circuit_size=6)

    print("\nAll circuit identification figures saved.")


if __name__ == "__main__":
    main()
