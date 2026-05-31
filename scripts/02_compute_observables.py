"""
Compute jet substructure observables (N-subjettiness, jet mass, ECF suite)
on the test set and save to disk.

Usage
-----
python scripts/02_compute_observables.py
python scripts/02_compute_observables.py --n_jets 10000 --out obs.npz
python scripts/02_compute_observables.py --full_ecf   # includes C3, N3 (slow)
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.config      import DATA_DIR
from src.dataset     import build_dataloaders
from src.observables import (compute_nsubjettiness_exact,
                              compute_jet_mass_and_lead,
                              compute_ecf_batch,
                              compute_ecf_batch_full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=DATA_DIR)
    parser.add_argument("--n_jets",   type=int, default=10_000)
    parser.add_argument("--out",      default="obs.npz")
    parser.add_argument("--full_ecf", action="store_true",
                        help="Compute C3 and N3 (O(n^4), slow)")
    parser.add_argument("--n_ecf_full", type=int, default=2_000)
    args = parser.parse_args()

    _, _, test_ds, _, _, _ = build_dataloaders(args.data)

    n = min(args.n_jets, len(test_ds))
    v_np    = test_ds.v[:n].numpy()
    mask_np = test_ds.mask[:n].numpy()
    labels  = test_ds.labels[:n].numpy()

    print(f"Computing observables on {n} jets ...")

    nsubj = compute_nsubjettiness_exact(v_np, mask_np, beta=1.0)
    jet_mass, lead_pt_frac, n_particles = compute_jet_mass_and_lead(v_np, mask_np)

    ecf_obs    = compute_ecf_batch(v_np, mask_np, beta=1.0, desc="beta=1.0")
    ecf_obs_b2 = compute_ecf_batch(v_np, mask_np, beta=2.0, desc="beta=2.0")

    save_dict = dict(
        labels      = labels,
        jet_mass    = jet_mass,
        lead_pt_frac= lead_pt_frac,
        n_particles = n_particles,
        **{f"nsubj_{k}": v for k, v in nsubj.items()},
        **{f"ecf_{k}":   v for k, v in ecf_obs.items()},
        **{f"ecf_b2_{k}":v for k, v in ecf_obs_b2.items()},
    )

    if args.full_ecf:
        n_full  = min(args.n_ecf_full, n)
        print(f"\nComputing full ECF suite (C3, N3) on {n_full} jets ...")
        ecf_full = compute_ecf_batch_full(
            v_np[:n_full], mask_np[:n_full], beta=1.0)
        for k, v in ecf_full.items():
            save_dict[f"ecf_full_{k}"] = v

    np.savez(args.out, **save_dict)
    print(f"\nSaved to {args.out}")
    print(f"Keys: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()
