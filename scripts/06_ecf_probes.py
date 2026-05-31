"""
ECF linear probes and mass-residualization control.
Produces Figures 11, 12 (ECF R² by layer, D2 vs tau32 residualized).

Usage
-----
python scripts/06_ecf_probes.py --checkpoint small_part_seed0.pt --obs obs.npz
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
                           run_jet_level_probes, run_mass_residualized_probes)
from src.plotting import plot_ecf_3prong, plot_mass_residualized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="small_part_seed0.pt")
    parser.add_argument("--data",       default=DATA_DIR)
    parser.add_argument("--obs",        default="obs.npz")
    parser.add_argument("--n_jets",     type=int, default=10_000)
    args = parser.parse_args()

    _, _, test_ds, _, _, _ = build_dataloaders(args.data)

    model = SmallParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    obs_data = np.load(args.obs)
    n = min(args.n_jets, len(test_ds))
    probe_idx = np.arange(n)
    labels    = test_ds.labels[:n].numpy()

    # ── Representations ───────────────────────────────────────────────────────
    print("Extracting representations ...")
    jet_streams, _ = extract_residual_streams(model, test_ds, probe_idx)
    cls_states     = extract_cls_states(model, test_ds, probe_idx)
    all_reps       = jet_streams + cls_states
    layer_labels   = ([f"L{l}" for l in range(CFG["num_layers"] + 1)] +
                      [f"Cls{i}" for i in range(CFG["num_cls_layers"])])

    # ── ECF probe targets ─────────────────────────────────────────────────────
    # These keys come from 02_compute_observables.py (full_ecf mode)
    ecf_keys = {
        "C1_b1"  : "ecf_full_C1",
        "C2_b1"  : "ecf_full_C2",
        "C3_b1"  : "ecf_full_C3",
        "D2_b1"  : "ecf_full_D2",
        "N3_b1"  : "ecf_full_N3",
        "tau32"  : "nsubj_tau32",
        "jet_mass": "jet_mass",
    }

    n_ecf = min(n, sum(1 for k in obs_data if k.startswith("ecf_full_")))
    probe_targets = {}
    for out_key, npz_key in ecf_keys.items():
        if npz_key in obs_data:
            vals = obs_data[npz_key][:n].astype(np.float32)
            probe_targets[out_key] = vals

    print("\nRunning ECF probes ...")
    ecf_results = run_jet_level_probes(all_reps, probe_targets, split=7000)

    # Summary
    print(f"\n── ECF probe R² ──")
    header = f"{'Observable':10s}" + "".join(f"  {lb:>6}" for lb in layer_labels)
    print(header)
    print("─" * len(header))
    for k, scores in ecf_results.items():
        if k != "jet_mass":
            print(f"{k:10s}" + "".join(f"  {s:6.3f}" for s in scores))

    plot_ecf_3prong(ecf_results, layer_labels)

    # ── Mass residualization ──────────────────────────────────────────────────
    if "D2_b1" in probe_targets and "tau32" in probe_targets:
        mass_vals = probe_targets.get("jet_mass",
                    obs_data["jet_mass"][:n].astype(np.float32))
        d2_vals   = probe_targets["D2_b1"]
        tau32_vals= probe_targets["tau32"]

        print("\nRunning mass-residualized probes ...")
        r2_d2_raw,   r2_d2_resid   = run_mass_residualized_probes(
            all_reps, d2_vals, mass_vals, split=7000)
        r2_tau_raw,  r2_tau_resid  = run_mass_residualized_probes(
            all_reps, tau32_vals, mass_vals, split=7000)

        from src.probing import run_jet_level_probes as _probe
        r2_mass_list = _probe(all_reps, {"jet_mass": mass_vals}, split=7000)["jet_mass"]

        plot_mass_residualized(
            r2_d2_raw, r2_d2_resid,
            r2_tau_raw, r2_tau_resid,
            r2_mass_list, layer_labels)

    print("\nFigures saved.")


if __name__ == "__main__":
    main()
