"""
Jet-level and class-token linear probing of the residual stream.
Produces Figure 10 (observable R² by layer).

Usage
-----
python scripts/05_residual_stream_probes.py --checkpoint small_part_seed0.pt \
    --obs obs.npz
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
                           run_jet_level_probes)
from src.plotting import plot_observable_probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="small_part_seed0.pt")
    parser.add_argument("--data",       default=DATA_DIR)
    parser.add_argument("--obs",        default="obs.npz",
                        help="npz file produced by 02_compute_observables.py")
    parser.add_argument("--n_jets",     type=int, default=10_000)
    args = parser.parse_args()

    _, _, test_ds, _, _, _ = build_dataloaders(args.data)

    model = SmallParT(CFG).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    n = min(args.n_jets, len(test_ds))
    probe_idx = np.arange(n)
    labels    = test_ds.labels[:n].numpy()

    # Load observables
    obs_data = np.load(args.obs)
    tau32    = obs_data["nsubj_tau32"][:n]
    tau21    = obs_data["nsubj_tau21"][:n]
    jet_mass = obs_data["jet_mass"][:n]
    lead_pt  = obs_data["lead_pt_frac"][:n]

    # Extract representations
    print("Extracting residual streams ...")
    jet_streams, _ = extract_residual_streams(model, test_ds, probe_idx)
    print("Extracting class token states ...")
    cls_states = extract_cls_states(model, test_ds, probe_idx)

    all_reps     = jet_streams + cls_states
    layer_labels = ([f"L{l}" for l in range(CFG["num_layers"] + 1)] +
                    [f"Cls{i}" for i in range(CFG["num_cls_layers"])])

    probe_targets = {
        "is_top"      : labels.astype(np.float32),
        "tau32"       : tau32,
        "tau21"       : tau21,
        "jet_mass"    : jet_mass,
        "lead_pt_frac": lead_pt,
    }

    print("\nRunning jet-level probes ...")
    results = run_jet_level_probes(all_reps, probe_targets, split=7000)

    # Summary table
    print(f"\n── Probe R² by layer ──")
    header = f"{'Observable':15s}" + "".join(f"  {lb:>6}" for lb in layer_labels)
    print(header)
    print("─" * len(header))
    for name, scores in results.items():
        print(f"{name:15s}" + "".join(f"  {s:6.3f}" for s in scores))

    # Figure
    plot_observable_probes(results, layer_labels)
    print("\nFigure saved.")


if __name__ == "__main__":
    main()
