# Dissecting Jet-Tagger Through Mechanistic Interpretability

Code for the paper *Dissecting Jet-Tagger Through Mechanistic Interpretability*
(Rai & Ganguly, IIT Kanpur, 2026).

---

## Repository layout

```
mech_interp_jet/
├── src/
│   ├── config.py               # global config, seeds, hardware, plot defaults
│   ├── dataset.py              # HDF5 data loading, feature extraction
│   ├── model.py                # Particle Transformer (SmallParT, InstrumentedBlock)
│   ├── training.py             # training loop, evaluation, multi-seed run
│   ├── observables.py          # N-subjettiness, jet mass, ECF suite (C1–C3, D2, N3)
│   ├── probing.py              # residual stream extraction, logit lens, linear probes
│   ├── ablation.py             # zero/mean ablation, circuit AUC, minimality, random baseline
│   ├── patching.py             # PatchableParT, activation cache, path-patching sweep
│   ├── interaction_features.py # Pearson attribution, causal feature ablation, ECF–attn correlations
│   └── plotting.py             # all relevant figures
└── scripts/
    ├── 01_train.py             # train a single model or all 5 seeds
    ├── 02_compute_observables.py  # compute and cache jet substructure observables
    ├── 03_logit_lens.py        # logit lens + per-layer probe (basis-rotation analysis)
    ├── 04_circuit_identification.py  # ablation, path patching, minimality, random baseline
    ├── 05_residual_stream_probes.py  # jet-level R² probes by layer
    ├── 06_ecf_probes.py        # ECF R² probes, mass residualization
    └── 07_interaction_features.py   # feature attribution, ECF–attention correlations
```

Output figures are written to `IMG/` (configurable via `IMG_DIR` in `src/config.py`).

---

## Data

Download the Top Quark Tagging reference dataset:

```
https://zenodo.org/record/2603256
```

Set the path in `src/config.py` (`DATA_DIR`) or pass `--data /path/to/data` to any script.
The directory should contain `train.h5`, `val.h5`, and `test.h5`.

---

## Dependencies

```
torch >= 2.0
numpy
pandas
scikit-learn
scipy
matplotlib
networkx
tqdm
fastjet
awkward
tables   # for HDF5 loading via pandas
```

Install with:

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib networkx tqdm awkward tables
pip install fastjet   # or install via conda-forge
```

---

## Typical run order

```bash
# 1. Train a single model (seed 0)
python scripts/01_train.py --data /path/to/data

# 2. (Optional) train all 5 seeds for cross-seed stability results
python scripts/01_train.py --data /path/to/data --multiseed

# 3. Pre-compute and cache jet substructure observables
python scripts/02_compute_observables.py --data /path/to/data --full_ecf

# 4. Logit lens and basis-rotation analysis
python scripts/03_logit_lens.py --checkpoint small_part_seed0.pt

# 5. Circuit identification (ablation + path patching + minimality)
python scripts/04_circuit_identification.py --checkpoint small_part_seed0.pt

# 6. Residual stream probes (physics observables by layer)
python scripts/05_residual_stream_probes.py --checkpoint small_part_seed0.pt --obs obs.npz

# 7. ECF probes and mass-residualization control
python scripts/06_ecf_probes.py --checkpoint small_part_seed0.pt --obs obs.npz

# 8. Interaction feature attribution and ECF–attention correlations
python scripts/07_interaction_features.py --checkpoint small_part_seed0.pt
```

Each script accepts `--help` for a full list of options.

---

## Model

The model studied in the paper is a small Particle Transformer (4 particle attention layers,
2 class attention layers, 4 heads per layer, embedding dimension 128, ~1.3M parameters)
trained on a 200k-jet subset of the Top Quark Tagging reference dataset.
Test AUC across 5 random seeds: 0.9794 ± 0.0005.
