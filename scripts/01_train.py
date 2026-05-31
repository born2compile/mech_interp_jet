"""
Train the small Particle Transformer on the Top Quark Tagging benchmark.

Usage
-----
python scripts/01_train.py                   # single run (seed 0)
python scripts/01_train.py --multiseed       # 5 independent seeds
python scripts/01_train.py --seed 2          # single run, custom seed
python scripts/01_train.py --data /path/to/data
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import random

from src.config   import CFG, DEVICE, DATA_DIR
from src.dataset  import build_dataloaders
from src.model    import SmallParT
from src.training import train, train_multiseed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default=DATA_DIR)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--n_seeds",   type=int, default=5)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader) = build_dataloaders(args.data)

    if args.multiseed:
        all_results = train_multiseed(
            train_loader, val_loader, test_loader,
            n_seeds=args.n_seeds, epochs=args.epochs,
            ablation_idx=np.arange(5_000))
        aucs = [r["test_auc"] for r in all_results]
        print(f"\nTest AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        model = SmallParT(CFG).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

        best_state, history, te_auc = train(
            model, train_loader, val_loader, test_loader,
            epochs=args.epochs,
            checkpoint_path=f"small_part_seed{args.seed}.pt")

        print(f"\nTest AUC (seed {args.seed}): {te_auc:.4f}")


if __name__ == "__main__":
    main()
