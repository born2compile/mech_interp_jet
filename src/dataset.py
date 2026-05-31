"""
Dataset loading and preprocessing for the Top Quark Tagging benchmark.
"""

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .config import CFG


def load_split(path, max_jets=None, desc=""):
    """
    Load a Top Tagging HDF5 file (PyTables/pandas format).

    Returns
    -------
    x     : (N, 7, 200)  float32   particle features
    v     : (N, 4, 200)  float32   (px, py, pz, E)  for interaction terms
    mask  : (N, 1, 200)  float32   1 = real particle, 0 = padding
    labels: (N,)         int64     1 = top, 0 = QCD
    """
    P = CFG["num_particles"]

    print(f"  Reading {desc} from {path} ...")
    df = pd.read_hdf(path, key="table", stop=max_jets)
    if max_jets is not None:
        df = df.iloc[:max_jets]
    N = len(df)
    print(f"  Rows loaded: {N}")

    with tqdm(total=4, desc=f"  Extracting {desc}", leave=False) as pbar:
        E  = df[[f"E_{i}"  for i in range(P)]].values.astype(np.float32); pbar.update(1)
        PX = df[[f"PX_{i}" for i in range(P)]].values.astype(np.float32); pbar.update(1)
        PY = df[[f"PY_{i}" for i in range(P)]].values.astype(np.float32); pbar.update(1)
        PZ = df[[f"PZ_{i}" for i in range(P)]].values.astype(np.float32); pbar.update(1)

    labels = df["is_signal_new"].values.astype(np.int64)
    del df

    # ── Mask ─────────────────────────────────────────────────────────────────
    mask = (E > 0).astype(np.float32)                          # (N, 200)

    # ── Derived kinematics ───────────────────────────────────────────────────
    print(f"  Computing kinematics for {desc} ...")
    pt       = np.sqrt(PX**2 + PY**2).clip(min=1e-8)
    eta      = np.arcsinh(PZ / pt)
    phi      = np.arctan2(PY, PX)

    pt_m     = pt * mask
    jet_pt   = pt_m.sum(axis=1, keepdims=True).clip(min=1e-8)
    jet_E    = (E  * mask).sum(axis=1, keepdims=True).clip(min=1e-8)
    jet_eta  = (eta * pt_m).sum(axis=1, keepdims=True) / jet_pt
    jet_sphi = (np.sin(phi) * pt_m).sum(axis=1, keepdims=True) / jet_pt
    jet_cphi = (np.cos(phi) * pt_m).sum(axis=1, keepdims=True) / jet_pt
    jet_phi  = np.arctan2(jet_sphi, jet_cphi)

    deta        = eta - jet_eta
    dphi        = (phi - jet_phi + math.pi) % (2 * math.pi) - math.pi
    dr          = np.sqrt(deta**2 + dphi**2).clip(min=1e-8)
    log_pt      = np.log(pt.clip(min=1e-8))
    log_e       = np.log(E.clip(min=1e-8))
    log_pt_rel  = np.log((pt / jet_pt).clip(min=1e-8))
    log_e_rel   = np.log((E  / jet_E ).clip(min=1e-8))

    # ── Feature matrix x: (N, 7, 200) ───────────────────────────────────────
    x = np.stack([deta, dphi, log_pt, log_e,
                  log_pt_rel, log_e_rel, dr], axis=1).astype(np.float32)
    x *= mask[:, np.newaxis, :]

    # ── Four-vector v: (N, 4, 200)  order: px, py, pz, E ────────────────────
    v = np.stack([PX, PY, PZ, E], axis=1).astype(np.float32)
    v *= mask[:, np.newaxis, :]

    # ── Mask: (N, 1, 200) ────────────────────────────────────────────────────
    mask_3d = mask[:, np.newaxis, :]

    print(f"  Done. Signal fraction: {labels.mean():.4f} | "
          f"Particles/jet: {mask.sum(axis=1).mean():.1f} avg")

    return (torch.from_numpy(x),
            torch.from_numpy(v),
            torch.from_numpy(mask_3d),
            torch.from_numpy(labels))


class TopTagDataset(Dataset):
    def __init__(self, path, max_jets=None, desc=""):
        super().__init__()
        self.x, self.v, self.mask, self.labels = load_split(
            path, max_jets=max_jets, desc=desc)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.v[idx], self.mask[idx], self.labels[idx]


def build_dataloaders(data_dir):
    """
    Construct train/val/test datasets and DataLoaders.

    Parameters
    ----------
    data_dir : str  Path to directory containing train.h5, val.h5, test.h5

    Returns
    -------
    train_ds, val_ds, test_ds : TopTagDataset
    train_loader, val_loader, test_loader : DataLoader
    """
    print("Loading datasets...")
    train_ds = TopTagDataset(f"{data_dir}/train.h5", max_jets=200_000, desc="train")
    val_ds   = TopTagDataset(f"{data_dir}/val.h5",   max_jets=50_000,  desc="val")
    test_ds  = TopTagDataset(f"{data_dir}/test.h5",  max_jets=50_000,  desc="test")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    return (train_ds, val_ds, test_ds,
            train_loader, val_loader, test_loader)
