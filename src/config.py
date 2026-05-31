"""
Global configuration, seeds, hardware, and plot defaults.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Output directory ──────────────────────────────────────────────────────────
IMG_DIR = "IMG"
os.makedirs(IMG_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = "/eos/user/s/sarai/TopTagData"

# ── Model configuration (small ParT: 4 layers, 4 heads) ──────────────────────
CFG = dict(
    num_layers      = 4,
    num_cls_layers  = 2,
    num_heads       = 4,
    embed_dim       = 128,
    ffn_ratio       = 4,
    pair_input_dim  = 4,
    pair_embed_dims = [64, 64, 64],
    num_classes     = 2,
    dropout         = 0.1,
    num_particles   = 200,
)

# ── Plot style ────────────────────────────────────────────────────────────────
FIG_W = 7
FIG_H = 5
DPI   = 150

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.labelsize" : 12,
    "axes.titlesize" : 12,
    "figure.dpi"     : DPI,
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
})


def savefig(name):
    """Save figure under IMG_DIR, creating sub-directories as needed."""
    path = os.path.join(IMG_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
