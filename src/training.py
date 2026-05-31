"""
Training loop, evaluation routine, LR schedule, and multi-seed training.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .config import CFG, DEVICE
from .model import SmallParT


# ── Single-epoch training ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for x, v, mask, labels in pbar:
        x, v, mask, labels = (x.to(DEVICE), v.to(DEVICE),
                               mask.to(DEVICE), labels.to(DEVICE))
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x, v, mask)
            if torch.isnan(logits).any():
                print("WARNING: NaN in logits — skipping batch")
                continue
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, desc="eval"):
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.
    all_probs  = []
    all_labels = []

    for x, v, mask, labels in tqdm(loader, desc=f"  [{desc}]", leave=False):
        x, v, mask, labels = (x.to(DEVICE), v.to(DEVICE),
                               mask.to(DEVICE), labels.to(DEVICE))
        with torch.amp.autocast("cuda"):
            logits = model(x, v, mask)

        probs_batch = torch.softmax(logits.float(), dim=1)[:, 1].cpu()
        if torch.isnan(probs_batch).any():
            print("WARNING: NaN in eval probs — replacing with 0.5")
            probs_batch = torch.nan_to_num(probs_batch, nan=0.5)

        total_loss += criterion(logits.float(), labels).item() * labels.size(0)
        all_probs.append(probs_batch)
        all_labels.append(labels.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    nan_frac = np.isnan(probs).mean()
    if nan_frac > 0:
        print(f"WARNING: {nan_frac * 100:.1f}% NaN in probs — replacing with 0.5")
        probs = np.nan_to_num(probs, nan=0.5)

    auc  = roc_auc_score(labels, probs)
    acc  = ((probs > 0.5).astype(int) == labels).mean()
    loss = total_loss / len(labels)
    return loss, acc, auc, probs, labels


# ── Learning-rate schedule ────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, total_steps,
                                   decay_start_frac=0.70):
    """
    AdamW optimiser + linear decay schedule used in the paper.

    Returns
    -------
    optimizer, scheduler, scaler
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3,
        betas=(0.95, 0.999), eps=1e-5, weight_decay=0.)

    decay_start = int(decay_start_frac * total_steps)

    def lr_lambda(step):
        if step <= decay_start:
            return 1.0
        return 0.01 ** ((step - decay_start) / (total_steps - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda")
    return optimizer, scheduler, scaler


# ── Main training run ─────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, test_loader,
          epochs=30, checkpoint_path="small_part_best.pt"):
    """
    Full training loop with best-checkpoint selection by val AUC.

    Returns
    -------
    best_state : state_dict of best checkpoint
    history    : list of per-epoch dicts
    te_auc     : test AUC of best model
    """
    total_steps = len(train_loader) * epochs
    optimizer, scheduler, scaler = build_optimizer_and_scheduler(
        model, total_steps)

    best_auc   = 0.
    best_state = None
    history    = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", position=0)
    for epoch in epoch_bar:
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch)
        scheduler.step()
        va_loss, va_acc, va_auc, _, _ = evaluate(model, val_loader, desc="val")

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc, va_auc=va_auc))

        is_best = va_auc > best_auc
        if is_best:
            best_auc   = va_auc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)

        epoch_bar.set_postfix(
            tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.4f}",
            va_auc=f"{va_auc:.4f}",   best=f"{best_auc:.4f}",
            marker="✓" if is_best else "")

    model.load_state_dict(best_state)
    print(f"\nBest val AUC : {best_auc:.4f}")

    te_loss, te_acc, te_auc, te_probs, te_labels = evaluate(
        model, test_loader, desc="test")
    print(f"Test AUC     : {te_auc:.4f}")
    print(f"Test Acc     : {te_acc:.4f}")

    return best_state, history, te_auc


# ── Multi-seed training ───────────────────────────────────────────────────────

def train_multiseed(train_loader, val_loader, test_loader,
                    n_seeds=5, epochs=30,
                    ablation_idx=None, n_bootstrap=1000):
    """
    Train N_SEEDS independent models and collect test AUCs and
    per-head zero-ablation importance matrices.

    Parameters
    ----------
    ablation_idx : np.ndarray  indices into test_ds for ablation measurement
                   (passed to logit_diff_with_uncertainty)

    Returns
    -------
    all_seed_results : list of dicts with keys
                       seed, test_auc, importance, state_dict
    """
    from .ablation import logit_diff_with_uncertainty

    n_layers = CFG["num_layers"]
    n_heads  = CFG["num_heads"]

    all_seed_results = []

    for seed in tqdm(range(n_seeds), desc="Training seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        m_seed = SmallParT(CFG).to(DEVICE)
        total_steps = len(train_loader) * epochs
        opt_s, sch_s, scl_s = build_optimizer_and_scheduler(m_seed, total_steps)

        best_auc_s   = 0.
        best_state_s = None

        for epoch in tqdm(range(1, epochs + 1), desc=f"  Seed {seed}", leave=False):
            train_one_epoch(m_seed, train_loader, opt_s, scl_s, epoch)
            sch_s.step()
            _, _, va_auc, _, _ = evaluate(m_seed, val_loader, desc="")
            if va_auc > best_auc_s:
                best_auc_s   = va_auc
                best_state_s = copy.deepcopy(m_seed.state_dict())

        m_seed.load_state_dict(best_state_s)
        _, _, te_auc, _, _ = evaluate(m_seed, test_loader, desc="")

        # ── Ablation importance for this seed ─────────────────────────────────
        if ablation_idx is not None:
            from .ablation import logit_diff_with_uncertainty
            imp_s   = np.zeros((n_layers, n_heads), dtype=np.float32)
            bl_s, _ = logit_diff_with_uncertainty(
                m_seed, train_loader.dataset, ablation_idx)

            for l in range(n_layers):
                block = m_seed.blocks[l]
                orig  = block.c_attn.data.clone()
                for h in range(n_heads):
                    ab     = orig.clone(); ab[h] = 0.
                    block.c_attn.data = ab
                    ab_s, _ = logit_diff_with_uncertainty(
                        m_seed, train_loader.dataset, ablation_idx)
                    imp_s[l, h] = bl_s - ab_s
                    block.c_attn.data = orig
        else:
            imp_s = None

        all_seed_results.append(dict(
            seed       = seed,
            test_auc   = te_auc,
            importance = imp_s,
            state_dict = copy.deepcopy(best_state_s),
        ))
        print(f"  Seed {seed}: test AUC = {te_auc:.4f}")
        del m_seed
        torch.cuda.empty_cache()

    aucs = [r["test_auc"] for r in all_seed_results]
    print(f"\nTest AUC across {n_seeds} seeds: "
          f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return all_seed_results
