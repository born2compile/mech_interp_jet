"""
Publication-quality figures for the mechanistic interpretability analysis.
All savefig calls write to IMG_DIR via config.savefig().
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable

from .config import CFG, FIG_W, FIG_H, savefig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_cls(ax, n_part, n_total):
    ax.axvspan(-0.5, n_part - 0.5,
               color="gray", alpha=0.06, zorder=0)
    ax.axvspan(n_part - 0.5, n_total - 0.5,
               color="blue", alpha=0.07, zorder=0)
    ax.axvline(n_part - 0.5, color="blue",
               ls="--", lw=1.0, alpha=0.55, zorder=1)


def _region_labels(ax, n_part, n_total):
    ax.text((n_part - 1) / 2, 1.035,
            "Particle attention",
            ha="center", va="bottom",
            fontsize=8.5, color="dimgray", style="italic",
            transform=ax.get_xaxis_transform())
    ax.text(n_part + (n_total - n_part - 1) / 2, 1.035,
            "Class attention",
            ha="center", va="bottom",
            fontsize=8.5, color="blue", style="italic",
            transform=ax.get_xaxis_transform())


# ── Figure: Logit lens AUC trajectory ────────────────────────────────────────

def plot_logit_lens_auc(lens_aucs, te_auc, filename="pdf/fig11n_logit_lens.pdf"):
    n_part   = CFG["num_layers"] + 1
    n_cls    = CFG["num_cls_layers"]
    n_total  = n_part + n_cls

    x_pos  = np.arange(n_total)
    x_lbls = [f"L{l}" for l in range(n_part)] + \
              [f"Cls{i}" for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _shade_cls(ax, n_part, n_total)

    ax.axhline(0.5, color="gray", ls=":",  lw=1.2, label="Random (AUC = 0.5)")
    ax.axhline(te_auc, color="firebrick", ls="--", lw=1.5,
               label=f"Full model (AUC = {te_auc:.4f})")

    ax.plot(x_pos, lens_aucs,
            "o-", color="blue", lw=2.4, ms=8,
            zorder=5, label="Logit lens AUC", clip_on=False)

    offsets = {0: (+0, +14), 1: (+0, -16), 2: (+0, +14),
               3: (+0, +14), 4: (-18, +8), 5: (+0, +10), 6: (+0, +10)}
    for i, auc in enumerate(lens_aucs):
        dx, dy = offsets.get(i, (0, 12))
        va = "bottom" if dy > 0 else "top"
        ax.annotate(f"{auc:.3f}",
                    xy=(i, auc), xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=9, color="blue")

    _region_labels(ax, n_part, n_total)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbls, fontsize=10)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(0.08, 1.06)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Representation depth", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(fontsize=9, loc="lower left",
              bbox_to_anchor=(0.02, 0.05),
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    plt.tight_layout()
    savefig(filename)


def plot_logit_lens_ld(lens_ld_top, lens_ld_qcd,
                        filename="pdf/fig11nb_logit_lens_ld.pdf"):
    n_part   = CFG["num_layers"] + 1
    n_cls    = CFG["num_cls_layers"]
    n_total  = n_part + n_cls

    x_pos  = np.arange(n_total)
    x_lbls = [f"L{l}" for l in range(n_part)] + \
              [f"Cls{i}" for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _shade_cls(ax, n_part, n_total)
    ax.axhline(0, color="black", lw=0.8, zorder=2)

    ax.plot(x_pos, lens_ld_top,
            "o-", color="firebrick", lw=2.2, ms=7, label="Top jets", zorder=4)
    ax.plot(x_pos, lens_ld_qcd,
            "s-", color="blue", lw=2.2, ms=7, label="QCD jets", zorder=4)

    ax.fill_between(x_pos, lens_ld_top, lens_ld_qcd,
                    where=[t >= q for t, q in zip(lens_ld_top, lens_ld_qcd)],
                    alpha=0.12, color="firebrick")
    ax.fill_between(x_pos, lens_ld_top, lens_ld_qcd,
                    where=[t < q for t, q in zip(lens_ld_top, lens_ld_qcd)],
                    alpha=0.12, color="blue")

    _region_labels(ax, n_part, n_total)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbls, fontsize=10)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_xlabel("Representation depth", fontsize=11)
    ax.set_ylabel(r"Mean logit difference"
                  "\n"
                  r"$\langle \log p_{\rm top} - \log p_{\rm QCD} \rangle$",
                  fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(fontsize=9, loc="upper left",
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    plt.tight_layout()
    savefig(filename)


# ── Figure: Logit lens vs per-layer probe (basis-rotation) ───────────────────

def plot_lens_vs_probe(lens_aucs, probe_aucs, te_auc, layer_labels,
                        filename="fig_E1_logitlens_control.png"):
    n_part  = CFG["num_layers"] + 1
    n_total = len(layer_labels)
    x_pos   = np.arange(n_total)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _shade_cls(ax, n_part, n_total)

    ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="Random (0.5)")
    ax.axhline(te_auc, color="firebrick", ls="--", lw=1.6,
               label=f"Full model ({te_auc:.4f})")

    ax.plot(x_pos, lens_aucs,
            "o-", color="blue", lw=2.2, ms=7,
            label="Logit lens (final head projection)")
    ax.plot(x_pos, probe_aucs,
            "s-", color="darkorange", lw=2.2, ms=7,
            label="Per-layer logistic probe (trained)")

    ax.fill_between(x_pos, lens_aucs, probe_aucs,
                    where=[p > l for p, l in zip(probe_aucs, lens_aucs)],
                    alpha=0.15, color="darkorange",
                    label="Basis mismatch region")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_xlim(-0.5, n_total - 0.5)
    ax.set_ylim(0.08, 1.06)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Representation depth", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left",
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    plt.tight_layout()
    savefig(filename)


# ── Figure: Head importance heatmap ──────────────────────────────────────────

def plot_importance_heatmap(imp_matrix, std_matrix=None, sig_matrix=None,
                             title="Head importance", filename="pdf/fig3_head_importance.pdf",
                             vmax=None):
    n_layers, n_heads = imp_matrix.shape

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax_ = vmax if vmax is not None else float(np.abs(imp_matrix).max())

    im = ax.imshow(imp_matrix, cmap="RdYlGn",
                   aspect="auto", vmin=-vmax_, vmax=vmax_)
    ax.grid(False)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {l}" for l in range(n_layers)], fontsize=10)
    ax.set_xlabel("Head",  fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=11, pad=10)

    for l in range(n_layers):
        for h in range(n_heads):
            val = imp_matrix[l, h]
            tc  = "white" if abs(val) > 0.55 * vmax_ else "black"
            if std_matrix is not None:
                std = std_matrix[l, h]
                sig = "*" if (sig_matrix is not None and sig_matrix[l, h]) else ""
                txt = f"{val:.2f}\n±{std:.2f}{sig}"
                fs  = 8.0
            else:
                sig = "*" if (sig_matrix is not None and sig_matrix[l, h]) else ""
                txt = f"{val:.2f}{sig}"
                fs  = 9.5
            ax.text(h, l, txt, ha="center", va="center",
                    fontsize=fs, color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03, aspect=18)
    cbar.set_label("ΔLogit diff", fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8)

    for l in range(n_layers):
        for h in range(n_heads):
            rect = plt.Rectangle(
                (h - 0.5, l - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=0.4, zorder=3)
            ax.add_patch(rect)

    plt.tight_layout()
    savefig(filename)
    plt.close()


# ── Figure: Direct-effect matrix ─────────────────────────────────────────────

def plot_direct_effect_matrix(direct_effects, n_layers, n_heads,
                               filename="pdf/figA_patch_importance.pdf"):
    imp_patch = np.zeros((n_layers, n_heads), dtype=np.float32)
    for (l, h), recovery in direct_effects.items():
        imp_patch[l, h] = float(recovery)
    vmax = float(np.abs(imp_patch).max())
    plot_importance_heatmap(imp_patch, filename=filename, vmax=vmax)


# ── Figure: Circuit graph ─────────────────────────────────────────────────────

def plot_circuit(circuit_heads, direct_eff, pe_raw, curve=None,
                 pos=None, filename="pdf/fig55n_circuit.pdf"):
    """
    Directed graph of the identified circuit.

    Parameters
    ----------
    circuit_heads : list of (l, h)
    direct_eff    : dict (l,h) -> float
    pe_raw        : dict ((src),(tgt)) -> float  (path effects)
    curve         : dict ((src),(tgt)) -> float  (arc curvature, optional)
    pos           : dict (l,h) -> (x, y)         (node positions, optional)
    """
    if pos is None:
        col_x = {0: 0.0, 1: 2.5, 3: 5.0}
        # default y positions within each layer
        layer_y = {
            (0, 1): 1.0, (0, 2): -1.0,
            (1, 0): 1.2, (1, 1): 0.0, (1, 3): -1.2,
            (3, 3): 0.0,
        }
        pos = {lh: (col_x.get(lh[0], lh[0] * 2.5), layer_y.get(lh, 0.))
               for lh in circuit_heads}

    if curve is None:
        curve = {}

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axis("off")

    # Edges
    pe_min = min(pe_raw.values())
    pe_max = max(pe_raw.values())

    for (src, tgt), delta in sorted(pe_raw.items(), key=lambda x: x[1]):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        t      = (delta - pe_min) / (pe_max - pe_min + 1e-8)
        lw     = 1.0 + 4.5 * t
        alpha  = 0.30 + 0.45 * t
        rad    = curve.get((src, tgt), 0.1)

        ax.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#555555",
                        lw=lw,
                        alpha=alpha,
                        mutation_scale=12,
                        connectionstyle=f"arc3,rad={rad:.2f}"),
                    zorder=2)

    # Nodes
    de_vals = list(direct_eff.values())
    node_norm = TwoSlopeNorm(vmin=min(de_vals), vcenter=0., vmax=max(de_vals))
    node_cmap = plt.cm.RdYlGn
    NODE_SIZE = 1800

    for lh in circuit_heads:
        x, y   = pos[lh]
        colour = node_cmap(node_norm(direct_eff[lh]))
        l, h   = lh
        ax.scatter(x, y, s=NODE_SIZE, color=colour,
                   edgecolors="#333333", linewidths=1.5, zorder=4)
        ax.text(x, y, f"L{l}H{h}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="black", zorder=6,
                path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")])

    # Layer labels
    col_x_uniq = {}
    for lh in circuit_heads:
        col_x_uniq[lh[0]] = pos[lh][0]
    for layer, cx in col_x_uniq.items():
        ax.text(cx, max(p[1] for p in pos.values()) + 0.3,
                f"Layer {layer}",
                ha="center", va="bottom",
                fontsize=10, fontweight="semibold", color="#333333")
        ax.axvline(cx, ymin=0.05, ymax=0.95,
                   color="lightgray", lw=0.8, ls="--", zorder=0)

    # Colourbar
    sm = ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, aspect=20, shrink=0.65)
    cbar.set_label("Direct effect\n(recovery score)", fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.axhline(y=node_norm(0.), color="black", lw=1.0, ls="--")

    ys = [p[1] for p in pos.values()]
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.7)

    plt.tight_layout()
    savefig(filename)


# ── Figure: Minimality test ───────────────────────────────────────────────────

def plot_minimality(min_results, te_auc, filename="pdf/fig_minimality.pdf"):
    n_s   = len(min_results)
    x_pos = np.arange(n_s)
    aucs_ = [r["auc"]   for r in min_results]
    lo_s  = [r["ci_lo"] for r in min_results]
    hi_s  = [r["ci_hi"] for r in min_results]

    fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H))
    ax.errorbar(x_pos, aucs_,
                yerr=[[a - l for a, l in zip(aucs_, lo_s)],
                      [h - a for h, a in zip(hi_s, aucs_)]],
                fmt="o-", color="blue", lw=2, ms=7,
                capsize=4, capthick=1.5,
                label="Circuit AUC ± 95% CI")
    ax.axhline(te_auc, color="firebrick", ls="--", lw=1.5,
               label=f"Full model ({te_auc:.4f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"+L{r['head'][0]}H{r['head'][1]}" for r in min_results],
        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.set_ylim(0.45, 1.01)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    savefig(filename)


# ── Figure: Random baseline histogram ────────────────────────────────────────

def plot_random_baseline(random_aucs, our_auc, full_auc, circuit_size,
                          filename="figR1_random_baseline.png"):
    from scipy import stats as _stats
    percentile = float((random_aucs < our_auc).mean() * 100)
    z_score    = (our_auc - random_aucs.mean()) / random_aucs.std()
    n_random   = len(random_aucs)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(random_aucs, bins=30, color="blue", alpha=0.70,
            edgecolor="white", linewidth=0.5,
            label=f"Random {circuit_size}-head circuits (n={n_random})")
    ax.axvline(our_auc,  color="firebrick", lw=2.5,
               label=f"Our circuit  AUC={our_auc:.4f}  ({percentile:.0f}th pct)")
    ax.axvline(full_auc, color="black",     lw=1.5, ls="--",
               label=f"Full model  AUC={full_auc:.4f}")
    ax.axvline(0.5,      color="gray",      lw=1.0, ls=":")
    ax.set_xlabel("Test AUC", fontsize=12)
    ax.set_ylabel("Count",    fontsize=12)
    ax.legend(fontsize=12)
    plt.legend(loc='center left')
    plt.tight_layout()
    savefig(filename)


# ── Figure: Physics observable probes ────────────────────────────────────────

def plot_observable_probes(cls_results, layer_labels,
                            filename="figC_cls_token_probes.png"):
    n_part  = CFG["num_layers"] + 1
    n_total = len(layer_labels)

    obs_plot    = ["tau32", "tau21", "jet_mass", "lead_pt_frac"]
    obs_labels_ = ["τ₃₂ (R²)", "τ₂₁ (R²)", "Jet mass (R²)", "Lead pT frac (R²)"]
    colors_obs_ = ["firebrick", "blue", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(FIG_W + 2, FIG_H))

    for name, label, color in zip(obs_plot, obs_labels_, colors_obs_):
        if name in cls_results:
            ax.plot(layer_labels, cls_results[name],
                    "o-", color=color, lw=2, ms=7, label=label)

    ax.axvspan(n_part - 0.5, n_total - 0.5,
               color="lightblue", alpha=0.25, label="Class attention blocks")
    ax.axvline(n_part - 0.5, color="royalblue", ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("Representation depth", fontsize=14)
    ax.set_ylabel("Linear probe R²",      fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    savefig(filename)


# ── Figure: ECF 3-prong vs 2-prong probes ────────────────────────────────────

def plot_ecf_3prong(ecf_probe_r2, layer_labels,
                    filename="pdf/fig_ecf_3prong.pdf"):
    n_part  = CFG["num_layers"] + 1
    n_total = len(layer_labels)
    x_pos   = np.arange(n_total)
    cls_start = n_part - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))

    for ax, (keys, title) in zip(axes, [
        (["C3_b1", "N3_b1", "tau32"],
         "3-prong energy correlators"),
        (["C3_b1", "N3_b1", "C2_b1", "D2_b1"],
         "3-prong vs 2-prong energy correlators"),
    ]):
        _shade_cls(ax, n_part, n_total)
        styles = {
            "C3_b1": ("o-", "firebrick",  r"$C_3^{(\beta=1)}$  [3-prong, target]"),
            "N3_b1": ("s-", "darkorange", r"$N_3^{(\beta=1)}$  [generalised 3-prong]"),
            "tau32": ("^--", "gray",      r"$\tau_{32}$  [reference]"),
            "C2_b1": ("^--", "blue", r"$C_2^{(\beta=1)}$  [2-prong]"),
            "D2_b1": ("v--", "seagreen",  r"$D_2^{(\beta=1)}$  [2-prong opt.]"),
        }
        for k in keys:
            if k not in ecf_probe_r2:
                continue
            fmt, col, lbl = styles[k]
            ax.plot(x_pos, ecf_probe_r2[k], fmt, color=col,
                    lw=2.2, ms=7, alpha=0.85, label=lbl)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layer_labels, fontsize=12)
        ax.set_xlim(-0.5, n_total - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Representation depth", fontsize=14)
        ax.set_ylabel("Linear probe $R^2$", fontsize=14)
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=12, loc="lower right",
                  frameon=True, framealpha=0.92, edgecolor="lightgray")
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    plt.tight_layout()
    savefig(filename)


# ── Figure: D2 vs tau32 mass-residualized ────────────────────────────────────

def plot_mass_residualized(r2_d2_raw, r2_d2_resid, r2_tau32_raw, r2_tau32_resid,
                            r2_mass, layer_labels,
                            filename="pdf/fig_mass_residualized.pdf"):
    n_total = len(layer_labels)
    x_pos   = np.arange(n_total)
    n_part  = CFG["num_layers"] + 1

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))

    for ax, (d2_vals, tau_vals, title) in zip(axes, [
        (r2_d2_raw,   r2_tau32_raw,   "Raw observables"),
        (r2_d2_resid, r2_tau32_resid, "Mass-residualized observables"),
    ]):
        _shade_cls(ax, n_part, n_total)
        ax.plot(x_pos, d2_vals,  "o-", color="blue", lw=2, ms=7,
                label=r"$D_2^{(\beta=1)}$")
        ax.plot(x_pos, tau_vals, "s-", color="firebrick",  lw=2, ms=7,
                label=r"$\tau_{32}$")
        if title == "Raw observables":
            ax.plot(x_pos, r2_mass, "^--", color="gray", lw=1.5, ms=5,
                    alpha=0.7, label="Jet mass")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layer_labels, fontsize=10)
        ax.set_xlim(-0.5, n_total - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Representation depth", fontsize=11)
        ax.set_ylabel("Linear probe $R^2$", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    plt.tight_layout()
    savefig(filename)


# ── Figure: ECF2 attention bar chart (2-prong discriminating power) ───────────

def plot_ecf_attn_bars(ecf2_attn_results, heads_to_show,
                        filename="pdf/fig_ecf_Csub_attn_vs_ecf2.pdf"):
    x_pos = np.arange(len(heads_to_show))
    width = 0.25

    r_tops   = [ecf2_attn_results[(l,h)]["r_top"]  for l,h in heads_to_show]
    r_qcds   = [ecf2_attn_results[(l,h)]["r_qcd"]  for l,h in heads_to_show]
    deltas   = [ecf2_attn_results[(l,h)]["delta"]   for l,h in heads_to_show]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(x_pos - width, r_tops,  width,
           color="firebrick", alpha=0.75, label=r"$r_{\rm top}$")
    ax.bar(x_pos,          r_qcds, width,
           color="blue", alpha=0.75, label=r"$r_{\rm QCD}$")
    ax.bar(x_pos + width,  deltas, width,
           color="black", alpha=0.80,
           label=r"$\delta = r_{\rm top} - r_{\rm QCD}$")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{l}H{h}" for l,h in heads_to_show],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig(filename)


# ── Figure: 2-prong vs 3-prong discriminating power per head ─────────────────

def plot_ecf_2v3_discriminating(ecf3_attn, heads,
                                  filename="pdf/fig_ecfattn_2vs3.pdf"):
    head_labels = [f"L{l}H{h}" for l, h in heads]
    d2_vals = [ecf3_attn[(l,h)]["delta2"] for l, h in heads]
    d3_vals = [ecf3_attn[(l,h)]["delta3"] for l, h in heads]

    x_bar = np.arange(len(heads))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(x_bar - width/2, d2_vals, width, color="blue", alpha=0.75,
           label=r"$\delta_2 = r^\mathrm{top}_2 - r^\mathrm{QCD}_2$"
                 "\n(2-prong discriminating power)")
    ax.bar(x_bar + width/2, d3_vals, width, color="firebrick", alpha=0.75,
           label=r"$\delta_3 = r^\mathrm{top}_3 - r^\mathrm{QCD}_3$"
                 "\n(3-prong discriminating power)")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(head_labels, fontsize=10)
    ax.set_xlabel("Circuit head", fontsize=11)
    ax.set_ylabel(r"Discriminating power $\delta$", fontsize=11)
    ax.set_title("2-prong vs 3-prong discriminating power\nper circuit head",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right",
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    plt.tight_layout()
    savefig(filename)


# ── Figure: Causal feature ablation (2x3 layout, all circuit heads) ──────────

def plot_feature_ablation(baseline_profiles, feat_results, circuit_heads,
                           imp_mean, BC,
                           filename="figB_feature_ablation.png"):
    colors_feat = {
        'ln_delta': '#e41a1c',
        'ln_kT'   : '#377eb8',
        'ln_z'    : '#4daf4a',
        'ln_m2'   : '#984ea3',
    }
    roles = {
        (0,1): 'source', (0,2): 'secondary source',
        (1,0): 'relay',  (1,1): 'relay',
        (1,3): 'relay',  (3,3): 'readout',
    }

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W * 3, FIG_H * 2))
    axes = axes.ravel()

    for ax, (l, h) in zip(axes, circuit_heads):
        diff_base = (np.nan_to_num(baseline_profiles[(l,h)]['top']) -
                     np.nan_to_num(baseline_profiles[(l,h)]['qcd']))
        ax.plot(BC, diff_base, 'k-', lw=2.5, label='All features', zorder=5)

        for fname, res in feat_results.items():
            diff_abl = (np.nan_to_num(res['profiles'][(l,h)]['top']) -
                        np.nan_to_num(res['profiles'][(l,h)]['qcd']))
            ax.plot(BC, diff_abl, '--', lw=1.5,
                    color=colors_feat[fname],
                    label=f'{fname} zeroed', alpha=0.85)

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel(r'$\Delta R$ between particle pair', fontsize=10)
        ax.set_ylabel('Attn diff (top $-$ QCD)', fontsize=9)
        ax.set_title(f'L{l}H{h}  [{roles.get((l,h), "?")}]  '
                     f'(imp $=$ {imp_mean[l,h]:+.2f})',
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.grid(False, axis='x')
        ax.set_xlim(0, 1.0)

    plt.suptitle(
        'Causal feature ablation on all six circuit heads\n'
        r'Top$-$QCD attention difference profiles under single-feature interventions',
        fontsize=12)
    plt.tight_layout()
    savefig(filename)
