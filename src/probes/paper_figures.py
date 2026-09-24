"""
Paper-ready figures for the manuscript, built at their final physical size.

Every figure is created with `figsize` in inches equal to the width it will
occupy on the page, so a font declared here at 8 pt is 8 pt in the compiled PDF.
Nothing is rescaled by \\includegraphics, which is what makes "readable at the
PDF scale" checkable rather than a guess: render the PNG at 1:1 and look at it.

Conventions:
  * Selectivity (Llama | Qwen) and RSA (alignment | lexical partial) are
    two-panel figures.
  * Panels are near-square; no in-figure titles; the caption says what the
    figure is.
  * Taxonomy names are spelled out ("GoEmotions", not "go_emotions").

Colours are tab10 on a seaborn `whitegrid` ground. A taxonomy keeps the same
colour in every figure, and each taxonomy has its own marker shape, so identity
does not rest on colour alone.

Reads only CSVs that are already in the repo. Run from the repo root:

    python src/probes/paper_figures.py
"""

import json
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# --- Page geometry -----------------------------------------------------------
# The conference style sets textwidth=5.5in. A figure drawn 5.5in wide and included at
# width=\textwidth is placed 1:1, so matplotlib points == LaTeX points.
TEXTWIDTH_IN = 5.5

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "figures", "manuscript_revised")

# PNG is what the manuscript includes. Physical size is fixed by `figsize`,
# so this is resolution only.
SAVE_DPI = 400

# --- Taxonomy identity -------------------------------------------------------
TAXONOMIES = ["ekman_basic_emotions", "plutchik_wheel", "go_emotions"]

PRETTY = {
    "ekman_basic_emotions": "Ekman",
    "plutchik_wheel": "Plutchik",
    "go_emotions": "GoEmotions",
}

# tab10. One mapping everywhere, so a taxonomy has the same colour in the
# selectivity and the RSA figures.
COLORS = {
    "ekman_basic_emotions": "tab:blue",
    "plutchik_wheel": "tab:orange",
    "go_emotions": "tab:green",
}
# One shape per taxonomy, on top of the tab10 colours. tab10's orange and green
# are separated by dE 0.7 under protanopia -- the same colour for a red-blind
# reader -- so shape carries the identity when colour cannot.
MARKERS = {
    "ekman_basic_emotions": "o",
    "plutchik_wheel": "s",
    "go_emotions": "^",
}

# --- Type scale --------------------------------------------------------------
# Body text is 10 pt and captions are 9 pt, so figure type sits just under the
# caption: nothing below 7 pt ends up on the page.
RC = {
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 2.6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "pdf.fonttype": 42,      # embed TrueType, not Type-3: required by most venues
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}

def save(fig, name):
    """Write the PNG that the manuscript includes.

    The figure is already the width it occupies on the page, so including it at
    that width reproduces the declared point sizes exactly and `dpi` only sets
    the raster resolution: 400 dpi is comfortably above print quality.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(png, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  {png}")


def style_axes(ax):
    """The seaborn-whitegrid look: full light-grey box around the axes, dashed
    grid behind the data."""
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_axisbelow(True)


def label_panel(ax, text):
    """Bold (a)/(b) above the top-left corner of the axes."""
    ax.text(0.0, 1.035, text["letter"], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


# =============================================================================
# Figure 3 - taxonomy granularity diagram
# =============================================================================
# Block geometry transcribed from a vector drawing of the diagram, then laid out
# at 0.75\linewidth with type large enough to read at that size.

SVG_BLOCKS = {
    "ekman": [(160.5, 128.7), (294.2, 128.7), (427.8, 117.5),
              (550.4, 128.7), (684.0, 139.8), (828.8, 128.7)],
    "plutchik": [(160.5, 84.1), (249.6, 117.5), (372.1, 95.3), (472.4, 84.1),
                 (561.5, 95.3), (661.8, 106.4), (773.1, 95.3), (873.4, 84.1)],
    "goemotions": [(160.5, 50.7), (216.2, 61.8), (283.0, 50.7), (338.7, 61.8),
                   (405.6, 50.7), (461.3, 50.7), (516.9, 50.7), (572.6, 50.7),
                   (628.3, 61.8), (695.2, 50.7), (750.9, 50.7), (806.6, 61.8),
                   (873.4, 50.7), (929.1, 28.4)],
}

# Colours follow the taxonomy map used everywhere else (Ekman blue, Plutchik
# orange, GoEmotions green). The SVG had Ekman and Plutchik the other way round,
# which put this figure at odds with every data figure in the paper.
ROWS = [
    ("ekman", "Ekman", "6 BASIC", "#d7deef", "#3f61bd", "#8291bf"),
    ("plutchik", "Plutchik", "8 PRIMARY", "#e7d3ce", "#b23e2f", "#b07a6d"),
    ("goemotions", "GoEmotions", "27 FINE-GRAINED", "#cfe6da", "#248661", "#7aab95"),
]

FIG3_WIDTH_FRAC = 0.75          # \includegraphics[width=0.75\linewidth]


def make_fig3():
    width_pt = TEXTWIDTH_IN * 72 * FIG3_WIDTH_FRAC     # 297 pt
    label_right = 78.0          # right edge of the text column, in points
    block_x0, block_x1 = 86.0, width_pt - 2.0
    block_h, row_pitch, top_pad = 24.0, 40.0, 5.0
    height_pt = top_pad + 2 * row_pitch + block_h + 5.0

    # SVG blocks span x 160.5..957.5; map that onto [block_x0, block_x1].
    scale = (block_x1 - block_x0) / (957.5 - 160.5)

    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_pt)
    ax.set_ylim(height_pt, 0)          # y grows downward, as in the SVG
    ax.axis("off")

    for i, (key, name, sub, fill, name_c, sub_c) in enumerate(ROWS):
        y0 = top_pad + i * row_pitch
        centre = y0 + block_h / 2
        for x, w in SVG_BLOCKS[key]:
            ax.add_patch(FancyBboxPatch(
                (block_x0 + (x - 160.5) * scale, y0), w * scale, block_h,
                boxstyle="round,pad=0,rounding_size=1.5",
                facecolor=fill, edgecolor="none"))
        ax.text(label_right, centre - 5.0, name, ha="right", va="center",
                fontsize=10, fontweight="bold", color=name_c)
        ax.text(label_right, centre + 6.5, sub, ha="right", va="center",
                fontsize=7, color=sub_c, family="monospace")

    save(fig, "fig3_taxonomies")


# =============================================================================
# Figures 4 + 5 - chance-corrected selectivity, Llama | Qwen
# =============================================================================

SELECTIVITY_PANELS = [
    dict(letter="(a)", name="Llama-2-7b-chat-hf",
         metrics="results/probes_generated_prompts_Llama-2-7b-chat-hf/"
                 "full_probing_metrics_Llama-2-7b-chat-hf_final_F1.csv",
         boot="results/probes_generated_prompts_Llama-2-7b-chat-hf/"
              "normalized_f1_bootstrap_ci_Llama-2-7b-chat-hf.csv"),
    dict(letter="(b)", name="Qwen2.5-14B-Instruct",
         metrics="results/probes_generated_prompts_Qwen2.5-14B-Instruct/"
                 "full_probing_metrics_Qwen2.5-14B-Instruct_final_F1.csv",
         boot="results/probes_generated_prompts_Qwen2.5-14B-Instruct/"
              "normalized_f1_bootstrap_ci_Qwen2.5-14B-Instruct.csv"),
]

# The taxonomy analysis (selectivity, RSA and raw macro-F1 figures, and the
# selectivity values quoted in the text) comes from the first probing run, kept
# next to the re-run CSVs with a `_taxonomy` suffix. The re-run CSVs at the
# unsuffixed paths feed the domain x position analysis and differ by up to 0.18
# normalized macro-F1 on some layers. `run="first"` reads the `_taxonomy` files.
#
# The bootstrap CSVs belong to the first run. `run="new"` therefore omits the
# bands rather than pairing re-run point estimates with those intervals.


def boot_ci(b):
    """2.5/97.5 bootstrap percentiles, under either CSV schema.

    The Llama file was written by the current script (`norm_f1_p2_5`); the Qwen
    file was written by an earlier version and uses `norm_f1_lower`. Both columns hold
    np.percentile(boot, 2.5) / (…, 97.5), so the band means the same thing.
    """
    if "norm_f1_p2_5" in b.columns:
        return b["norm_f1_p2_5"], b["norm_f1_p97_5"]
    return b["norm_f1_lower"], b["norm_f1_upper"]


def read_csv_at_run(rel_path, run):
    """Re-run CSV for `run="new"`, its `_taxonomy` counterpart for `run="first"`."""
    if run == "first":
        rel_path = rel_path[:-len(".csv")] + "_taxonomy.csv"
    return pd.read_csv(os.path.join(REPO, rel_path))


def make_fig45(run):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.95),
                             layout="constrained")
    y_top = 0.0

    for ax, panel in zip(axes, SELECTIVITY_PANELS):
        df = read_csv_at_run(panel["metrics"], run)
        boot = pd.read_csv(os.path.join(REPO, panel["boot"]))

        for tax in TAXONOMIES:
            sub = df[df.taxonomy == tax].sort_values("layer")
            b = boot[boot.taxonomy == tax].sort_values("layer")
            if sub.empty:
                continue
            c = COLORS[tax]
            ax.plot(sub.layer, sub.normalized_macro_f1,
                    marker=MARKERS[tax], color=c, label=PRETTY[tax])
            y_top = max(y_top, float(sub.normalized_macro_f1.max()))
            if run == "first" and not b.empty:
                # 95% percentile CI of a 10,000-iteration test-set bootstrap.
                lo, hi = boot_ci(b)
                ax.fill_between(b.layer, lo, hi,
                                color=c, alpha=0.18, linewidth=0)
                y_top = max(y_top, float(hi.max()))

        style_axes(ax)
        ax.set_box_aspect(0.92)          # near-square panels
        ax.set_xlabel("Layer")
        # No legend title: at half width it cost four lines and covered the
        # curves. What H is belongs in the caption.
        ax.legend(loc="lower right")
        label_panel(ax, panel)

    # One scale for both models: the panels are meant to be read against each
    # other, so the eye must not have to rescale between them.
    for ax in axes:
        ax.set_ylim(0, y_top * 1.04)
    axes[0].set_ylabel("Chance-corrected macro-F1")

    save(fig, f"fig4_5_selectivity_{run}run_letters")


# =============================================================================
# Figures 7 + 8 - RSA on Llama: alignment vs shuffle, and the lexical partial
# =============================================================================

RSA_LLAMA = "results/Llama-2-7b-chat-hf_generated_prompts/rsa_analysis/rsa_robust_metrics.csv"


def make_fig78(run):
    # Unlike the selectivity CSVs, this file carries its own dispersion
    # (`std_corr`), so both runs are internally consistent.
    df = read_csv_at_run(RSA_LLAMA, run)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.95),
                             layout="constrained")
    ax_a, ax_b = axes

    for tax in TAXONOMIES:
        sub = df[df.taxonomy == tax].sort_values("layer")
        c = COLORS[tax]
        ax_a.plot(sub.layer, sub.mean_corr, marker=MARKERS[tax], color=c,
                  label=PRETTY[tax])
        ax_a.fill_between(sub.layer, sub.mean_corr - sub.std_corr,
                          sub.mean_corr + sub.std_corr,
                          color=c, alpha=0.16, linewidth=0)
        ax_a.plot(sub.layer, sub.shuffle_corr, linestyle=":", color=c,
                  linewidth=0.9, alpha=0.75)

        ax_b.plot(sub.layer, sub.lexical_partial, marker=MARKERS[tax], color=c,
                  label=PRETTY[tax])
        # Shaded gap = what the Bag-of-Words partial removes.
        ax_b.fill_between(sub.layer, sub.mean_corr, sub.lexical_partial,
                          color=c, alpha=0.13, linewidth=0)

    ax_a.set_ylabel("Spearman correlation")
    ax_b.set_ylabel("Partial correlation, words removed")

    for ax, panel in zip(axes,
                         [dict(letter="(a)", name="taxonomy vs shuffle"),
                          dict(letter="(b)", name="lexical control")]):
        style_axes(ax)
        ax.set_box_aspect(0.92)
        ax.set_xlabel("Layer")
        label_panel(ax, panel)

    # Same scale in both panels: the claim is that partialling out the lexical
    # RDM barely lowers the curves, which is only visible on a shared axis.
    lo = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
    hi = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    ax_a.set_ylim(lo, hi)
    ax_b.set_ylim(lo, hi)

    # One legend for the whole figure; the control line gets its own entry so
    # the dotted curves are never identified by colour alone.
    handles, labels = ax_a.get_legend_handles_labels()
    ctrl = mpl.lines.Line2D([], [], color="0.45", linestyle=":", linewidth=0.9)
    fig.legend(handles + [ctrl], labels + ["shuffled-label control"],
               loc="outside lower center", ncol=4, fontsize=7.5,
               frameon=False, columnspacing=1.6)

    save(fig, f"fig7_8_rsa_{run}run_letters")


# =============================================================================
# Figure 11 - plateau selectivity, prompt token vs generated token, 7 models
# =============================================================================
# Three panels side by side would leave 1.7in per panel for seven rotated model
# names. Stacking the domains and sharing one x-axis gives the labels the full
# column width.

SLOTS_CSV = "results_human_centric_rerun/04_capture_slots/capture_slots_ekman_plateau.csv"

MODEL_ORDER = ["Llama-2-7b", "Llama-3.1-8B", "Qwen2.5-14B", "Qwen3-14B",
               "GLM-4-32B", "gemma-4-12b", "Qwen3.6-27B"]
# Split over two lines: seven names set on one line collide at any size that is
# still readable in print.
MODEL_TICK = {
    "Llama-2-7b": "Llama-2\n7b (dense)",
    "Llama-3.1-8B": "Llama-3.1\n8B (dense)",
    "Qwen2.5-14B": "Qwen2.5\n14B (dense)",
    "Qwen3-14B": "Qwen3\n14B (dense)",
    "GLM-4-32B": "GLM-4\n32B (dense)",
    "gemma-4-12b": "gemma-4\n12b (MoE)",
    "Qwen3.6-27B": "Qwen3.6\n27B (hybrid)",
}
DOMAIN_ORDER = [("ai_centric", "AI-centric"),
                ("human_3rd", "human 3rd-person"),
                ("human_conversation", "human conversational")]

POS_COLORS = {"prompt": "tab:blue", "gen": "tab:red"}


def make_fig11():
    df = pd.read_csv(os.path.join(REPO, SLOTS_CSV))
    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH_IN, 4.1), sharex=True,
                             layout="constrained")
    x = range(len(MODEL_ORDER))
    w = 0.38

    for ax, (key, pretty), letter in zip(axes, DOMAIN_ORDER, ["(a)", "(b)", "(c)"]):
        sub = df[df.domain == key].set_index("model").reindex(MODEL_ORDER)
        ax.bar([i - w / 2 for i in x], sub.prompt_last, w,
               color=POS_COLORS["prompt"], label="prompt last-token")
        ax.bar([i + w / 2 for i in x], sub.gen_last, w,
               color=POS_COLORS["gen"], label="generated last-token")
        ax.grid(False)                    # whitegrid would draw both axes
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 0.52)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        label_panel(ax, dict(letter=letter))

    axes[1].set_ylabel("Plateau chance-corrected macro-F1")
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels([MODEL_TICK[m] for m in MODEL_ORDER], fontsize=7)
    axes[0].legend(loc="upper right", ncol=2, fontsize=7.5)

    save(fig, "fig11_slots_7models_letters")


# =============================================================================
# Appendix - same treatment, paired the way the main-text figures now are
# =============================================================================

def make_appendix_lr_raw(run):
    """Raw macro-F1 vs shuffled-label control (appendix Figs 21-22)."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.95),
                             layout="constrained")
    for ax, panel in zip(axes, SELECTIVITY_PANELS):
        df = read_csv_at_run(panel["metrics"], run)
        for tax in TAXONOMIES:
            sub = df[df.taxonomy == tax].sort_values("layer")
            ax.plot(sub.layer, sub.macro_f1, marker=MARKERS[tax],
                    color=COLORS[tax], label=PRETTY[tax])
            ax.plot(sub.layer, sub.control_macro_f1, linestyle="--",
                    color=COLORS[tax], linewidth=0.9, alpha=0.7)
        style_axes(ax)
        ax.set_box_aspect(0.92)
        ax.set_xlabel("Layer")
        ax.set_ylim(bottom=0)
        label_panel(ax, panel)
    axes[0].set_ylabel("Macro-F1")

    handles, labels = axes[0].get_legend_handles_labels()
    ctrl = mpl.lines.Line2D([], [], color="0.45", linestyle="--", linewidth=0.9)
    fig.legend(handles + [ctrl], labels + ["shuffled-label control"],
               loc="outside lower center", ncol=4, fontsize=7.5,
               frameon=False, columnspacing=1.6)
    save(fig, f"figA_lr_raw_{run}run")


def make_appendix_imbalance(run):
    """Normalized label entropy per taxonomy (appendix Figs 19-20)."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.5),
                             layout="constrained")
    for ax, panel in zip(axes, SELECTIVITY_PANELS):
        df = read_csv_at_run(panel["metrics"], run)
        ent = [df[df.taxonomy == t].imbalance_entropy.iloc[0] for t in TAXONOMIES]
        bars = ax.bar([PRETTY[t] for t in TAXONOMIES], ent,
                      color=[COLORS[t] for t in TAXONOMIES], width=0.62)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.16)
        ax.grid(False)                    # whitegrid would draw both axes
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_axisbelow(True)
        label_panel(ax, panel)
    axes[0].set_ylabel("Normalized entropy\n(0 = imbalanced, 1 = balanced)",
                       fontsize=8)
    save(fig, f"figA_class_imbalance_{run}run")


RSA_QWEN = "results/Qwen2.5-14B-Instruct_generated_prompts/rsa_analysis/rsa_robust_metrics.csv"


def make_appendix_rsa_qwen():
    """RSA on Qwen (appendix Figs 28-29), paired like Figures 7+8."""
    df = pd.read_csv(os.path.join(REPO, RSA_QWEN))   # only ever the first run
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.95),
                             layout="constrained")
    ax_a, ax_b = axes
    for tax in TAXONOMIES:
        sub = df[df.taxonomy == tax].sort_values("layer")
        c = COLORS[tax]
        ax_a.plot(sub.layer, sub.mean_corr, marker=MARKERS[tax], color=c,
                  label=PRETTY[tax])
        ax_a.fill_between(sub.layer, sub.mean_corr - sub.std_corr,
                          sub.mean_corr + sub.std_corr, color=c,
                          alpha=0.16, linewidth=0)
        ax_a.plot(sub.layer, sub.shuffle_corr, linestyle=":", color=c,
                  linewidth=0.9, alpha=0.75)
        ax_b.plot(sub.layer, sub.lexical_partial, marker=MARKERS[tax], color=c,
                  label=PRETTY[tax])
        ax_b.fill_between(sub.layer, sub.mean_corr, sub.lexical_partial,
                          color=c, alpha=0.13, linewidth=0)

    ax_a.set_ylabel("Spearman correlation")
    ax_b.set_ylabel("Partial correlation, words removed")
    for ax, panel in zip(axes,
                         [dict(letter="(a)", name="taxonomy vs shuffle"),
                          dict(letter="(b)", name="lexical control")]):
        style_axes(ax)
        ax.set_box_aspect(0.92)
        ax.set_xlabel("Layer")
        label_panel(ax, panel)
    lo = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
    hi = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
    ax_a.set_ylim(lo, hi)
    ax_b.set_ylim(lo, hi)

    handles, labels = ax_a.get_legend_handles_labels()
    ctrl = mpl.lines.Line2D([], [], color="0.45", linestyle=":", linewidth=0.9)
    fig.legend(handles + [ctrl], labels + ["shuffled-label control"],
               loc="outside lower center", ncol=4, fontsize=7.5,
               frameon=False, columnspacing=1.6)
    save(fig, "figA_rsa_qwen")


# =============================================================================
# Appendix - label frequency countplots (Llama only)
# =============================================================================
# Counts come from data/04_annotated/annotated_results.jsonl (first run, Llama).
# One custom_id is duplicated in that file, so one GoEmotions bar is off by one.

ANNOTATIONS_JSONL = "data/04_annotated/annotated_results.jsonl"
CUSTOM_ID = re.compile(r"response-(\d+)-(.+)")


def _emotions_from_record(record):
    """Same parse as scripts/annotations_and_activations_merge.py."""
    try:
        for item in record.get("response", {}).get("body", {}).get("output", []):
            if item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text")
                    if not text:
                        return None
                    clean = text.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean).get("emotions", [])
                return None
    except (json.JSONDecodeError, AttributeError, IndexError):
        return None
    return None


def make_appendix_countplots():
    rows = {}
    with open(os.path.join(REPO, ANNOTATIONS_JSONL), encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            m = CUSTOM_ID.match(rec.get("custom_id", "") or "")
            if not m or m.group(2) not in TAXONOMIES:
                continue
            lst = _emotions_from_record(rec)
            if lst:
                rows.setdefault(int(m.group(1)), {})[m.group(2)] = lst

    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=TAXONOMIES)
    print(f"  parsed {len(df)} annotated responses")

    for tax in TAXONOMIES:
        primary = (df[tax].dropna()
                   .apply(lambda x: x[0] if isinstance(x, list) and x else None)
                   .dropna())
        counts = primary.value_counts()
        n = len(counts)
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.85,
                                        max(1.6, 0.155 * n + 0.55)),
                               layout="constrained")
        palette = plt.get_cmap("viridis")([i / max(n - 1, 1) for i in range(n)])
        ax.barh(range(n), counts.values, color=palette, height=0.78)
        ax.set_yticks(range(n))
        ax.set_yticklabels(counts.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_ylabel("Emotion label")
        ax.set_xlim(0, counts.values.max() * 1.13)
        for i, v in enumerate(counts.values):
            ax.text(v + counts.values.max() * 0.01, i, str(v),
                    va="center", fontsize=7)
        # These were the one family drawn on `sns.set_style("white")` + despine,
        # not on whitegrid, so they keep their own ground.
        ax.grid(False)
        sns.despine(ax=ax)
        save(fig, f"figA_countplot_{tax}_llama")


def main():
    # set_theme() resets rcParams, so the type scale goes on afterwards.
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(RC)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Figure 3:")
    make_fig3()
    for run in ("first", "new"):
        print(f"Figures 4+5 [{run} run]:")
        make_fig45(run)
        print(f"Figures 7+8 [{run} run]:")
        make_fig78(run)

    print("Figure 11:")
    make_fig11()

    print("Appendix - raw macro-F1 vs control:")
    make_appendix_lr_raw("first")
    print("Appendix - class imbalance:")
    make_appendix_imbalance("first")
    print("Appendix - RSA on Qwen:")
    make_appendix_rsa_qwen()
    print("Appendix - label countplots (Llama):")
    make_appendix_countplots()


if __name__ == "__main__":
    main()
