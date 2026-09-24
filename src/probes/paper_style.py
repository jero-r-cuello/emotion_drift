"""Shared figure style for everything that ends up in the manuscript.

Import this from any plotting script so re-runs come out paper-ready instead of
needing a second pass. It encodes four conventions:

1. Figures are built at the width they occupy on the page (``TEXTWIDTH_IN``),
   so a font declared at 8 pt is 8 pt in the compiled PDF. Nothing is rescaled
   by ``\\includegraphics``, which is what makes legibility checkable.
2. No in-figure titles. The caption already says what the figure is.
3. Variable names are spelled out: "GoEmotions", not "go_emotions".
4. Colours are tab10 on a seaborn `whitegrid` ground. A taxonomy keeps one
   colour across every figure, and each taxonomy has its own marker shape, so
   identity never rests on colour alone.

Usage:

    from paper_style import apply_paper_style, pretty, COLORS, MARKERS
    apply_paper_style()
"""

import matplotlib as mpl
import seaborn as sns

# The conference style sets textwidth=5.5in; body text is 10 pt, captions 9 pt.
TEXTWIDTH_IN = 5.5

TAXONOMIES = ["ekman_basic_emotions", "plutchik_wheel", "go_emotions"]

PRETTY = {
    "ekman_basic_emotions": "Ekman",
    "plutchik_wheel": "Plutchik",
    "go_emotions": "GoEmotions",
    # stimulus domains
    "generated_prompts": "AI-centric",
    "ai_centric": "AI-centric",
    "human_centric": "human 3rd-person",
    "human_3rd": "human 3rd-person",
    "human_conversation": "human conversational",
    "human_conv": "human conversational",
    # read-out positions
    "last_token_activation": "prompt last-token",
    "gen_last_token": "generated last-token",
    "gen_last_token_activation": "generated last-token",
}

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
    "pdf.fonttype": 42,      # embed TrueType, not Type-3
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}


def apply_paper_style():
    # set_theme() resets rcParams, so the type scale has to go on afterwards.
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(RC)


def pretty(name):
    """Display name for a variable: mapped if known, de-underscored otherwise."""
    if name in PRETTY:
        return PRETTY[name]
    return str(name).replace("_", " ")


def heatmap_figsize(n_rows, n_cols, width_frac=1.0):
    """Figure size for an annotated heatmap, at its on-page width.

    Height follows the row count so cells stay tall enough for a 6 pt
    annotation: below about 9 pt per row the numbers stop being readable.
    """
    width = TEXTWIDTH_IN * width_frac
    # ~0.72in of the width goes to the y tick labels and the colourbar.
    cell_w = (width - 0.72) / max(n_cols, 1)
    height = min(max(n_rows * cell_w * 0.62, 1.8), 7.5) + 0.75
    return (width, height)


def annot_size(n_rows, n_cols):
    """Annotation font size that still fits the cell at the printed size."""
    return 6.0 if (n_rows > 12 or n_cols > 12) else 7.0


# PNG is what the manuscript includes. Physical size is fixed by `figsize`,
# so this only sets the raster resolution.
SAVE_DPI = 400


def save_paper_figure(fig, path_without_ext):
    """Write the PNG that the manuscript includes."""
    fig.savefig(path_without_ext + ".png", dpi=SAVE_DPI)
