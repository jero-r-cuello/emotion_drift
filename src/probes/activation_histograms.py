"""
============================================================================
ACTIVATION HISTOGRAMS — AI-centric vs Human-centric (raw, last-token)
============================================================================

Exploratory script to compare the distribution of RAW last-token activation
values between AI-centric (generated_prompts) and Human-centric (andyzou
emotion_query) stimuli, conditioning on the SAME response emotion.

No normalization / standardization / scaling is applied to the activations:
they are taken exactly as stored (raw float32 vectors). The only optional
normalization is at the *plot* level (`--density`, histogram area = 1), which
is purely cosmetic so distributions with different sample counts remain
comparable — it never touches the activation values themselves.

Always uses `last_token_activation`.

The "response emotion" is the primary annotated label (first element of the
taxonomy list, e.g. ekman_basic_emotions) attached to each model RESPONSE.

Designed to escalate gradually:
  --mode pair     : pick ONE prompt from each dataset (same response emotion)
                    and overlay the histogram of their two activation vectors.
                    (smallest possible comparison — sanity check)
  --mode dataset  : pool ALL last-token activations from every prompt in each
                    dataset matching the emotion, and overlay the two pooled
                    distributions. (full-scale comparison)

Usage examples
--------------
  # List how many prompts each dataset has per response emotion, then exit
  python activation_histograms.py --list-emotions

  # Single-pair sanity check, layer 16, emotion picked automatically
  python activation_histograms.py --mode pair --layer 16

  # Single pair, explicit emotion and explicit row indices
  python activation_histograms.py --mode pair --layer 16 --emotion joy \
      --index-ai 0 --index-human 0

  # Full dataset distributions for one emotion across several layers
  python activation_histograms.py --mode dataset --emotion joy --layers 8 16 24

  # Full dataset distributions for ALL shared emotions, one layer
  # (one separate figure per emotion)
  python activation_histograms.py --mode dataset --layer 16 --all-emotions

  # ALL emotions pooled into a SINGLE comparison (no emotion conditioning)
  python activation_histograms.py --mode dataset --layer 16 --combine-emotions

Emotion conditioning options
----------------------------
  --emotion <name>    : one specific response emotion
  (default)           : auto-pick the shared emotion with most prompts
  --all-emotions      : iterate emotions, one figure per emotion
  --combine-emotions  : pool every emotion into one AI-vs-Human figure
============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_USED = "Llama-2-7b-chat-hf"

PATH_AI = (
    "data/03_activations/"
    "generated_prompts_Llama-2-7b-chat-hf_20251014_203636_"
    "FINAL_WITH_RATINGS_AND_CATS.pkl"
)
PATH_HUMAN = (
    "data/03_activations/"
    "MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
)

NAME_AI = "AI-centric"
NAME_HUMAN = "Human-centric"
COLOR_AI = "#1f77b4"
COLOR_HUMAN = "#ff7f0e"

ACTIVATION_COL = "last_token_activation"
DEFAULT_TAXONOMY = "ekman_basic_emotions"

# Common candidates for the column holding the raw prompt text (only used to
# print a snippet in pair mode for context).
PROMPT_COL_CANDIDATES = ["prompt", "situation", "prompt_text", "generated_prompt"]

OUTPUT_DIR = os.path.join("figures", f"activation_histograms_{LLM_USED}")


# ============================================================================
# DATA HELPERS
# ============================================================================
def primary_label(labels):
    """First element of a taxonomy list = primary response emotion, else None."""
    if isinstance(labels, list) and len(labels) > 0:
        return labels[0]
    return None


def find_prompt_col(df):
    for c in PROMPT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def get_last_token_vector(row, layer):
    """Return the raw last-token activation vector for a given layer, or None."""
    try:
        nested = row["activations"]
        if layer not in nested.index:
            return None
        vec = nested.loc[layer, ACTIVATION_COL]
        if isinstance(vec, np.ndarray):
            return vec
    except Exception:
        return None
    return None


def filter_by_emotion(df, taxonomy, emotion):
    """Rows whose PRIMARY response label under `taxonomy` equals `emotion`."""
    mask = df[taxonomy].apply(lambda x: primary_label(x) == emotion)
    return df[mask]


def emotion_counts(df, taxonomy):
    """Series: primary response emotion -> count of prompts."""
    prim = df[taxonomy].apply(primary_label).dropna()
    return prim.value_counts()


def stack_layer_activations(df, layer):
    """Stack all last-token vectors for `layer` across rows -> (N, D) matrix."""
    vecs = []
    for _, row in df.iterrows():
        v = get_last_token_vector(row, layer)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.stack(vecs)


# ============================================================================
# PLOTTING
# ============================================================================
def _finalize(ax, title, density):
    ax.set_xlabel("Raw activation value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_pair(ai_vec, human_vec, emotion, layer, bins, density,
              ai_text=None, human_text=None):
    """Overlay the histogram of two single activation vectors."""
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(ai_vec, bins=bins, density=density, alpha=0.55,
            color=COLOR_AI, label=f"{NAME_AI} (D={ai_vec.shape[0]})")
    ax.hist(human_vec, bins=bins, density=density, alpha=0.55,
            color=COLOR_HUMAN, label=f"{NAME_HUMAN} (D={human_vec.shape[0]})")

    title = (f"Last-token activation distribution — single pair\n"
             f"emotion='{emotion}'  |  layer={layer}  |  {LLM_USED}")
    _finalize(ax, title, density)

    if ai_text or human_text:
        caption = ""
        if ai_text:
            caption += f"AI:    {ai_text[:120]}\n"
        if human_text:
            caption += f"Human: {human_text[:120]}"
        fig.text(0.5, -0.02, caption, ha="center", fontsize=8,
                 wrap=True, family="monospace")

    fig.tight_layout()
    fname = f"pair_{emotion}_layer{layer}.png"
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def plot_dataset(ai_mat, human_mat, emotion, layer, bins, density):
    """Overlay the pooled histogram of all activation values per dataset."""
    ai_vals = ai_mat.ravel()
    human_vals = human_mat.ravel()

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(ai_vals, bins=bins, density=density, alpha=0.55, color=COLOR_AI,
            label=f"{NAME_AI} (n={ai_mat.shape[0]} prompts, {ai_vals.size:,} vals)")
    ax.hist(human_vals, bins=bins, density=density, alpha=0.55, color=COLOR_HUMAN,
            label=f"{NAME_HUMAN} (n={human_mat.shape[0]} prompts, {human_vals.size:,} vals)")

    title = (f"Pooled last-token activation distribution — full datasets\n"
             f"emotion='{emotion}'  |  layer={layer}  |  {LLM_USED}")
    _finalize(ax, title, density)

    # Quick distributional summary in the corner (raw, unnormalized).
    summ = (f"{NAME_AI}: μ={ai_vals.mean():.3f} σ={ai_vals.std():.3f} "
            f"[{ai_vals.min():.2f}, {ai_vals.max():.2f}]\n"
            f"{NAME_HUMAN}: μ={human_vals.mean():.3f} σ={human_vals.std():.3f} "
            f"[{human_vals.min():.2f}, {human_vals.max():.2f}]")
    ax.text(0.02, 0.97, summ, transform=ax.transAxes, va="top", fontsize=8,
            family="monospace", bbox=dict(facecolor="white", alpha=0.7, pad=4))

    fig.tight_layout()
    fname = f"dataset_{emotion}_layer{layer}.png"
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pair", "dataset"], default="pair")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY,
                    help="Response-emotion taxonomy column (primary label used).")
    ap.add_argument("--emotion", default=None,
                    help="Response emotion to condition on. If omitted, the most "
                         "frequent emotion shared by both datasets is used.")
    ap.add_argument("--all-emotions", action="store_true",
                    help="Run a SEPARATE comparison for every emotion shared by "
                         "both datasets (one figure per emotion).")
    ap.add_argument("--combine-emotions", action="store_true",
                    help="Pool ALL emotions together into a SINGLE AI-vs-Human "
                         "comparison (no emotion conditioning at all). "
                         "Takes precedence over --emotion / --all-emotions.")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Multiple layers (overrides --layer).")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--density", dest="density", action="store_true", default=True,
                    help="Plot density (area=1) so different N are comparable [default].")
    ap.add_argument("--no-density", dest="density", action="store_false",
                    help="Plot raw counts instead of density.")
    ap.add_argument("--index-ai", type=int, default=0,
                    help="(pair mode) row index within AI-centric matches.")
    ap.add_argument("--index-human", type=int, default=0,
                    help="(pair mode) row index within Human-centric matches.")
    ap.add_argument("--list-emotions", action="store_true",
                    help="Print per-dataset emotion counts and exit.")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...")
    df_ai = pd.read_pickle(PATH_AI)
    df_human = pd.read_pickle(PATH_HUMAN)
    print(f"  {NAME_AI}:    {len(df_ai)} rows")
    print(f"  {NAME_HUMAN}: {len(df_human)} rows")

    tax = args.taxonomy
    for d, name in [(df_ai, NAME_AI), (df_human, NAME_HUMAN)]:
        if tax not in d.columns:
            raise SystemExit(f"Taxonomy column '{tax}' not in {name} dataframe. "
                             f"Available: {list(d.columns)}")

    counts_ai = emotion_counts(df_ai, tax)
    counts_human = emotion_counts(df_human, tax)
    shared = sorted(set(counts_ai.index) & set(counts_human.index))

    print(f"\nPrimary response-emotion counts under '{tax}':")
    print(f"{'emotion':<18}{NAME_AI:>14}{NAME_HUMAN:>16}")
    for emo in sorted(set(counts_ai.index) | set(counts_human.index)):
        a = int(counts_ai.get(emo, 0))
        h = int(counts_human.get(emo, 0))
        flag = "  <-- shared" if emo in shared else ""
        print(f"{emo:<18}{a:>14}{h:>16}{flag}")

    if args.list_emotions:
        return

    # ---- Decide which emotions / layers to run --------------------------------
    # Each item is (emotion_label, sub_ai, sub_human). When emotions are pooled,
    # the label is "all_emotions" and the full datasets are used unfiltered.
    if args.combine_emotions:
        jobs = [("all_emotions", df_ai, df_human)]
        print("\nPooling ALL emotions into a single AI-vs-Human comparison.")
    else:
        if args.all_emotions:
            emotions = shared
        elif args.emotion is not None:
            emotions = [args.emotion]
        else:
            # most frequent emotion shared by both (by min count across datasets)
            if not shared:
                raise SystemExit("No emotion is shared by both datasets under this taxonomy.")
            emotions = [max(shared, key=lambda e: min(counts_ai[e], counts_human[e]))]
            print(f"\nNo --emotion given; auto-selected shared emotion: '{emotions[0]}'")
        jobs = [(e, filter_by_emotion(df_ai, tax, e),
                 filter_by_emotion(df_human, tax, e)) for e in emotions]

    layers = args.layers if args.layers else [args.layer]

    prompt_col_ai = find_prompt_col(df_ai)
    prompt_col_human = find_prompt_col(df_human)

    # ---- Run ------------------------------------------------------------------
    for emotion, sub_ai, sub_human in jobs:
        print(f"\n=== emotion='{emotion}' | {NAME_AI}: {len(sub_ai)} prompts, "
              f"{NAME_HUMAN}: {len(sub_human)} prompts ===")

        if len(sub_ai) == 0 or len(sub_human) == 0:
            print("  skipping: one dataset has no prompts for this emotion.")
            continue

        for layer in layers:
            if args.mode == "pair":
                if args.index_ai >= len(sub_ai) or args.index_human >= len(sub_human):
                    print(f"  layer {layer}: index out of range "
                          f"(ai max {len(sub_ai)-1}, human max {len(sub_human)-1}); skipping.")
                    continue
                row_ai = sub_ai.iloc[args.index_ai]
                row_human = sub_human.iloc[args.index_human]
                vec_ai = get_last_token_vector(row_ai, layer)
                vec_human = get_last_token_vector(row_human, layer)
                if vec_ai is None or vec_human is None:
                    print(f"  layer {layer}: missing activation for the picked pair; skipping.")
                    continue
                ai_text = row_ai[prompt_col_ai] if prompt_col_ai else None
                human_text = row_human[prompt_col_human] if prompt_col_human else None
                plot_pair(vec_ai, vec_human, emotion, layer, args.bins,
                          args.density, ai_text, human_text)

            else:  # dataset
                mat_ai = stack_layer_activations(sub_ai, layer)
                mat_human = stack_layer_activations(sub_human, layer)
                if mat_ai is None or mat_human is None:
                    print(f"  layer {layer}: no activations found; skipping.")
                    continue
                plot_dataset(mat_ai, mat_human, emotion, layer, args.bins, args.density)

    print(f"\nDone. Figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
