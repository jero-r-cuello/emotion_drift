"""
============================================================================
STAGE B — MATCHED-BY-COMPLEXITY CROSS-DOMAIN PROBE TRANSFER TEST
(Refactored version with checkpoints + CLI + incremental output)
============================================================================

This is a refactored version of the Stage B test designed to be RESUMABLE
and SPLITTABLE. Key changes from v1:

1. CHECKPOINTS: After each (design × taxonomy) combo finishes, results are
   appended to a master CSV and a checkpoint file is updated. If the
   process dies, re-running skips combos already done.

2. CLI ARGUMENTS: Run a specific design or taxonomy only.
   Usage examples:
     python stage_b_test.py                          # run everything
     python stage_b_test.py --design mtld_median     # one design only
     python stage_b_test.py --taxonomy ekman_basic_emotions  # one tax only
     python stage_b_test.py --design fk_q1q4 --taxonomy plutchik_wheel
     python stage_b_test.py --layers 13 14 15        # specific layers only
     python stage_b_test.py --bootstrap 1000         # faster, less precise
     python stage_b_test.py --reset                  # ignore checkpoints

3. INCREMENTAL PLOTS: Each (design × taxonomy) combo generates its own
   plot immediately after its results are written. No waiting until end.

4. SAFER BOOTSTRAP: --bootstrap N controls iterations (default 10000).
   For a quick exploratory pass, use --bootstrap 1000 (10x faster).

Recommended workflow for slow runs:
-----------------------------------
First pass (fast, see if anything works):
  python stage_b_test.py --bootstrap 1000 --layers 13 14 15

If results look sensible, full pass design-by-design:
  python stage_b_test.py --design mtld_median        # ~most robust, run first
  python stage_b_test.py --design fk_q1q4
  python stage_b_test.py --design length_q1q4       # ~smallest cells, run last

Generate cross-design summary after all 3 designs complete:
  python stage_b_test.py --summary-only
============================================================================
"""

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.utils import resample
import re


# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_USED = "Llama-2-7b-chat-hf"
MODELS_DIR_BASE = "models"

PATH_AI = (
    "data/03_activations/"
    "generated_prompts_Llama-2-7b-chat-hf_20251014_203636_"
    "FINAL_WITH_RATINGS_AND_CATS.pkl"
)
PATH_HUMAN = (
    "data/03_activations/"
    "MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
)

NAME_AI = "generated_prompts"
NAME_HUMAN = "human_centric"

TAXONOMIES = ["ekman_basic_emotions"]
ACTIVATION_COL = "last_token_activation"
PROMPT_COL = "prompt"

LAYERS_DEFAULT = list(range(33))
BOOTSTRAP_DEFAULT = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

OUTPUT_DIR = os.path.join("figures", f"stage_b_test_{LLM_USED}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "_checkpoint.json")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "stage_b_full_results.csv")


# ============================================================================
# STRUCTURAL METRICS
# ============================================================================
def tokenize_simple(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[\w']+\b", text.lower())


def split_sentences_simple(text):
    if not isinstance(text, str):
        return []
    text = re.sub(
        r"\b(Mr|Mrs|Dr|Ms|St|Jr|Sr|vs|etc|e\.g|i\.e)\.",
        r"\1<DOT>", text,
    )
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.replace("<DOT>", ".").strip() for s in sents if s.strip()]
    return sents


def type_token_ratio(tokens):
    if len(tokens) == 0:
        return np.nan
    return len(set(tokens)) / len(tokens)


def mtld(tokens, threshold=0.72):
    if len(tokens) < 50:
        return type_token_ratio(tokens)

    def _factor_count(toks):
        types = set()
        factors = 0.0
        token_count = 0
        running_ttr = 1.0
        for t in toks:
            types.add(t)
            token_count += 1
            running_ttr = len(types) / token_count
            if running_ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0
                running_ttr = 1.0
        if token_count > 0:
            partial = (1.0 - running_ttr) / (1.0 - threshold)
            factors += partial
        return len(toks) / factors if factors > 0 else len(toks)

    forward = _factor_count(tokens)
    backward = _factor_count(tokens[::-1])
    return (forward + backward) / 2.0


def count_syllables_word(word):
    word = word.lower()
    if len(word) == 0:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def flesch_kincaid_grade(text):
    sents = split_sentences_simple(text)
    tokens = tokenize_simple(text)
    if len(sents) == 0 or len(tokens) == 0:
        return np.nan
    syllables = sum(count_syllables_word(t) for t in tokens)
    wps = len(tokens) / len(sents)
    spw = syllables / len(tokens)
    return 0.39 * wps + 11.8 * spw - 15.59


def compute_complexity_metrics(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"n_tokens": np.nan, "fk_grade": np.nan, "mtld": np.nan}
    tokens = tokenize_simple(text)
    return {
        "n_tokens": len(tokens),
        "fk_grade": flesch_kincaid_grade(text),
        "mtld": mtld(tokens),
    }


# ============================================================================
# DATA EXTRACTION
# ============================================================================
def get_activation_and_labels(df, layer, taxonomy):
    out = {}
    for idx, row in enumerate(df.itertuples()):
        try:
            nested = row.activations
            if layer not in nested.index:
                continue
            act = nested.loc[layer, ACTIVATION_COL]
            if not isinstance(act, np.ndarray):
                continue
            labels = getattr(row, taxonomy)
            if not isinstance(labels, list) or len(labels) == 0:
                continue
            out[idx] = (act, labels[0])
        except Exception:
            continue
    return out


def load_probe(dataset_prefix, taxonomy, layer):
    fname = f"{dataset_prefix}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
    path = os.path.join(MODELS_DIR_BASE, fname)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# ============================================================================
# BOOTSTRAP
# ============================================================================
def stratified_bootstrap_f1(y_true, y_pred, n_iter, ci, seed):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 5:
        return np.nan, np.nan, np.nan

    try:
        _, counts = np.unique(y_true, return_counts=True)
        stratified = np.all(counts >= 2)
    except Exception:
        stratified = False

    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_iter):
        iter_seed = rng.randint(0, 2**32 - 1)
        try:
            if stratified:
                yt, yp = resample(
                    y_true, y_pred, replace=True,
                    stratify=y_true, random_state=iter_seed,
                )
            else:
                yt, yp = resample(
                    y_true, y_pred, replace=True,
                    random_state=iter_seed,
                )
            stats.append(
                f1_score(yt, yp, average="macro", zero_division=0)
            )
        except Exception:
            continue

    if len(stats) == 0:
        return np.nan, np.nan, np.nan

    alpha = (1.0 - ci) / 2.0
    mean = float(np.mean(stats))
    low = float(np.percentile(stats, alpha * 100))
    high = float(np.percentile(stats, (1.0 - alpha) * 100))
    return mean, low, high


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
def load_checkpoint():
    """Load checkpoint dict mapping (design, taxonomy) → done."""
    if not os.path.exists(CHECKPOINT_FILE):
        return {}
    try:
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def combo_key(design, taxonomy):
    return f"{design}::{taxonomy}"


def append_results_to_csv(rows, results_file):
    """Append rows to the master results CSV. Creates file if absent."""
    df = pd.DataFrame(rows)
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, mode="w", header=True, index=False)


# ============================================================================
# COMBO RUNNER (single design × taxonomy)
# ============================================================================
def run_combo(
    design_name, design_cfg, taxonomy,
    df_ai, df_human, ai_metrics, human_metrics,
    layers, bootstrap_iter,
):
    """Run one design × taxonomy combo. Returns list of result rows."""
    metric_col = design_cfg["metric"]
    low_thr = design_cfg["low_threshold"]
    high_thr = design_cfg["high_threshold"]

    if low_thr == high_thr:
        ai_low_mask = ai_metrics[metric_col] <= low_thr
        ai_high_mask = ai_metrics[metric_col] > high_thr
        human_low_mask = human_metrics[metric_col] <= low_thr
        human_high_mask = human_metrics[metric_col] > high_thr
    else:
        ai_low_mask = ai_metrics[metric_col] <= low_thr
        ai_high_mask = ai_metrics[metric_col] >= high_thr
        human_low_mask = human_metrics[metric_col] <= low_thr
        human_high_mask = human_metrics[metric_col] >= high_thr

    ai_low_indices = set(np.where(ai_low_mask)[0])
    ai_high_indices = set(np.where(ai_high_mask)[0])
    human_low_indices = set(np.where(human_low_mask)[0])
    human_high_indices = set(np.where(human_high_mask)[0])

    print(
        f"    Cell sizes: "
        f"AI low={len(ai_low_indices)} high={len(ai_high_indices)} | "
        f"Human low={len(human_low_indices)} high={len(human_high_indices)}"
    )

    rows = []
    for layer in tqdm(layers, desc=f"    Layers", leave=False):
        ai_data = get_activation_and_labels(df_ai, layer, taxonomy)
        human_data = get_activation_and_labels(df_human, layer, taxonomy)
        if not ai_data or not human_data:
            continue

        probe_ai = load_probe(NAME_AI, taxonomy, layer)
        probe_human = load_probe(NAME_HUMAN, taxonomy, layer)
        if probe_ai is None or probe_human is None:
            continue

        for cplx_label, ai_idx_set, human_idx_set in [
            ("matched_low", ai_low_indices, human_low_indices),
            ("matched_high", ai_high_indices, human_high_indices),
        ]:
            ai_sub = {
                i: v for i, v in ai_data.items() if i in ai_idx_set
            }
            human_sub = {
                i: v for i, v in human_data.items() if i in human_idx_set
            }
            if len(ai_sub) < 10 or len(human_sub) < 10:
                continue

            X_ai_sub = np.stack([v[0] for v in ai_sub.values()])
            y_ai_sub = np.array([v[1] for v in ai_sub.values()])
            X_human_sub = np.stack([v[0] for v in human_sub.values()])
            y_human_sub = np.array([v[1] for v in human_sub.values()])

            for (tr, te, probe, X_test, y_test) in [
                ("AI", "Human", probe_ai, X_human_sub, y_human_sub),
                ("Human", "AI", probe_human, X_ai_sub, y_ai_sub),
                ("AI", "AI", probe_ai, X_ai_sub, y_ai_sub),
                ("Human", "Human", probe_human, X_human_sub, y_human_sub),
            ]:
                try:
                    y_pred = probe.predict(X_test)
                    mean, low, high = stratified_bootstrap_f1(
                        y_test, y_pred,
                        bootstrap_iter, CONFIDENCE_LEVEL,
                        RANDOM_SEED,
                    )
                    rows.append({
                        "design": design_name,
                        "taxonomy": taxonomy,
                        "layer": layer,
                        "complexity_cell": cplx_label,
                        "train_source": tr,
                        "test_source": te,
                        "n_test": len(y_test),
                        "f1_mean": mean,
                        "f1_lower": low,
                        "f1_upper": high,
                    })
                except Exception:
                    continue

    return rows


# ============================================================================
# PLOTS
# ============================================================================
def plot_combo(design_name, taxonomy, design_description):
    """Plot a single (design, taxonomy) result from the master CSV."""
    if not os.path.exists(RESULTS_FILE):
        return
    full_df = pd.read_csv(RESULTS_FILE)
    subset = full_df[
        (full_df["design"] == design_name)
        & (full_df["taxonomy"] == taxonomy)
    ]
    if subset.empty:
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, cell in zip(axes, ["matched_low", "matched_high"]):
        cell_df = subset[subset["complexity_cell"] == cell]
        if cell_df.empty:
            ax.set_title(f"{cell} — no data")
            continue
        for (tr, te), color, ls in [
            (("AI", "AI"), "#1f77b4", "--"),
            (("AI", "Human"), "#1f77b4", "-"),
            (("Human", "Human"), "#ff7f0e", "--"),
            (("Human", "AI"), "#ff7f0e", "-"),
        ]:
            sub = cell_df[
                (cell_df["train_source"] == tr)
                & (cell_df["test_source"] == te)
            ].sort_values("layer")
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["f1_mean"],
                label=f"Train: {tr} → Test: {te}",
                color=color, linestyle=ls, linewidth=2,
            )
            ax.fill_between(
                sub["layer"], sub["f1_lower"], sub["f1_upper"],
                color=color, alpha=0.15,
            )
        ax.set_xlabel("Layer")
        ax.set_title(f"{cell}")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Macro F1 (95% CI bootstrap)")
    plt.suptitle(
        f"Stage B: {design_description}\nTaxonomy: {taxonomy}",
        fontsize=13,
    )
    plt.tight_layout()
    plot_path = os.path.join(
        OUTPUT_DIR, f"stage_b_{design_name}_{taxonomy}.png",
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved: {plot_path}")


# ============================================================================
# SUMMARY
# ============================================================================
def generate_summary():
    """Generate cross-design summary from master CSV. Run whenever you want
    an updated view; can be called standalone via --summary-only."""
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}. Nothing to summarize.")
        return

    results_df = pd.read_csv(RESULTS_FILE)

    # Asymmetry per (design, taxonomy, layer, complexity_cell)
    asym_rows = []
    grouping = ["design", "taxonomy", "layer", "complexity_cell"]
    for keys, g in results_df.groupby(grouping):
        ai_to_h = g[
            (g["train_source"] == "AI") & (g["test_source"] == "Human")
        ]
        h_to_ai = g[
            (g["train_source"] == "Human") & (g["test_source"] == "AI")
        ]
        if len(ai_to_h) == 0 or len(h_to_ai) == 0:
            continue
        f1_aih = ai_to_h["f1_mean"].values[0]
        f1_hai = h_to_ai["f1_mean"].values[0]
        ratio = (f1_aih / f1_hai) if f1_hai > 0 else np.nan
        asym_rows.append({
            "design": keys[0],
            "taxonomy": keys[1],
            "layer": keys[2],
            "complexity_cell": keys[3],
            "f1_AI_to_Human": f1_aih,
            "f1_Human_to_AI": f1_hai,
            "asymmetry_ratio": ratio,
        })
    asym_df = pd.DataFrame(asym_rows)
    asym_df.to_csv(
        os.path.join(OUTPUT_DIR, "stage_b_asymmetry.csv"),
        index=False,
    )

    summary = (
        asym_df.groupby(["design", "complexity_cell"])
        .agg(
            median_asymmetry_ratio=("asymmetry_ratio", "median"),
            mean_asymmetry_ratio=("asymmetry_ratio", "mean"),
            n_cells=("asymmetry_ratio", "count"),
        )
        .reset_index()
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "stage_b_summary.csv"),
        index=False,
    )

    print("\n" + "=" * 70)
    print("CROSS-DESIGN SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR}/stage_b_summary.csv")
    print(f"Saved: {OUTPUT_DIR}/stage_b_asymmetry.csv")

    print("\n" + "-" * 70)
    print("VERDICT GUIDANCE")
    print("-" * 70)
    print(
        "median_asymmetry_ratio interpretation:\n"
        "  > 2.0  → asymmetry strongly persists; structural-β NOT driver\n"
        "  1.5-2  → asymmetry persists; structural-β partial contributor\n"
        "  1.0-1.5 → asymmetry weakened; structural-β plausible explanation\n"
        "  ~ 1.0  → asymmetry collapsed; structural-β likely main driver"
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage B matched-by-complexity test (resumable)."
    )
    parser.add_argument(
        "--design", type=str, default=None,
        help="Run only this design (length_q1q4, fk_q1q4, mtld_median)",
    )
    parser.add_argument(
        "--taxonomy", type=str, default=None,
        help="Run only this taxonomy",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to run (default: L10-20)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=BOOTSTRAP_DEFAULT,
        help=f"Bootstrap iterations (default {BOOTSTRAP_DEFAULT})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore checkpoint, restart from scratch",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip computation, just generate summary from existing CSV",
    )
    args = parser.parse_args()

    if args.summary_only:
        generate_summary()
        return

    if args.reset:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print("Checkpoint and results reset.")

    layers_to_run = args.layers if args.layers else LAYERS_DEFAULT

    print("\n" + "=" * 70)
    print("STAGE B — MATCHED-BY-COMPLEXITY CROSS-DOMAIN TRANSFER")
    print("=" * 70)
    print(f"  Layers:    {layers_to_run}")
    print(f"  Bootstrap: {args.bootstrap} iterations")
    print(f"  Output:    {OUTPUT_DIR}/")

    # Load data
    print("\nLoading datasets...")
    df_ai = pd.read_pickle(PATH_AI)
    df_human = pd.read_pickle(PATH_HUMAN)
    print(f"  AI-centric:    {len(df_ai)} rows")
    print(f"  Human-centric: {len(df_human)} rows")

    if PROMPT_COL not in df_ai.columns:
        raise KeyError(f"Column '{PROMPT_COL}' not in df_ai")
    if PROMPT_COL not in df_human.columns:
        raise KeyError(f"Column '{PROMPT_COL}' not in df_human")

    print("\nComputing complexity metrics per prompt...")
    ai_metrics = pd.DataFrame([
        compute_complexity_metrics(p) for p in df_ai[PROMPT_COL]
    ])
    human_metrics = pd.DataFrame([
        compute_complexity_metrics(p) for p in df_human[PROMPT_COL]
    ])

    pooled = pd.concat([ai_metrics, human_metrics], ignore_index=True)

    designs = {
        "length_q1q4": {
            "metric": "n_tokens",
            "low_threshold": pooled["n_tokens"].quantile(0.25),
            "high_threshold": pooled["n_tokens"].quantile(0.75),
            "description": "Length-matched (n_tokens Q1 vs Q4)",
        },
        "fk_q1q4": {
            "metric": "fk_grade",
            "low_threshold": pooled["fk_grade"].quantile(0.25),
            "high_threshold": pooled["fk_grade"].quantile(0.75),
            "description": "FK-grade-matched (fk_grade Q1 vs Q4)",
        },
        "mtld_median": {
            "metric": "mtld",
            "low_threshold": pooled["mtld"].quantile(0.50),
            "high_threshold": pooled["mtld"].quantile(0.50),
            "description": "MTLD-matched (median split)",
        },
    }

    # Filter designs/taxonomies if specified
    designs_to_run = (
        {args.design: designs[args.design]} if args.design else designs
    )
    if args.design and args.design not in designs:
        print(f"Unknown design: {args.design}")
        print(f"Available: {list(designs.keys())}")
        return
    taxonomies_to_run = [args.taxonomy] if args.taxonomy else TAXONOMIES

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"\nCheckpoint state: {len(checkpoint)} combos already done")

    print("\nDesign thresholds (pooled):")
    for n, c in designs_to_run.items():
        print(f"  {n}: low<={c['low_threshold']:.2f}, "
              f"high>={c['high_threshold']:.2f}")

    # ------------------------------------------------------------------------
    # RUN COMBOS
    # ------------------------------------------------------------------------
    for design_name, cfg in designs_to_run.items():
        print("\n" + "=" * 70)
        print(f"DESIGN: {cfg['description']}")
        print("=" * 70)

        for taxonomy in taxonomies_to_run:
            key = combo_key(design_name, taxonomy)
            if key in checkpoint and not args.reset:
                print(f"  [SKIP] {taxonomy} — already done")
                continue

            print(f"\n  Taxonomy: {taxonomy}")
            rows = run_combo(
                design_name, cfg, taxonomy,
                df_ai, df_human, ai_metrics, human_metrics,
                layers_to_run, args.bootstrap,
            )

            if rows:
                append_results_to_csv(rows, RESULTS_FILE)
                print(f"    Appended {len(rows)} rows to {RESULTS_FILE}")

                # Incremental plot
                plot_combo(design_name, taxonomy, cfg["description"])

                # Update checkpoint
                checkpoint[key] = {
                    "n_rows": len(rows),
                    "design": design_name,
                    "taxonomy": taxonomy,
                }
                save_checkpoint(checkpoint)
            else:
                print(f"    No rows generated for {key}")

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------
    generate_summary()
    print(f"\nDone. All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()