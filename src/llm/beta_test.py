"""
============================================================================
STRUCTURAL-β CHARACTERIZATION + STAGE B VIABILITY DIAGNOSTIC
============================================================================

Purpose:
--------
Tests how much AI-centric and human-centric stimuli differ in structural
(non-emotional) properties. This addresses the structural-β alternative
hypothesis from Stage 1 Layer 2:

  α-substrate (preferred): AI-centric stimuli engage a more stable
      functional substrate. Probes generalize because the underlying
      representation is coherent across stimulus distributions.

  structural-β (THIS TEST): AI-centric stimuli are syntactically /
      discursively richer than simple human-centric stimuli, and probe
      transferability is driven by stimulus complexity rather than
      affective-substrate stability.

  γ-intensity: tested in gamma_intensity_test.py.

  training-overlap: field-wide caveat, not testable on existing data.

What this script does:
----------------------
STAGE A — DESCRIPTIVE CHARACTERIZATION
  For each prompt, compute 5 structural metrics:
    1. Length (token count, char count, sentence count)
    2. Lexical diversity (Type-Token Ratio, MTLD)
    3. Syntactic complexity (Flesch-Kincaid grade level proxy)
    4. Discourse-marker frequency (count of common connectives)
    5. Mean sentence length (tokens / sentences)

  Compare AI-centric vs human-centric distributions:
    - Mann-Whitney U test (non-parametric)
    - Cliff's delta (effect size)
    - Median ratio
  Output: summary CSV + boxplot panels.

STAGE B VIABILITY DIAGNOSTIC
  For multiple candidate matched-complexity designs, report cell sizes:
    - 2x2 design (origin × length-quartile, Q1 vs Q4)
    - 2x2 design (origin × length-tercile, T1 vs T3)
    - 2x2 design (origin × length-median split)
    - Same three but using lexical diversity instead of length
    - Same three but using syntactic complexity instead of length
    - 2x2x2 design (origin × length × diversity, both Q1/Q4)

  Output: diagnostic CSV with cell sizes for each design, sorted by
  smallest-cell size. Decision rule guidance:
    - Smallest cell ≥ 100 prompts: Stage B is well-powered, proceed.
    - Smallest cell 50-100 prompts: Stage B is marginal, run with
      caution — bootstrap CIs will be wide.
    - Smallest cell < 50 prompts: Stage B is underpowered for that
      design. Try a coarser split (e.g., median split) or admit in
      paper that data does not support a matched test.

Reading the result:
-------------------
- Stage A tells you "how confounded" your two sets are. Expect
  AI-centric to score higher on every metric. Cliff's δ > 0.474 (large)
  is informative; this is not a "fail" — it documents the magnitude of
  the confound the reader should know about.
- Stage B diagnostic tells you whether you can RUN a controlled test
  on this data, or whether the confound is total (e.g., zero overlap
  in length distributions = no matched test possible).

How this enters the paper:
--------------------------
- Stage A always enters: as a table + boxplot panel in methods or
  appendix, with the "structural-β confound is real, magnitude is X"
  paragraph in limitations.
- Stage B enters CONDITIONALLY: only if smallest cell sizes permit
  stable inference. If they don't, you cite this script's diagnostic
  as evidence that "matched-complexity controls are not feasible on
  the present data; we scope this as future work requiring purpose-
  generated stimuli (Section X.X)."

Inputs expected:
----------------
- Two prompt-text sources. Adjust paths below if your local clone
  has them in different locations. If they're inside the .pkl
  activation files, this script can extract them; if they're in
  separate CSV/JSON files (lighter to ship in a clone), use those.
============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from collections import Counter


# ============================================================================
# CONFIGURATION — adjust paths to your local clone
# ============================================================================
LLM_USED = "Llama-2-7b-chat-hf"

# OPTION 1: Extract prompts from the activation pickles (heavy files).
# Use this if you only have the .pkl files locally.
PATH_AI_CENTRIC_PKL = (
    "data/03_activations/"
    "generated_prompts_Llama-2-7b-chat-hf_20251014_203636_"
    "FINAL_WITH_RATINGS_AND_CATS.pkl"
)
PATH_HUMAN_CENTRIC_PKL = (
    "data/03_activations/"
    "MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
)

# OPTION 2: If you have lighter prompt-text-only files in your clone,
# point to them here and set USE_TEXT_ONLY = True. Faster to load.
# Expected: a CSV/JSON with at least a "prompt" column.
PATH_AI_CENTRIC_TXT = "data/01_stimuli/generated_prompts/generated_emotional_prompts_batched.csv"
PATH_HUMAN_CENTRIC_TXT = "data/01_stimuli/andy_zou_emotion_query.csv"  # adjust if different
USE_TEXT_ONLY = False  # flip to True if text-only files exist

PROMPT_COLUMN = "prompt"  # column name in the CSV/pickle holding prompt text

OUTPUT_DIR = os.path.join(
    "figures", f"structural_beta_test_{LLM_USED}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STRUCTURAL METRICS
# ============================================================================
DISCOURSE_MARKERS = {
    # Causal / consequence
    "because", "therefore", "thus", "hence", "consequently", "so",
    # Contrast
    "however", "but", "although", "though", "yet", "nevertheless",
    "nonetheless", "whereas", "while",
    # Addition
    "moreover", "furthermore", "additionally", "also", "besides",
    "in addition",
    # Sequence / time
    "first", "second", "third", "next", "then", "finally",
    "subsequently", "afterwards", "meanwhile",
    # Conditional / hypothetical
    "if", "unless", "provided", "assuming", "suppose",
    # Exemplification
    "for example", "for instance", "such as", "specifically",
    "in particular",
    # Conclusion / summary
    "in conclusion", "to summarize", "in short", "overall",
}


def tokenize_simple(text):
    """Lowercase + word tokenize via regex. No NLTK dependency."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[\w']+\b", text.lower())


def split_sentences_simple(text):
    """Naive sentence split on .!? followed by whitespace.
    Good enough for descriptive metrics; a parser would be overkill here."""
    if not isinstance(text, str):
        return []
    # Avoid splitting on common abbreviations
    text = re.sub(r"\b(Mr|Mrs|Dr|Ms|St|Jr|Sr|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.replace("<DOT>", ".").strip() for s in sents if s.strip()]
    return sents


def type_token_ratio(tokens):
    """TTR. Inflated for short texts, but fine for relative comparisons
    if the AI/human length distributions overlap. Otherwise interpret
    with caution."""
    if len(tokens) == 0:
        return np.nan
    return len(set(tokens)) / len(tokens)


def mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity. Length-robust; a better
    metric than TTR when comparing texts of different sizes.
    Returns the average run length needed for TTR to drop to threshold."""
    if len(tokens) < 50:
        # MTLD is unstable below ~50 tokens. Fall back to TTR for short.
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
        # Partial factor at the end
        if token_count > 0:
            partial = (1.0 - running_ttr) / (1.0 - threshold)
            factors += partial
        return len(toks) / factors if factors > 0 else len(toks)

    forward = _factor_count(tokens)
    backward = _factor_count(tokens[::-1])
    return (forward + backward) / 2.0


def count_syllables_word(word):
    """Rough syllable count. Used by Flesch-Kincaid."""
    word = word.lower()
    if len(word) == 0:
        return 0
    # Count vowel groups
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    # Silent 'e'
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def flesch_kincaid_grade(text):
    """FK grade level. Higher = more syntactically/lexically complex.
    Proxy for syntactic depth without a parser."""
    sents = split_sentences_simple(text)
    tokens = tokenize_simple(text)
    if len(sents) == 0 or len(tokens) == 0:
        return np.nan
    syllables = sum(count_syllables_word(t) for t in tokens)
    words_per_sentence = len(tokens) / len(sents)
    syllables_per_word = syllables / len(tokens)
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


def discourse_marker_count(text):
    """Frequency of discourse markers (normalized per 100 tokens)."""
    if not isinstance(text, str):
        return np.nan
    text_lower = text.lower()
    tokens = tokenize_simple(text)
    if len(tokens) == 0:
        return np.nan
    count = 0
    # Single-word markers
    token_set = Counter(tokens)
    for marker in DISCOURSE_MARKERS:
        if " " not in marker:
            count += token_set.get(marker, 0)
        else:
            # Multi-word: substring search
            count += text_lower.count(marker)
    return (count / len(tokens)) * 100.0  # per 100 tokens


def compute_metrics_for_prompt(text):
    """Return dict of structural metrics for a single prompt text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    tokens = tokenize_simple(text)
    sents = split_sentences_simple(text)
    return {
        "n_tokens": len(tokens),
        "n_chars": len(text),
        "n_sentences": len(sents),
        "mean_sent_len": (len(tokens) / len(sents)) if len(sents) else np.nan,
        "ttr": type_token_ratio(tokens),
        "mtld": mtld(tokens),
        "fk_grade": flesch_kincaid_grade(text),
        "discourse_markers_per100": discourse_marker_count(text),
    }


# ============================================================================
# DATA LOADING
# ============================================================================
def load_prompts_from_pkl(path, prompt_column):
    """Loads a pickle and extracts prompt-text column."""
    df = pd.read_pickle(path)
    if prompt_column not in df.columns:
        raise KeyError(
            f"Column '{prompt_column}' not in {path}. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
    return df[prompt_column].astype(str).tolist()


def load_prompts_from_text(path, prompt_column):
    """Loads a CSV/JSON of prompt text."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported file format: {path}")
    if prompt_column not in df.columns:
        raise KeyError(
            f"Column '{prompt_column}' not in {path}. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
    return df[prompt_column].astype(str).tolist()


def load_prompts(path_pkl, path_txt, use_text_only, prompt_col):
    if use_text_only:
        return load_prompts_from_text(path_txt, prompt_col)
    return load_prompts_from_pkl(path_pkl, prompt_col)


# ============================================================================
# STATS HELPERS
# ============================================================================
def cliffs_delta(x, y):
    """Cliff's delta via Mann-Whitney U statistic."""
    x = np.asarray([v for v in x if not np.isnan(v)])
    y = np.asarray([v for v in y if not np.isnan(v)])
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return (2.0 * u_stat) / (n_x * n_y) - 1.0


def compare_metric(ai_vals, human_vals):
    ai = np.asarray([v for v in ai_vals if not np.isnan(v)])
    hu = np.asarray([v for v in human_vals if not np.isnan(v)])
    if len(ai) < 3 or len(hu) < 3:
        return None
    u, p = mannwhitneyu(ai, hu, alternative="two-sided")
    delta = cliffs_delta(ai, hu)
    return {
        "n_ai": len(ai),
        "n_human": len(hu),
        "ai_median": float(np.median(ai)),
        "human_median": float(np.median(hu)),
        "ai_mean": float(np.mean(ai)),
        "human_mean": float(np.mean(hu)),
        "u_stat": float(u),
        "p_value": float(p),
        "cliffs_delta": float(delta),
        "median_ratio_ai_over_human": (
            float(np.median(ai) / np.median(hu)) if np.median(hu) != 0 else np.nan
        ),
    }


# ============================================================================
# STAGE A
# ============================================================================
def run_stage_a():
    print("\n" + "=" * 70)
    print("STAGE A — STRUCTURAL CHARACTERIZATION")
    print("=" * 70)

    print("\nLoading prompts...")
    ai_prompts = load_prompts(
        PATH_AI_CENTRIC_PKL, PATH_AI_CENTRIC_TXT,
        USE_TEXT_ONLY, PROMPT_COLUMN,
    )
    human_prompts = load_prompts(
        PATH_HUMAN_CENTRIC_PKL, PATH_HUMAN_CENTRIC_TXT,
        USE_TEXT_ONLY, PROMPT_COLUMN,
    )
    print(f"  AI-centric:    {len(ai_prompts)} prompts")
    print(f"  Human-centric: {len(human_prompts)} prompts")

    print("\nComputing structural metrics...")
    ai_metrics = pd.DataFrame([
        m for m in (compute_metrics_for_prompt(p) for p in ai_prompts)
        if m is not None
    ])
    ai_metrics["origin"] = "AI-centric"

    human_metrics = pd.DataFrame([
        m for m in (compute_metrics_for_prompt(p) for p in human_prompts)
        if m is not None
    ])
    human_metrics["origin"] = "Human-centric"

    all_metrics = pd.concat([ai_metrics, human_metrics], ignore_index=True)
    metrics_path = os.path.join(OUTPUT_DIR, "all_metrics.csv")
    all_metrics.to_csv(metrics_path, index=False)
    print(f"  Saved per-prompt metrics: {metrics_path}")

    # Comparison table
    metric_cols = [
        "n_tokens", "n_chars", "n_sentences", "mean_sent_len",
        "ttr", "mtld", "fk_grade", "discourse_markers_per100",
    ]
    summary_rows = []
    for col in metric_cols:
        result = compare_metric(
            ai_metrics[col].values, human_metrics[col].values,
        )
        if result is not None:
            result["metric"] = col
            summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        ["metric", "n_ai", "n_human",
         "ai_median", "human_median", "median_ratio_ai_over_human",
         "ai_mean", "human_mean",
         "cliffs_delta", "u_stat", "p_value"]
    ]
    summary_path = os.path.join(OUTPUT_DIR, "stage_a_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\nSaved summary: {summary_path}")

    # Boxplot panel
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(metric_cols):
        ax = axes[i]
        sns.boxplot(
            data=all_metrics, x="origin", y=col, ax=ax,
            palette={"AI-centric": "#1f77b4", "Human-centric": "#ff7f0e"},
        )
        delta_row = summary_df[summary_df["metric"] == col]
        if not delta_row.empty:
            d = delta_row["cliffs_delta"].values[0]
            p = delta_row["p_value"].values[0]
            ax.set_title(f"{col}\nδ={d:+.2f}, p={p:.2e}")
        else:
            ax.set_title(col)
        ax.set_xlabel("")
    plt.suptitle(
        "Structural-β characterization: AI-centric vs Human-centric stimuli",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "stage_a_boxplots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplots: {plot_path}")

    return all_metrics, summary_df


# ============================================================================
# STAGE B VIABILITY DIAGNOSTIC
# ============================================================================
def diagnose_design_viability(all_metrics):
    """For each candidate matched-complexity design, report cell sizes.
    Output a sorted table with smallest-cell highlighted."""
    print("\n" + "=" * 70)
    print("STAGE B VIABILITY DIAGNOSTIC")
    print("=" * 70)

    designs = []

    # Single-axis splits
    for split_metric in ["n_tokens", "mtld", "fk_grade"]:
        # Compute thresholds across the FULL pooled distribution
        all_vals = all_metrics[split_metric].dropna()

        for split_name, low_q, high_q in [
            ("Q1 vs Q4", 0.25, 0.75),
            ("T1 vs T3", 1/3, 2/3),
            ("median split", 0.50, 0.50),
        ]:
            q_low = all_vals.quantile(low_q)
            q_high = all_vals.quantile(high_q)

            for origin in ["AI-centric", "Human-centric"]:
                origin_df = all_metrics[all_metrics["origin"] == origin]
                if low_q == high_q:
                    # Median split: <= median = low, > median = high
                    n_low = (origin_df[split_metric] <= q_low).sum()
                    n_high = (origin_df[split_metric] > q_high).sum()
                else:
                    n_low = (origin_df[split_metric] <= q_low).sum()
                    n_high = (origin_df[split_metric] >= q_high).sum()

                designs.append({
                    "design": f"2x2 (origin × {split_metric}, {split_name})",
                    "split_metric": split_metric,
                    "split_type": split_name,
                    "origin": origin,
                    "n_low_complexity": int(n_low),
                    "n_high_complexity": int(n_high),
                    "smallest_cell_in_pair": int(min(n_low, n_high)),
                })

    designs_df = pd.DataFrame(designs)

    # Roll up to per-design min cell size (across origins)
    rollup = designs_df.groupby(
        ["design", "split_metric", "split_type"], as_index=False,
    )["smallest_cell_in_pair"].min().rename(
        columns={"smallest_cell_in_pair": "smallest_cell_overall"},
    )

    # 2x2x2 design: origin × length-quartile × diversity-quartile
    n_token_q1 = all_metrics["n_tokens"].quantile(0.25)
    n_token_q3 = all_metrics["n_tokens"].quantile(0.75)
    mtld_q1 = all_metrics["mtld"].quantile(0.25)
    mtld_q3 = all_metrics["mtld"].quantile(0.75)

    cells_222 = []
    for origin in ["AI-centric", "Human-centric"]:
        for tok_label, tok_filter in [
            ("low_tokens", lambda d: d["n_tokens"] <= n_token_q1),
            ("high_tokens", lambda d: d["n_tokens"] >= n_token_q3),
        ]:
            for div_label, div_filter in [
                ("low_mtld", lambda d: d["mtld"] <= mtld_q1),
                ("high_mtld", lambda d: d["mtld"] >= mtld_q3),
            ]:
                origin_df = all_metrics[all_metrics["origin"] == origin]
                cell = origin_df[tok_filter(origin_df) & div_filter(origin_df)]
                cells_222.append({
                    "design": "2x2x2 (origin × tokens-Q1/Q4 × mtld-Q1/Q4)",
                    "split_metric": "tokens + mtld",
                    "split_type": "Q1 vs Q4 (both axes)",
                    "origin": origin,
                    "cell_label": f"{tok_label} & {div_label}",
                    "n_in_cell": int(len(cell)),
                })
    cells_222_df = pd.DataFrame(cells_222)
    smallest_222 = cells_222_df["n_in_cell"].min()
    rollup_222 = pd.DataFrame([{
        "design": "2x2x2 (origin × tokens-Q1/Q4 × mtld-Q1/Q4)",
        "split_metric": "tokens + mtld",
        "split_type": "Q1 vs Q4 (both axes)",
        "smallest_cell_overall": int(smallest_222),
    }])
    rollup_full = pd.concat([rollup, rollup_222], ignore_index=True)
    rollup_full = rollup_full.sort_values(
        "smallest_cell_overall", ascending=False,
    )

    # Decision rule annotations
    def decision(n):
        if n >= 100:
            return "well-powered"
        if n >= 50:
            return "marginal — wide CIs expected"
        if n >= 20:
            return "underpowered — interpret cautiously"
        return "infeasible"

    rollup_full["viability"] = rollup_full[
        "smallest_cell_overall"
    ].apply(decision)

    rollup_path = os.path.join(OUTPUT_DIR, "stage_b_viability.csv")
    rollup_full.to_csv(rollup_path, index=False)

    # Verbose per-cell file
    full_cells_path = os.path.join(OUTPUT_DIR, "stage_b_per_cell.csv")
    pd.concat([designs_df, cells_222_df], ignore_index=True).to_csv(
        full_cells_path, index=False,
    )

    print(f"\n{rollup_full.to_string(index=False)}")
    print(f"\nSaved diagnostic rollup: {rollup_path}")
    print(f"Saved per-cell counts: {full_cells_path}")

    # Recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    well_powered = rollup_full[
        rollup_full["viability"] == "well-powered"
    ]
    marginal = rollup_full[
        rollup_full["viability"] == "marginal — wide CIs expected"
    ]
    if not well_powered.empty:
        best = well_powered.iloc[0]
        print(
            f"Best design: {best['design']}\n"
            f"  Smallest cell: {best['smallest_cell_overall']} prompts\n"
            f"  Proceed with Stage B using this design."
        )
    elif not marginal.empty:
        best = marginal.iloc[0]
        print(
            f"No well-powered design found. Best marginal option:\n"
            f"  {best['design']}\n"
            f"  Smallest cell: {best['smallest_cell_overall']} prompts\n"
            f"  Stage B can run but expect wide bootstrap CIs.\n"
            f"  Report results with caveats; not a clean test."
        )
    else:
        print(
            "No viable design (>=50 prompts smallest cell). The two "
            "stimulus distributions do not overlap enough in any axis to "
            "support a matched-complexity test on present data.\n"
            "Recommendation: report Stage A characterization in the paper "
            "as evidence of structural confound magnitude; scope rigorous "
            "structural-β test as future work requiring purpose-generated "
            "matched-complexity stimuli."
        )

    return rollup_full


# ============================================================================
# MAIN
# ============================================================================
def main():
    all_metrics, summary_df = run_stage_a()
    rollup = diagnose_design_viability(all_metrics)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Outputs in: {OUTPUT_DIR}/")
    print("Files:")
    print("  - all_metrics.csv          (per-prompt metrics)")
    print("  - stage_a_summary.csv      (descriptive comparison)")
    print("  - stage_a_boxplots.png     (visual panel)")
    print("  - stage_b_viability.csv    (design viability rollup)")
    print("  - stage_b_per_cell.csv     (per-cell counts, all designs)")
    print()
    print("Next step:")
    print("  - Inspect stage_b_viability.csv. The 'viability' column tells")
    print("    you whether Stage B (matched-complexity probe transfer test)")
    print("    is feasible. If 'well-powered' or 'marginal', request the")
    print("    Stage B script. If all rows are 'infeasible' or 'underpowered'")
    print("    Stage A characterization is the empirical contribution; cite")
    print("    Sclar 2024 / Mizrahi 2024 in limitations and scope rigorous")
    print("    structural-β test as future work.")


if __name__ == "__main__":
    main()