"""
============================================================================
γ-INTENSITY TEST
============================================================================

Purpose:
--------
Tests the γ-intensity hypothesis as an alternative explanation for the
asymmetric cross-domain generalization observed between AI-centric and
human-centric (simple) stimuli.

The hypothesis space (negotiated during Stage 1):

  α-substrate (preferred): AI-centric stimuli engage a more stable functional
      substrate. Probes generalize because the underlying representation is
      coherent across stimulus distributions.

  structural-β: AI-centric stimuli are syntactically/discursively more
      diverse. Probes generalize because they catch a broader feature set.
      NOT TESTABLE without matched-complexity controls. Scoped as future
      direction.

  γ-intensity (THIS TEST): AI-centric stimuli engage emotional encoding at
      higher amplitude (larger activation norms in the relevant subspace).
      Probes generalize because AI-centric → simple is a high-amplitude →
      low-amplitude case, while simple → AI-centric is the harder direction.

  training-overlap: target/judge LLM trained on similar emotional text.
      Field-wide caveat. Out of scope.

What this test does:
--------------------
1. Loads activations from BOTH stimulus sources (AI-centric and human-
   centric) at the *plateau layer range* identified by the layer × pooling
   sweep (default: L10-20 for Llama-2-7B; adjust based on your sweep).
2. Computes the L2 norm of the last_token_activation for each prompt.
3. Matches by emotion category (primary label).
4. Compares norm distributions per category and aggregated:
      - Mann-Whitney U test (non-parametric, no normality assumption)
      - Cliff's delta (effect size, robust)
      - Median ratio (interpretable)
5. Outputs CSV summaries and per-layer + aggregated plots.

Reading the result:
-------------------
- If AI-centric L2 norms are SYSTEMATICALLY LARGER (Cliff's δ > 0.3, p <
  0.05 across most categories and across the layer range) → γ is consistent
  with the data. Report in Appendix; discussion notes it as a partial
  contributor to the asymmetric generalization, NOT a replacement for α.
- If norms are COMPARABLE (Cliff's δ ≈ 0, p > 0.05) → γ is NOT supported.
  α reading is reinforced; structural-β remains the only live alternative.
- If pattern is INCONSISTENT across categories or layers → mixed evidence.
  Report honestly; γ may operate for some emotion categories but not others.

Interpretation caveat:
----------------------
Even if γ is supported, it does NOT eliminate α — the two are not mutually
exclusive. Higher amplitude could itself be a SIGNATURE of more stable
substrate engagement. The cleanest discussion frames γ as an observable
correlate, not a competing mechanism.

Adjustment to make if structure of pickles changes:
---------------------------------------------------
- ACTIVATION_COL: change pooling method if you want to test e.g. mean
  instead of last_token. The column name must exist in the nested
  activations DataFrame.
- PLATEAU_LAYERS: change based on your layer × pooling sweep results.
  For Llama-2-7B with last_token, L10-20 is a reasonable plateau window.
  For Qwen2.5-14B (40 layers), scale proportionally e.g. L13-25.
- TAXONOMY_FOR_MATCHING: defaults to ekman_basic_emotions as the primary
  match key (smaller class space → more samples per cell). Switch to
  go_emotions if you want fine-grained matching at the cost of cell sizes.
============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_USED = "Llama-2-7b-chat-hf"

# Stimulus sources. Same paths as your cross-dataset validation script.
PATH_AI_CENTRIC = (
    "data/03_activations/"
    "generated_prompts_Llama-2-7b-chat-hf_20251014_203636_"
    "FINAL_WITH_RATINGS_AND_CATS.pkl"
)
PATH_HUMAN_CENTRIC = (
    "data/03_activations/"
    "MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
)

# Output
FIGURES_DIR = os.path.join("figures", f"gamma_intensity_test_{LLM_USED}")
RESULTS_CSV = os.path.join(FIGURES_DIR, "gamma_intensity_results.csv")
SUMMARY_CSV = os.path.join(FIGURES_DIR, "gamma_intensity_summary.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Activation extraction config
ACTIVATION_COL = "last_token_activation"
# Plateau window — adjust based on your layer × pooling sweep.
# For Llama-2-7B (32 layers + embedding = layers 0-32), the probe-perf
# plateau on Ekman appears around L10-20 in your figure. Use the full
# plateau, not a single layer, to avoid cherry-picking.
PLATEAU_LAYERS = list(range(33))  # L0 through L32 inclusive

# Match by primary emotion label of which taxonomy?
# Ekman = 7 categories (smaller, more samples per cell, cleaner test)
# GoEmotions = 28 categories (finer, but cells become tiny → unstable)
TAXONOMY_FOR_MATCHING = "ekman_basic_emotions"

# Statistical thresholds (for reporting, not gating)
# Cliff's delta convention: 0.147 = small, 0.33 = medium, 0.474 = large
# We treat δ > 0.3 as "γ consistent" by convention; tune if you want
# stricter or looser reporting.
EFFECT_SIZE_THRESHOLD = 0.3
ALPHA = 0.05

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# DATA EXTRACTION
# ============================================================================
def extract_norms_for_layer(df, layer, taxonomy):
    """
    For each row in df, extract the L2 norm of the last_token_activation
    at the given layer, paired with the primary emotion label of the given
    taxonomy.

    Returns: DataFrame with columns [primary_label, norm].
    Rows where activation extraction fails or no label exists are skipped.
    """
    records = []
    for row in df.itertuples():
        try:
            nested_df = row.activations
            if layer not in nested_df.index:
                continue
            act_vector = nested_df.loc[layer, ACTIVATION_COL]
            if not isinstance(act_vector, np.ndarray):
                continue

            labels_list = getattr(row, taxonomy)
            if not isinstance(labels_list, list) or len(labels_list) == 0:
                continue
            primary_label = str(labels_list[0]).lower().strip()

            # L2 norm of the activation vector
            norm = float(np.linalg.norm(act_vector, ord=2))
            records.append({
                "primary_label": primary_label,
                "norm": norm,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


# ============================================================================
# STATISTICAL TESTS
# ============================================================================
def cliffs_delta(x, y):
    """
    Cliff's delta — non-parametric effect size for two independent samples.
    Range: [-1, 1]. Positive δ means x tends to be larger than y.

    Conventions (Romano et al., 2006):
      |δ| < 0.147 = negligible
      |δ| < 0.33  = small
      |δ| < 0.474 = medium
      |δ| ≥ 0.474 = large

    Implementation note: O(n*m) naive comparison. For n,m up to ~10k this
    is fine; if your cells get huge, consider the U-statistic-based formula:
        δ = 2 * U / (n*m) - 1
    where U is the Mann-Whitney U statistic.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan

    # Use U-statistic relation for efficiency
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    delta = (2.0 * u_stat) / (n_x * n_y) - 1.0
    return float(delta)


def median_ratio(x, y):
    """Median(x) / Median(y). Interpretable scale comparison."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    med_y = np.median(y)
    if med_y == 0:
        return np.nan
    return float(np.median(x) / med_y)


def compare_distributions(ai_norms, human_norms):
    """
    Compare AI-centric vs human-centric norms.
    Returns dict with U-statistic, p-value, Cliff's delta, median ratio,
    and sample sizes.
    """
    if len(ai_norms) < 3 or len(human_norms) < 3:
        return {
            "u_stat": np.nan, "p_value": np.nan,
            "cliffs_delta": np.nan, "median_ratio": np.nan,
            "n_ai": len(ai_norms), "n_human": len(human_norms),
            "ai_median": np.nan, "human_median": np.nan,
        }

    u_stat, p_val = mannwhitneyu(
        ai_norms, human_norms, alternative="two-sided"
    )
    delta = cliffs_delta(ai_norms, human_norms)
    ratio = median_ratio(ai_norms, human_norms)
    return {
        "u_stat": float(u_stat),
        "p_value": float(p_val),
        "cliffs_delta": delta,
        "median_ratio": ratio,
        "n_ai": int(len(ai_norms)),
        "n_human": int(len(human_norms)),
        "ai_median": float(np.median(ai_norms)),
        "human_median": float(np.median(human_norms)),
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def main():
    print("LOADING DATASETS...")
    df_ai = pd.read_pickle(PATH_AI_CENTRIC)
    df_human = pd.read_pickle(PATH_HUMAN_CENTRIC)
    print(f"  AI-centric:    {len(df_ai)} prompts")
    print(f"  Human-centric: {len(df_human)} prompts")
    print(f"  Plateau layers: L{PLATEAU_LAYERS[0]}-L{PLATEAU_LAYERS[-1]}")
    print(f"  Matching taxonomy: {TAXONOMY_FOR_MATCHING}")

    all_results = []

    print(
        f"\nCOMPUTING L2 NORMS ACROSS PLATEAU LAYERS "
        f"[{PLATEAU_LAYERS[0]}-{PLATEAU_LAYERS[-1]}]"
    )
    for layer in tqdm(PLATEAU_LAYERS, desc="Layers"):
        ai_norms_df = extract_norms_for_layer(
            df_ai, layer, TAXONOMY_FOR_MATCHING
        )
        human_norms_df = extract_norms_for_layer(
            df_human, layer, TAXONOMY_FOR_MATCHING
        )

        if ai_norms_df.empty or human_norms_df.empty:
            continue

        # Per-category comparison: only categories present in BOTH sources
        common_cats = sorted(
            set(ai_norms_df["primary_label"].unique())
            & set(human_norms_df["primary_label"].unique())
        )

        for cat in common_cats:
            ai_cat = ai_norms_df.loc[
                ai_norms_df["primary_label"] == cat, "norm"
            ].values
            human_cat = human_norms_df.loc[
                human_norms_df["primary_label"] == cat, "norm"
            ].values
            stats = compare_distributions(ai_cat, human_cat)
            stats.update({
                "layer": layer,
                "category": cat,
                "scope": "per_category",
            })
            all_results.append(stats)

        # Aggregated comparison: all categories pooled
        # (gives global signal; per-category gives the granular pattern)
        agg_stats = compare_distributions(
            ai_norms_df["norm"].values,
            human_norms_df["norm"].values,
        )
        agg_stats.update({
            "layer": layer,
            "category": "_ALL_AGGREGATED_",
            "scope": "aggregated",
        })
        all_results.append(agg_stats)

    if not all_results:
        print("ERROR: No results computed. Check data paths and layer range.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-layer/per-category results: {RESULTS_CSV}")

    # ========================================================================
    # SUMMARY: aggregate Cliff's delta across the plateau window per category
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS PLATEAU WINDOW")
    print("=" * 70)

    summary_rows = []
    per_cat = results_df[results_df["scope"] == "per_category"]
    for cat in sorted(per_cat["category"].unique()):
        cat_df = per_cat[per_cat["category"] == cat]
        # Drop NaNs (cases where cell sizes were too small)
        deltas = cat_df["cliffs_delta"].dropna().values
        ratios = cat_df["median_ratio"].dropna().values
        pvals = cat_df["p_value"].dropna().values
        n_layers_with_data = len(deltas)
        if n_layers_with_data == 0:
            continue
        median_delta = float(np.median(deltas))
        median_ratio_val = float(np.median(ratios))
        # How many layers show a "significant" effect at the threshold
        n_sig_layers = int(np.sum(
            (np.abs(deltas) >= EFFECT_SIZE_THRESHOLD)
            & (pvals < ALPHA)
        ))
        consistency = n_sig_layers / n_layers_with_data

        summary_rows.append({
            "category": cat,
            "n_layers_with_data": n_layers_with_data,
            "median_cliffs_delta": median_delta,
            "median_norm_ratio": median_ratio_val,
            "n_layers_significant": n_sig_layers,
            "consistency_fraction": consistency,
        })

    # Aggregated row
    agg_df = results_df[results_df["scope"] == "aggregated"]
    if not agg_df.empty:
        deltas_agg = agg_df["cliffs_delta"].dropna().values
        ratios_agg = agg_df["median_ratio"].dropna().values
        pvals_agg = agg_df["p_value"].dropna().values
        if len(deltas_agg) > 0:
            n_sig_agg = int(np.sum(
                (np.abs(deltas_agg) >= EFFECT_SIZE_THRESHOLD)
                & (pvals_agg < ALPHA)
            ))
            summary_rows.append({
                "category": "_ALL_AGGREGATED_",
                "n_layers_with_data": len(deltas_agg),
                "median_cliffs_delta": float(np.median(deltas_agg)),
                "median_norm_ratio": float(np.median(ratios_agg)),
                "n_layers_significant": n_sig_agg,
                "consistency_fraction": (
                    n_sig_agg / len(deltas_agg)
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        "median_cliffs_delta", ascending=False
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {SUMMARY_CSV}")

    # ========================================================================
    # VERDICT (printed to stdout for at-a-glance reading)
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERDICT GUIDANCE")
    print("=" * 70)

    agg_row = summary_df[summary_df["category"] == "_ALL_AGGREGATED_"]
    if not agg_row.empty:
        agg_delta = agg_row["median_cliffs_delta"].values[0]
        agg_consistency = agg_row["consistency_fraction"].values[0]
        agg_ratio = agg_row["median_norm_ratio"].values[0]

        print(
            f"\nAggregated (all categories pooled across plateau window):"
        )
        print(f"  Median Cliff's δ        = {agg_delta:+.3f}")
        print(f"  Median norm ratio AI/H  = {agg_ratio:.3f}")
        print(
            f"  Consistency (sig layers): "
            f"{agg_consistency:.0%}"
        )

        if agg_delta > EFFECT_SIZE_THRESHOLD and agg_consistency > 0.5:
            verdict = (
                "γ-INTENSITY IS CONSISTENT WITH THE DATA. AI-centric "
                "stimuli yield systematically higher activation norms.\n"
                "Report in Appendix as supporting evidence; frame as "
                "an observable correlate of the asymmetric generalization, "
                "NOT a replacement for α-substrate."
            )
        elif abs(agg_delta) < 0.147 and agg_consistency < 0.3:
            verdict = (
                "γ-INTENSITY IS NOT SUPPORTED. Norms are comparable. "
                "α-substrate reading is reinforced; structural-β remains "
                "the only live alternative."
            )
        else:
            verdict = (
                "MIXED EVIDENCE. γ may operate for some categories or "
                "layers but not consistently. Report honestly; do not "
                "claim γ as a clean explanation. Consider per-category "
                "discussion."
            )
        print(f"\n{verdict}\n")

    # ========================================================================
    # PLOTS
    # ========================================================================
    sns.set_style("whitegrid")

    # Plot 1: Aggregated Cliff's delta across layers
    if not agg_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            agg_df["layer"], agg_df["cliffs_delta"],
            marker="o", linewidth=2, color="#cc4c2a",
            label="Cliff's δ (AI vs Human)",
        )
        ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax.axhline(
            EFFECT_SIZE_THRESHOLD, color="green", linestyle="--",
            alpha=0.5, label=f"δ = {EFFECT_SIZE_THRESHOLD} (medium effect)",
        )
        ax.axhline(
            -EFFECT_SIZE_THRESHOLD, color="green", linestyle="--",
            alpha=0.5,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cliff's δ (positive = AI-centric norms larger)")
        ax.set_title(
            f"γ-Intensity Test (aggregated): "
            f"L2 norm ratio AI-centric vs Human-centric\n"
            f"{LLM_USED} | {TAXONOMY_FOR_MATCHING}"
        )
        ax.set_ylim(-1, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, "cliffs_delta_aggregated.png"),
            dpi=300,
        )
        plt.close()

    # Plot 2: Heatmap — category × layer Cliff's delta
    pivot = per_cat.pivot_table(
        index="category", columns="layer", values="cliffs_delta",
        aggfunc="mean",
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(pivot))))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={"label": "Cliff's δ"},
        )
        ax.set_title(
            f"γ-Intensity per Category × Layer\n"
            f"{LLM_USED} | {TAXONOMY_FOR_MATCHING} | "
            f"Positive = AI-centric norms larger"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, "cliffs_delta_heatmap.png"),
            dpi=300,
        )
        plt.close()

    # Plot 3: Norm distributions per category (one example layer = middle of
    # plateau). Useful for sanity-checking that the medians are different.
    sample_layer = PLATEAU_LAYERS[len(PLATEAU_LAYERS) // 2]
    ai_sample = extract_norms_for_layer(
        df_ai, sample_layer, TAXONOMY_FOR_MATCHING
    )
    human_sample = extract_norms_for_layer(
        df_human, sample_layer, TAXONOMY_FOR_MATCHING
    )
    if not ai_sample.empty and not human_sample.empty:
        ai_sample["source"] = "AI-centric"
        human_sample["source"] = "Human-centric"
        combined = pd.concat([ai_sample, human_sample], ignore_index=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(
            data=combined, x="primary_label", y="norm", hue="source",
            ax=ax,
        )
        ax.set_title(
            f"L2 Norm distribution per emotion category | "
            f"Layer {sample_layer} (plateau midpoint)\n"
            f"{LLM_USED} | {TAXONOMY_FOR_MATCHING}"
        )
        ax.set_xlabel("Primary emotion label")
        ax.set_ylabel("L2 norm of last_token_activation")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                f"norm_distribution_layer_{sample_layer}.png",
            ),
            dpi=300,
        )
        plt.close()

    print(f"\nAll outputs in: {FIGURES_DIR}/")
    print("Files:")
    print("  - gamma_intensity_results.csv  (per layer × category)")
    print("  - gamma_intensity_summary.csv  (aggregated across plateau)")
    print("  - cliffs_delta_aggregated.png")
    print("  - cliffs_delta_heatmap.png")
    print(f"  - norm_distribution_layer_{sample_layer}.png")


if __name__ == "__main__":
    main()