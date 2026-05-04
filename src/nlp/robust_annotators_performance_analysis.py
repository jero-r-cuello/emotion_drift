"""
Inter-rater agreement analysis for multi-label emotion annotation.

Replaces the previous "broad kappa" construction with standard, defensible
multi-label IRR metrics:

- Krippendorff's alpha (nominal): conservative, single-label match.
- Krippendorff's alpha (MASI distance): multi-label-aware. Headline metric.
- Jaccard set-overlap: secondary, intuitive metric (no chance correction).

Computes two analyses:
1. Annotator-vs-human agreement: each model annotator vs the manual subset.
2. Inter-model agreement matrix: pairwise across all model annotators.

Outputs CSV summaries and bootstrap CIs at 95%.

Required: pip install krippendorff
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from itertools import combinations
from tqdm import tqdm


# Path to the multi-model annotations (must contain all frontier annotators on
# the manual subset). Switch this to gpt-5-mini-only for the single-judge
# variant analysis.
MODEL_ANNOTATIONS_PATH = "data/04_annotated/models_annotations_final.csv"
MANUAL_ANNOTATIONS_PATH = "data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv"

# Annotators to drop from the analysis (e.g. baselines we know are bad).
# Per the user's earlier setup, BERT was included only as a sanity-check
# baseline and should be excluded from the headline numbers.
ANNOTATORS_TO_EXCLUDE = ["monologg/bert-base-cased"]

OUTPUT_DIR = "figures/inter_rater_agreement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bootstrap settings
N_BOOTSTRAPS = 2000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Taxonomies analyzed
TAXONOMIES = ["ekman_basic_emotions", "go_emotions"]


def parse_labels_robust(label_string):
    """Parse a string like '['anger', 'sadness']' into ['anger', 'sadness']."""
    if not isinstance(label_string, str) or label_string.strip() == "":
        return []
    cleaned = label_string.strip().strip('[]"\'')
    if not cleaned:
        return []
    return [
        lbl.strip().strip('\'"').lower()
        for lbl in cleaned.split(",")
        if lbl.strip()
    ]


def masi_distance(set_a, set_b):
    """
    MASI (Measuring Agreement on Set-valued Items) distance.

    Passonneau (2006). Standard distance metric for multi-label annotation
    in NLP. Values in [0, 1] where 0 = identical sets, 1 = disjoint sets.

    Distance = 1 - (Jaccard * monotonicity_factor)
    monotonicity_factor: 1 if equal, 0.67 if one subset of the other,
                        0.33 if intersection nonempty, 0 if disjoint.
    """
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    intersection = a & b
    if not intersection:
        return 1.0  # disjoint

    union = a | b
    jaccard = len(intersection) / len(union)

    # Monotonicity factor (Passonneau 2006)
    if a == b:
        m = 1.0
    elif a.issubset(b) or b.issubset(a):
        m = 0.67
    else:
        m = 0.33

    return 1.0 - (jaccard * m)


def jaccard_similarity(set_a, set_b):
    """Jaccard set similarity. 1 = identical, 0 = disjoint."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def krippendorff_alpha_nominal(annotations_per_rater):
    """
    Nominal Krippendorff's alpha.

    Input: list of lists of single labels (one per item per rater).
    Conservative: only single-label match counts as agreement.
    """
    # Build reliability matrix: rows = raters, cols = items
    # Use np.nan for missing
    n_raters = len(annotations_per_rater)
    n_items = len(annotations_per_rater[0])

    # Encode labels to integers for nominal alpha
    all_labels = set()
    for rater_anns in annotations_per_rater:
        for ann in rater_anns:
            if ann:
                all_labels.add(ann[0] if isinstance(ann, list) else ann)
    label_to_int = {lbl: i for i, lbl in enumerate(sorted(all_labels))}

    matrix = np.full((n_raters, n_items), np.nan)
    for r, rater_anns in enumerate(annotations_per_rater):
        for i, ann in enumerate(rater_anns):
            if ann:
                top = ann[0] if isinstance(ann, list) else ann
                if top in label_to_int:
                    matrix[r, i] = label_to_int[top]

    return krippendorff.alpha(
        reliability_data=matrix, level_of_measurement="nominal"
    )


def krippendorff_alpha_masi(annotations_per_rater):
    """
    Krippendorff's alpha with MASI distance.

    Multi-label-aware. Each cell holds a set of labels, distance computed
    pairwise via MASI. This is the headline metric for multi-label emotion
    annotation.

    Implementation follows Krippendorff (2011, 'Computing Krippendorff's
    Alpha-Reliability'): general formula
        alpha = 1 - (D_observed / D_expected)
    where D values are mean pairwise distances under the chosen metric.
    """
    n_raters = len(annotations_per_rater)
    n_items = len(annotations_per_rater[0])

    # Convert annotations to sets (handle empty -> None for masking)
    sets_per_rater = []
    for rater_anns in annotations_per_rater:
        rater_sets = []
        for ann in rater_anns:
            if not ann:
                rater_sets.append(None)
            else:
                if isinstance(ann, list):
                    rater_sets.append(frozenset(ann))
                else:
                    rater_sets.append(frozenset([ann]))
        sets_per_rater.append(rater_sets)

    # Observed disagreement: average MASI over all pairs (i, j)
    # within the same item, across all items
    obs_distances = []
    for i in range(n_items):
        valid_sets = [
            sets_per_rater[r][i] for r in range(n_raters)
            if sets_per_rater[r][i] is not None
        ]
        if len(valid_sets) < 2:
            continue
        for s1, s2 in combinations(valid_sets, 2):
            obs_distances.append(masi_distance(s1, s2))

    if not obs_distances:
        return float("nan")
    D_obs = np.mean(obs_distances)

    # Expected disagreement: average MASI over all pairs of annotations
    # regardless of item
    all_valid = [
        sets_per_rater[r][i]
        for r in range(n_raters) for i in range(n_items)
        if sets_per_rater[r][i] is not None
    ]
    if len(all_valid) < 2:
        return float("nan")

    # Sample-based estimate of expected distance to avoid O(N^2) on large data
    # For N <= 1000 we compute exactly; otherwise sample 500k pairs
    n = len(all_valid)
    n_pairs_total = n * (n - 1) // 2
    if n_pairs_total <= 500_000:
        exp_distances = [
            masi_distance(s1, s2) for s1, s2 in combinations(all_valid, 2)
        ]
    else:
        rng = np.random.RandomState(RANDOM_SEED)
        idx_pairs = rng.randint(0, n, size=(500_000, 2))
        idx_pairs = idx_pairs[idx_pairs[:, 0] != idx_pairs[:, 1]]
        exp_distances = [
            masi_distance(all_valid[i], all_valid[j])
            for i, j in idx_pairs
        ]
    D_exp = np.mean(exp_distances)

    if D_exp == 0:
        return float("nan")
    return 1.0 - (D_obs / D_exp)


def jaccard_mean(annotations_a, annotations_b):
    """Mean Jaccard similarity between two annotators across paired items."""
    sims = []
    for ann_a, ann_b in zip(annotations_a, annotations_b):
        if not ann_a or not ann_b:
            continue
        sims.append(jaccard_similarity(ann_a, ann_b))
    if not sims:
        return float("nan")
    return np.mean(sims)


def bootstrap_ci(values, ci=0.95, seed=42):
    """Percentile bootstrap CI."""
    alpha = (1 - ci) / 2
    return (np.percentile(values, alpha * 100),
            np.percentile(values, (1 - alpha) * 100))


def compute_pairwise_agreement(
    df, taxonomy_col_pred, taxonomy_col_gt,
    rater_a_label, rater_b_label,
    rater_a_data, rater_b_data
):
    """
    Compute headline agreement metrics + bootstrap CI between two raters
    on a single taxonomy. Returns a dict.
    """
    # Build paired arrays
    paired_a, paired_b = [], []
    for ann_a, ann_b in zip(rater_a_data, rater_b_data):
        if ann_a and ann_b:
            paired_a.append(ann_a)
            paired_b.append(ann_b)

    if len(paired_a) < 5:
        return None

    # Point estimates
    alpha_nom = krippendorff_alpha_nominal([paired_a, paired_b])
    alpha_masi = krippendorff_alpha_masi([paired_a, paired_b])
    jac = jaccard_mean(paired_a, paired_b)

    # Bootstrap (lighter than 2k for inter-model; use N_BOOTSTRAPS // 2)
    n_boots = N_BOOTSTRAPS // 2
    rng = np.random.RandomState(RANDOM_SEED)
    boots_nom, boots_masi, boots_jac = [], [], []
    n = len(paired_a)
    for _ in range(n_boots):
        idx = rng.choice(n, size=n, replace=True)
        b_a = [paired_a[i] for i in idx]
        b_b = [paired_b[i] for i in idx]
        try:
            boots_nom.append(krippendorff_alpha_nominal([b_a, b_b]))
        except Exception:
            pass
        try:
            boots_masi.append(krippendorff_alpha_masi([b_a, b_b]))
        except Exception:
            pass
        boots_jac.append(jaccard_mean(b_a, b_b))

    nom_ci = bootstrap_ci(boots_nom) if boots_nom else (np.nan, np.nan)
    masi_ci = bootstrap_ci(boots_masi) if boots_masi else (np.nan, np.nan)
    jac_ci = bootstrap_ci(boots_jac)

    return {
        "rater_a": rater_a_label,
        "rater_b": rater_b_label,
        "n_paired": len(paired_a),
        "alpha_nominal": alpha_nom,
        "alpha_nominal_low": nom_ci[0],
        "alpha_nominal_upp": nom_ci[1],
        "alpha_masi": alpha_masi,
        "alpha_masi_low": masi_ci[0],
        "alpha_masi_upp": masi_ci[1],
        "jaccard": jac,
        "jaccard_low": jac_ci[0],
        "jaccard_upp": jac_ci[1],
    }


def main():
    print("Loading data...")
    manual = pd.read_csv(MANUAL_ANNOTATIONS_PATH)
    models = pd.read_csv(MODEL_ANNOTATIONS_PATH)

    if "model" in models.columns:
        models.rename(columns={"model": "annotator"}, inplace=True)

    df = pd.merge(manual, models, on="response_text")

    # Parse all label columns
    parse_map = {
        "ekman_manual_labels_list": "ekman_manual_label",
        "go_emotions_manual_labels_list": "go_emotions_manual_label",
        "ekman_annotator_labels_list": "ekman_labels",
        "go_emotions_annotator_labels_list": "go_emotions_labels",
    }
    for new_col, old_col in parse_map.items():
        df[new_col] = df[old_col].apply(parse_labels_robust)

    # Filter excluded annotators
    df = df[~df["annotator"].isin(ANNOTATORS_TO_EXCLUDE)].copy()
    annotators = sorted(df["annotator"].unique())
    print(f"Annotators in analysis: {annotators}")
    print(f"Total rows after merge: {len(df)}")

    # =========================================================================
    # Analysis 1: Each annotator vs human ground truth, per taxonomy
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Annotator vs human ground truth")
    print("=" * 70)

    vs_human_results = []
    for tax_short, gt_col, pred_col in [
        ("ekman", "ekman_manual_labels_list", "ekman_annotator_labels_list"),
        ("go_emotions", "go_emotions_manual_labels_list",
         "go_emotions_annotator_labels_list"),
    ]:
        for annotator in tqdm(annotators, desc=f"{tax_short} vs human"):
            df_ann = df[df["annotator"] == annotator]
            gt_data = df_ann[gt_col].tolist()
            pred_data = df_ann[pred_col].tolist()

            result = compute_pairwise_agreement(
                df_ann, pred_col, gt_col,
                "human", annotator,
                gt_data, pred_data,
            )
            if result is not None:
                result["taxonomy"] = tax_short
                vs_human_results.append(result)

    vs_human_df = pd.DataFrame(vs_human_results)
    vs_human_df.to_csv(
        os.path.join(OUTPUT_DIR, "annotator_vs_human.csv"), index=False
    )
    print(f"\nSaved: {OUTPUT_DIR}/annotator_vs_human.csv")
    print(vs_human_df[
        ["taxonomy", "rater_b", "alpha_nominal", "alpha_masi", "jaccard",
         "n_paired"]
    ].to_string(index=False))

    # =========================================================================
    # Analysis 2: Inter-model agreement matrix, per taxonomy
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Inter-model agreement matrix")
    print("=" * 70)

    inter_model_results = []
    for tax_short, pred_col in [
        ("ekman", "ekman_annotator_labels_list"),
        ("go_emotions", "go_emotions_annotator_labels_list"),
    ]:
        # Pivot to get one column per annotator, indexed by response_text
        pivot = df.pivot_table(
            index="response_text",
            columns="annotator",
            values=pred_col,
            aggfunc="first",
        )
        # Drop rows where any annotator is missing for fair pairwise comparison
        pivot = pivot.dropna()

        pairs = list(combinations(annotators, 2))
        for ann_a, ann_b in tqdm(pairs, desc=f"{tax_short} pairs"):
            data_a = pivot[ann_a].tolist()
            data_b = pivot[ann_b].tolist()

            result = compute_pairwise_agreement(
                df, pred_col, pred_col,
                ann_a, ann_b,
                data_a, data_b,
            )
            if result is not None:
                result["taxonomy"] = tax_short
                inter_model_results.append(result)

    inter_df = pd.DataFrame(inter_model_results)
    inter_df.to_csv(
        os.path.join(OUTPUT_DIR, "inter_model_agreement.csv"), index=False
    )
    print(f"\nSaved: {OUTPUT_DIR}/inter_model_agreement.csv")

    # =========================================================================
    # Visualization 1: Bar plot of vs-human agreement
    # =========================================================================
    print("\nGenerating visualizations...")
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 11})

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, tax in zip(axes, ["ekman", "go_emotions"]):
        sub = vs_human_df[vs_human_df["taxonomy"] == tax].sort_values(
            "alpha_masi", ascending=False
        )
        x = np.arange(len(sub))
        width = 0.27

        ax.bar(
            x - width, sub["alpha_nominal"], width,
            yerr=[sub["alpha_nominal"] - sub["alpha_nominal_low"],
                  sub["alpha_nominal_upp"] - sub["alpha_nominal"]],
            label="α (nominal)", color="#4878d0", capsize=4,
        )
        ax.bar(
            x, sub["alpha_masi"], width,
            yerr=[sub["alpha_masi"] - sub["alpha_masi_low"],
                  sub["alpha_masi_upp"] - sub["alpha_masi"]],
            label="α (MASI)", color="#ee854a", capsize=4,
        )
        ax.bar(
            x + width, sub["jaccard"], width,
            yerr=[sub["jaccard"] - sub["jaccard_low"],
                  sub["jaccard_upp"] - sub["jaccard"]],
            label="Jaccard", color="#6acc64", capsize=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["rater_b"], rotation=30, ha="right")
        ax.set_title(f"{tax.replace('_', ' ').title()}: Annotator vs Human")
        ax.set_ylim(0, 1)
        ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5,
                   label="α = 0.6 (substantial)" if tax == "ekman" else None)
        ax.set_ylabel("Agreement")
        ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "annotator_vs_human.png"), dpi=300
    )
    plt.close()

    # =========================================================================
    # Visualization 2: Inter-model heatmap (MASI alpha)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, tax in zip(axes, ["ekman", "go_emotions"]):
        sub = inter_df[inter_df["taxonomy"] == tax]

        # Build symmetric matrix
        mat = pd.DataFrame(
            np.nan, index=annotators, columns=annotators
        )
        for _, row in sub.iterrows():
            mat.loc[row["rater_a"], row["rater_b"]] = row["alpha_masi"]
            mat.loc[row["rater_b"], row["rater_a"]] = row["alpha_masi"]
        np.fill_diagonal(mat.values, 1.0)

        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Krippendorff α (MASI)"},
        )
        ax.set_title(
            f"{tax.replace('_', ' ').title()}: Inter-model agreement (MASI α)"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "inter_model_heatmap_masi.png"), dpi=300
    )
    plt.close()

    # Same for Jaccard
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, tax in zip(axes, ["ekman", "go_emotions"]):
        sub = inter_df[inter_df["taxonomy"] == tax]
        mat = pd.DataFrame(np.nan, index=annotators, columns=annotators)
        for _, row in sub.iterrows():
            mat.loc[row["rater_a"], row["rater_b"]] = row["jaccard"]
            mat.loc[row["rater_b"], row["rater_a"]] = row["jaccard"]
        np.fill_diagonal(mat.values, 1.0)

        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Mean Jaccard"},
        )
        ax.set_title(
            f"{tax.replace('_', ' ').title()}: Inter-model agreement "
            f"(Jaccard)"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "inter_model_heatmap_jaccard.png"), dpi=300
    )
    plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Files:")
    print("  - annotator_vs_human.csv")
    print("  - inter_model_agreement.csv")
    print("  - annotator_vs_human.png")
    print("  - inter_model_heatmap_masi.png")
    print("  - inter_model_heatmap_jaccard.png")


if __name__ == "__main__":
    main()