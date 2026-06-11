"""
Inter-taxonomy RSA.

Instead of comparing each taxonomy's Representational Dissimilarity Matrix (RDM)
against the LLM activation RDMs (as in RSA_multilabel.py), here we compare the
taxonomy RDMs *against each other*. This tells us how geometrically aligned the
emotion theories are over the SAME set of stimuli, independently of any model.

Note: a taxonomy RDM depends only on the labels assigned to each stimulus, NOT on
the layer or the LLM. We still take Llama-2-7b as the "subject" LLM because it
fixes the stimulus set and the stratified-sampling scheme (the rows that survived
activation extraction). The result is model-agnostic, but reported on Llama's set.
"""

import os
import itertools
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_NAME = "Llama-2-7b-chat-hf"          # subject LLM (fixes the stimulus set)
DATASET = "generated_prompts"

DATA_PATH = "data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
SENTIMENT_TARGETS = ['ekman_basic_emotions', 'plutchik_wheel', 'go_emotions']
PRETTY = {
    'ekman_basic_emotions': 'Ekman',
    'plutchik_wheel': 'Plutchik',
    'go_emotions': 'GoEmotions',
}
OUTPUT_DIR = f"results/{LLM_NAME}_{DATASET}/rsa_taxonomy_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RSA / bootstrap parameters
N_ITERATIONS = 200          # bootstrap iterations for the CIs
SAMPLES_PER_CLASS = 15      # stratified sampling on the primary go_emotions label
SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers (mirrored from RSA_multilabel.py)
# ---------------------------------------------------------------------------
def compute_rdm_torch(tensor):
    """Dissimilarity Matrix (1 - Cosine Sim) on GPU."""
    norm_tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    cosine_sim = torch.mm(norm_tensor, norm_tensor.t())
    return 1 - cosine_sim


def get_weighted_rdm(list_of_lists_labels):
    """
    Build a stimulus x stimulus RDM from multilabel emotion lists.

    1. Vocabulary of all unique labels in this batch.
    2. Each stimulus -> vector with reciprocal-rank weights
       (1st label = 1.0, 2nd = 0.5, 3rd = 0.33, ...).
    3. RDM = cosine distance between those vectors.
    """
    unique_labels = sorted(list(set([lbl for sub in list_of_lists_labels for lbl in sub])))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    n_samples = len(list_of_lists_labels)
    n_dims = len(unique_labels)
    matrix = torch.zeros((n_samples, n_dims), device=DEVICE, dtype=torch.float32)

    for i, labels in enumerate(list_of_lists_labels):
        for rank, label in enumerate(labels):
            if label in label_to_idx:
                matrix[i, label_to_idx[label]] = 1.0 / (rank + 1)

    return compute_rdm_torch(matrix)


def upper_triangle(matrix):
    """Flatten the strict upper triangle of a square RDM to a 1D numpy vector."""
    idx = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[idx[0], idx[1]].cpu().numpy()


def spearman_vec(vec_a, vec_b):
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 0.0
    return spearmanr(vec_a, vec_b).correlation


def get_stratified_indices(df, theory_column, samples_per_class=15):
    """Stratify on the PRIMARY label (first element of each list)."""
    indices = []
    primary = df[theory_column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "None")
    for label in primary.unique():
        if label == "None":
            continue
        class_indices = df[primary == label].index.values
        n = min(len(class_indices), samples_per_class)
        if n > 0:
            indices.extend(np.random.choice(class_indices, n, replace=False))
    return np.array(indices)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_pickle(DATA_PATH)

for target in SENTIMENT_TARGETS:
    mask = df[target].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df = df[mask].copy()
df = df.reset_index(drop=True)
print(f"Data ready: {len(df)} stimuli with all three taxonomies present.")

pairs = list(itertools.combinations(SENTIMENT_TARGETS, 2))


# ---------------------------------------------------------------------------
# 2. Bootstrap inter-taxonomy RSA
# ---------------------------------------------------------------------------
boot = {pair: [] for pair in pairs}            # observed pairwise Spearman
boot_shuffle = {pair: [] for pair in pairs}    # chance baseline (one tax shuffled)

for _ in tqdm(range(N_ITERATIONS), desc="Bootstrap"):
    idx = get_stratified_indices(df, 'go_emotions', samples_per_class=SAMPLES_PER_CLASS)
    if len(idx) < 20:
        continue

    # Build one RDM (and its upper triangle) per taxonomy on this subset
    vecs = {}
    vecs_shuffled = {}
    for target in SENTIMENT_TARGETS:
        labels = df[target].iloc[idx].tolist()
        vecs[target] = upper_triangle(get_weighted_rdm(labels))

        shuffled = np.random.permutation(np.array(labels, dtype=object)).tolist()
        vecs_shuffled[target] = upper_triangle(get_weighted_rdm(shuffled))

    for a, b in pairs:
        boot[(a, b)].append(spearman_vec(vecs[a], vecs[b]))
        # chance: real RDM of A vs shuffled RDM of B
        boot_shuffle[(a, b)].append(spearman_vec(vecs[a], vecs_shuffled[b]))


# ---------------------------------------------------------------------------
# 3. Aggregate
# ---------------------------------------------------------------------------
rows = []
for a, b in pairs:
    arr = np.array(boot[(a, b)])
    sh = np.array(boot_shuffle[(a, b)])
    rows.append({
        'taxonomy_a': a,
        'taxonomy_b': b,
        'mean_corr': arr.mean(),
        'std_corr': arr.std(),
        'ci_low': np.percentile(arr, 2.5),
        'ci_high': np.percentile(arr, 97.5),
        'shuffle_mean': sh.mean(),
        'shuffle_ci_high': np.percentile(sh, 97.5),
    })
results_df = pd.DataFrame(rows)
results_df.to_csv(f"{OUTPUT_DIR}/taxonomy_rsa_metrics.csv", index=False)
print("\nPairwise inter-taxonomy Spearman (bootstrap mean [95% CI]):")
for r in rows:
    print(f"  {PRETTY[r['taxonomy_a']]:>10} vs {PRETTY[r['taxonomy_b']]:<10}: "
          f"{r['mean_corr']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  "
          f"(chance {r['shuffle_mean']:.3f})")

# Build symmetric correlation matrix for the heatmap
corr_mat = pd.DataFrame(np.eye(len(SENTIMENT_TARGETS)),
                        index=[PRETTY[t] for t in SENTIMENT_TARGETS],
                        columns=[PRETTY[t] for t in SENTIMENT_TARGETS])
for r in rows:
    a, b = PRETTY[r['taxonomy_a']], PRETTY[r['taxonomy_b']]
    corr_mat.loc[a, b] = r['mean_corr']
    corr_mat.loc[b, a] = r['mean_corr']


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

# PLOT 1: Heatmap of pairwise RDM correlations
plt.figure(figsize=(7, 6))
sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1,
            square=True, cbar_kws={'label': 'Spearman correlation (RDM upper-tri)'},
            linewidths=0.5, linecolor='white')
plt.title(f"Inter-taxonomy RSA — RDM alignment\n(subject LLM: {LLM_NAME}, {N_ITERATIONS} bootstraps)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_taxonomy_rsa_heatmap.png", dpi=150)
plt.close()

# PLOT 2: Bar chart of the three unique pairs with 95% CI + shuffle baseline
plt.figure(figsize=(9, 6))
labels_x = [f"{PRETTY[a]}\nvs {PRETTY[b]}" for a, b in pairs]
means = [r['mean_corr'] for r in rows]
err_low = [r['mean_corr'] - r['ci_low'] for r in rows]
err_high = [r['ci_high'] - r['mean_corr'] for r in rows]
x = np.arange(len(pairs))

bars = plt.bar(x, means, yerr=[err_low, err_high], capsize=6,
               color='tab:teal', alpha=0.85, label='Observed (95% CI)')
# shuffle chance band (max upper CI across pairs as a reference line)
for i, r in enumerate(rows):
    plt.hlines(r['shuffle_ci_high'], x[i] - 0.4, x[i] + 0.4,
               color='tab:red', linestyle='--',
               label='Shuffle 97.5% (chance)' if i == 0 else None)

plt.xticks(x, labels_x)
plt.ylabel("Spearman correlation between RDMs")
plt.title(f"How geometrically aligned are the emotion taxonomies?\n(subject LLM: {LLM_NAME})")
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_taxonomy_rsa_bars.png", dpi=150)
plt.close()

# PLOT 3: Side-by-side RDM heatmaps on one representative sample
# Sort the sample by the primary Ekman emotion so block structure is visible.
np.random.seed(SEED)  # reproducible representative subset
idx = get_stratified_indices(df, 'go_emotions', samples_per_class=SAMPLES_PER_CLASS)
sub = df.iloc[idx].copy()
sort_key = sub['ekman_basic_emotions'].apply(lambda x: x[0])
order = np.argsort(sort_key.values, kind='stable')
sub = sub.iloc[order]

fig, axes = plt.subplots(1, len(SENTIMENT_TARGETS),
                         figsize=(6 * len(SENTIMENT_TARGETS), 6))
for ax, target in zip(axes, SENTIMENT_TARGETS):
    rdm = get_weighted_rdm(sub[target].tolist()).cpu().numpy()
    sns.heatmap(rdm, ax=ax, cmap="magma", vmin=0, vmax=1, square=True, cbar=True)
    ax.set_title(f"{PRETTY[target]} RDM")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Taxonomy RDMs on a shared sample (rows sorted by primary Ekman emotion)",
             fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_taxonomy_rdms_side_by_side.png", dpi=150)
plt.close()

# PLOT 4: Dendrogram of taxonomies (distance = 1 - mean correlation)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

dist_mat = 1.0 - corr_mat.values
np.fill_diagonal(dist_mat, 0.0)
condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method='average')

plt.figure(figsize=(7, 5))
dendrogram(Z, labels=[PRETTY[t] for t in SENTIMENT_TARGETS], leaf_font_size=12)
plt.ylabel("1 - Spearman(RDM, RDM)")
plt.title("Taxonomy clustering by representational geometry")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_taxonomy_dendrogram.png", dpi=150)
plt.close()

print(f"\nSUCCESS. Metrics + 4 plots saved in {OUTPUT_DIR}")
