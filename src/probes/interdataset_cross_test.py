import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.utils import resample

LLM_USED = "Llama-2-7b-chat-hf"
MODELS_DIR_BASE = "models"
FIGURES_DIR = os.path.join("figures", f"cross_testing_performance_{LLM_USED}")

PATH_HUMAN = "data/03_activations/MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
PATH_GEN = "data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"

NAME_GEN = "generated_prompts"
NAME_HUMAN = "human_centric"

TAXONOMIES = ["ekman_basic_emotions", "plutchik_wheel", "go_emotions"]
LAYERS = range(33)
ACTIVATION_COL = "last_token_activation"

# Statistical parameters
BOOTSTRAP_ITERATIONS = 10000 
CONFIDENCE_LEVEL = 0.95  
RANDOM_SEED = 42

os.makedirs(FIGURES_DIR, exist_ok=True)

print("LOADING DATASETS...")
df_human = pd.read_pickle(PATH_HUMAN)
df_gen = pd.read_pickle(PATH_GEN)
print(f"Datasets loaded.")

def get_data_for_layer_strict(df, layer, taxonomy):
    """Extracts X and Y (first label)."""
    X = []
    y_ground_truth = [] 

    for row in df.itertuples():
        try:
            nested_df = row.activations

            if layer in nested_df.index:
                act_vector = nested_df.loc[layer, ACTIVATION_COL]

            else:
                continue 
            
            labels_list = getattr(row, taxonomy)

            if not isinstance(labels_list, list) or len(labels_list) == 0:
                continue
            
            primary_label = labels_list[0]

            if isinstance(act_vector, np.ndarray):
                X.append(act_vector)
                y_ground_truth.append(primary_label)
                
        except Exception:
            continue

    if len(X) == 0:
        return None, None
    return np.stack(X), np.array(y_ground_truth)

def load_model(dataset_prefix, taxonomy, layer):
    filename = f"{dataset_prefix}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
    path = os.path.join(MODELS_DIR_BASE, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        
        except:
            return None
        
    return None

# Percentiles stored from the bootstrap distribution so any dispersion band
# (percentile CI, mean +- k*SD, ...) can be drawn later without re-bootstrapping.
PERCENTILES = [2.5, 5, 16, 25, 75, 84, 95, 97.5]


def bootstrap_f1(y_true, y_pred, n_iterations=10000, seed=42):
    """Bootstrap distribution of macro F1.

    Stratified (preserving class proportions) when every class has >= 2 examples;
    otherwise falls back to a simple bootstrap. Returns the array of replicates.
    """
    rng = np.random.RandomState(seed)
    _, counts = np.unique(y_true, return_counts=True)
    stratify_possible = np.all(counts >= 2)

    boot = np.empty(n_iterations)
    for i in range(n_iterations):
        iter_seed = rng.randint(0, 2**32 - 1)
        if stratify_possible:
            y_t, y_p = resample(y_true, y_pred, replace=True,
                                stratify=y_true, random_state=iter_seed)
        else:
            y_t, y_p = resample(y_true, y_pred, replace=True, random_state=iter_seed)
        boot[i] = f1_score(y_t, y_p, average="macro", zero_division=0)
    return boot


def summarize_boot(boot):
    """Summary stats of a bootstrap distribution: mean, std (SE) and percentiles."""
    out = {"f1_mean": float(np.mean(boot)), "f1_std": float(np.std(boot, ddof=1))}
    for p in PERCENTILES:
        out[f"f1_p{str(p).replace('.', '_')}"] = float(np.percentile(boot, p))
    out["f1_lower"] = out["f1_p2_5"]    # backward-compat (95% CI)
    out["f1_upper"] = out["f1_p97_5"]
    return out


results = []

print(f"\nSTARTING ULTRA-ROBUST CROSS-TESTING ({BOOTSTRAP_ITERATIONS} iters, Stratified)...")

for taxonomy in TAXONOMIES:
    print(f"\nRunning on {taxonomy}")
    
    for layer in tqdm(LAYERS, desc=f"Layers ({taxonomy})"):
        
        X_human, y_human = get_data_for_layer_strict(df_human, layer, taxonomy)
        X_gen, y_gen = get_data_for_layer_strict(df_gen, layer, taxonomy)
        
        if X_human is None or X_gen is None:
            continue
            
        # Test A: Gen -> Human
        model_gen = load_model(NAME_GEN, taxonomy, layer)
        if model_gen is not None:
            y_pred_gh = model_gen.predict(X_human)
            boot = bootstrap_f1(y_human, y_pred_gh, BOOTSTRAP_ITERATIONS, RANDOM_SEED)
            row = {"taxonomy": taxonomy, "layer": layer,
                   "train_source": "Generated", "test_source": "Human"}
            row.update(summarize_boot(boot))
            results.append(row)

        # Test B: Human -> Gen
        model_human = load_model(NAME_HUMAN, taxonomy, layer)
        if model_human is not None:
            y_pred_hg = model_human.predict(X_gen)
            boot = bootstrap_f1(y_gen, y_pred_hg, BOOTSTRAP_ITERATIONS, RANDOM_SEED)
            row = {"taxonomy": taxonomy, "layer": layer,
                   "train_source": "Human", "test_source": "Generated"}
            row.update(summarize_boot(boot))
            results.append(row)


if not results:
    exit()

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(FIGURES_DIR, "cross_test_bootstrap_results.csv"), index=False)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

for tax in TAXONOMIES:
    subset = results_df[results_df["taxonomy"] == tax]
    if subset.empty: continue
        
    plt.figure(figsize=(12, 7))
    
    # SERIE 1: Gen -> Human
    data_gh = subset[subset["train_source"] == "Generated"].sort_values("layer")
    
    plt.plot(data_gh["layer"], data_gh["f1_mean"], 
             label="Train: Generated $\\to$ Test: Human", 
             color="#1f77b4", linewidth=2)
    
    plt.fill_between(data_gh["layer"], data_gh["f1_lower"], data_gh["f1_upper"], 
                     color="#1f77b4", alpha=0.25)

    # SERIE 2: Human -> Gen
    data_hg = subset[subset["train_source"] == "Human"].sort_values("layer")
    
    plt.plot(data_hg["layer"], data_hg["f1_mean"], 
             label="Train: Human $\\to$ Test: Generated", 
             color="#ff7f0e", linewidth=2)
    
    plt.fill_between(data_hg["layer"], data_hg["f1_lower"], data_hg["f1_upper"], 
                     color="#ff7f0e", alpha=0.25)
    
    # SERIE 3: Gen -> Gen (baseline)
    data_gg = pd.read_csv("results/probes_generated_prompts_Llama-2-7b-chat-hf/full_probing_metrics_Llama-2-7b-chat-hf_final_F1.csv")

    plt.plot(data_gg["layer"], data_gg["macro_f1"], 
             label="Train: Generated $\\to$ Test: Generated", 
             color="#1f77b4", linewidth=2, linestyle="--")
    
    # SERIE 4: Human -> Human (baseline)
    data_hh = pd.read_csv("results/probes_human_centric_Llama-2-7b-chat-hf/full_probing_metrics_Llama-2-7b-chat-hf_final_F1.csv")

    plt.plot(data_hh["layer"], data_hh["macro_f1"], 
             label="Train: Human $\\to$ Test: Human", 
             color="#ff7f0e", linewidth=2, linestyle="--")

    plt.title(f"Cross-Dataset Robustness (Stratified Bootstrap 10k)\nTaxonomy: {tax} | Metric: Macro F1", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Macro F1-Score (with 95% CI)", fontsize=12)
    plt.legend(loc="lower right")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(FIGURES_DIR, f"cross_test_{tax}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"\nAnalysis completed! Plots saved in: {FIGURES_DIR}")