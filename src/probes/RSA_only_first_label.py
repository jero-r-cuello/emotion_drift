import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_NAME = "Llama-2-7b-chat-hf"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
SENTIMENT_TARGETS = ['ekman_basic_emotions', 'plutchik_wheel', 'go_emotions']
OUTPUT_DIR = "/home/jcuello/emotion_drift/results/rsa_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RSA Parameters
N_ITERATIONS = 50        
SAMPLES_PER_CLASS = 15   

# --- Helper Functions ---

def compute_rdm_torch(tensor):
    """Computes a Dissimilarity Matrix (1 - Cosine Sim) using GPU."""
    norm_tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    cosine_sim = torch.mm(norm_tensor, norm_tensor.t())
    return 1 - cosine_sim

def get_categorical_rdm(labels):
    """Theoretical RDM: 0 if same category, 1 if different."""
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    label_tensor = torch.tensor(encoded, device=DEVICE).unsqueeze(1)
    return (label_tensor != label_tensor.t()).float()

def compute_lexical_rdm(texts):
    """Computes a RDM based on word overlap (Bag of Words) to control for semantics."""
    # Use binary=True to just check word presence/absence
    vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=5000)
    try:
        bow = vectorizer.fit_transform(texts).toarray()
        # Cosine distance on BoW vectors
        dist = pairwise_distances(bow, metric='cosine')
        return torch.tensor(dist, device=DEVICE, dtype=torch.float32)
    except ValueError:
        # Fallback for empty text or errors
        return torch.zeros((len(texts), len(texts)), device=DEVICE)

def spearman_correlation(matrix_a, matrix_b):
    """Spearman correlation between upper triangles."""
    indices = torch.triu_indices(matrix_a.shape[0], matrix_a.shape[1], offset=1)
    vec_a = matrix_a[indices[0], indices[1]].cpu().numpy()
    vec_b = matrix_b[indices[0], indices[1]].cpu().numpy()
    if np.std(vec_a) == 0 or np.std(vec_b) == 0: return 0.0
    return spearmanr(vec_a, vec_b).correlation

def partial_correlation(target_vec, model_vec, control_vec):
    """Cor(Target, Model | Control)"""
    def get_residuals(x, y):
        x = x.reshape(-1, 1)
        return y - LinearRegression().fit(x, y).predict(x)
    res_target = get_residuals(control_vec, target_vec)
    res_model = get_residuals(control_vec, model_vec)
    if np.std(res_target) == 0 or np.std(res_model) == 0: return 0.0
    return spearmanr(res_target, res_model).correlation

def get_stratified_indices(df, theory_column, samples_per_class=15):
    indices = []
    unique_labels = df[theory_column].unique()
    for label in unique_labels:
        class_indices = df[df[theory_column] == label].index.values
        n_to_sample = min(len(class_indices), samples_per_class)
        if n_to_sample > 0:
            sampled = np.random.choice(class_indices, n_to_sample, replace=False)
            indices.extend(sampled)
    return np.array(indices)

def plot_complexity_percentage(df_results, output_path):
    plt.figure(figsize=(14, 8))
    ax1 = plt.gca()
    
    go_data = df_results[df_results['taxonomy'] == 'go_emotions'].copy()
    
    # Calculate % Unique Info
    go_data['unique_info_ratio'] = (go_data['complex_partial'] / go_data['mean_corr']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

    line1, = ax1.plot(go_data['layer'], go_data['mean_corr'], 
                      label='Total GoEmotions', marker='o', color='tab:green', linewidth=3)
    line2, = ax1.plot(go_data['layer'], go_data['complex_partial'], 
                      label='Unique (Partial Ekman)', linestyle='--', marker='s', color='tab:red', linewidth=2)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Spearman Correlation', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(go_data['layer'], 0, go_data['unique_info_ratio'], 
                     color='tab:purple', alpha=0.1, label='% Unique Information')
    ax2.plot(go_data['layer'], go_data['unique_info_ratio'], 
             color='tab:purple', alpha=0.5, linestyle=':', linewidth=1)
    
    ax2.set_ylabel('Percentage of Unique Info (%)', color='tab:purple', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax2.set_ylim(0, 100)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f"Complexity Analysis: How much DOES GoEmotions add beyond Ekman?\n(LLM: {LLM_NAME})", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- 1. Load Data ---
print(f"Loading data from {DATA_PATH}...")
df_original = pd.read_pickle(DATA_PATH)

# Clean Labels
for target in SENTIMENT_TARGETS:
    mask = df_original[target].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df_original = df_original[mask].copy()
    df_original[target] = df_original[target].str[0]

# Text Column for Lexical Control
text_col = 'prompt'

num_layers = len(df_original['activations'].iloc[0])
print(f"Data ready: {len(df_original)} samples across {num_layers} layers.")

# --- 2. RSA Main Loop ---
all_layer_metrics = []

for layer_num in range(num_layers):
    print(f"\nProcessing Layer {layer_num}")
    
    # Robust Extraction (Fixes Layer 8 issues)
    X_list, y_df_list = [], []
    for i in range(len(df_original)):
        try:
            act = df_original['activations'].iloc[i].iloc[layer_num]['last_token_activation']
            if isinstance(act, np.ndarray) and act.squeeze().shape == (4096,):
                X_list.append(act.squeeze())
                y_df_list.append(df_original.iloc[i])
        except: continue
            
    if not X_list: continue

    X_layer_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32, device=DEVICE)
    df_layer = pd.DataFrame(y_df_list).reset_index(drop=True)
    
    # Metrics Storage
    layer_stats = {target: [] for target in SENTIMENT_TARGETS}
    layer_shuffle = {target: [] for target in SENTIMENT_TARGETS} # Control 1
    layer_lexical_partial = {target: [] for target in SENTIMENT_TARGETS} # Control 2
    layer_complex_partial = [] # Plot 3
    layer_noise_ceiling = []

    for i in tqdm(range(N_ITERATIONS), desc=f"Bootstrap L{layer_num}"):
        # Stratified Sampling (GoEmotions)
        idx = get_stratified_indices(df_layer, 'go_emotions', samples_per_class=SAMPLES_PER_CLASS)
        if len(idx) < 20: continue
        
        # Data RDM
        rdm_full = compute_rdm_torch(X_layer_tensor[idx])
        triu = torch.triu_indices(len(idx), len(idx), offset=1)
        data_vec = rdm_full[triu[0], triu[1]].cpu().numpy()

        # Control 2: Lexical RDM (if text col exists)
        lex_vec = None
        if text_col:
            texts = df_layer.iloc[idx][text_col].astype(str).tolist()
            rdm_lexical = compute_lexical_rdm(texts)
            lex_vec = rdm_lexical[triu[0], triu[1]].cpu().numpy()

        # Theory RDMs
        theory_vecs = {}
        for target in SENTIMENT_TARGETS:
            labels = df_layer[target].iloc[idx].tolist()
            rdm_theory = get_categorical_rdm(labels)
            theory_vec = rdm_theory[triu[0], triu[1]].cpu().numpy()
            theory_vecs[target] = theory_vec

            # 1. Standard Correlation
            layer_stats[target].append(spearman_correlation(rdm_full, rdm_theory))
            
            # 2. Control 1: Shuffle Labels
            shuffled_labels = np.random.permutation(labels)
            rdm_shuffle = get_categorical_rdm(shuffled_labels)
            layer_shuffle[target].append(spearman_correlation(rdm_full, rdm_shuffle))
            
            # 3. Control 2: Partial Lexical (Emotion | Words)
            if lex_vec is not None:
                layer_lexical_partial[target].append(partial_correlation(data_vec, theory_vec, lex_vec))
            else:
                layer_lexical_partial[target].append(np.nan)

        # 4. Complexity Analysis: GoEmotions unique beyond Ekman
        # Cor(Data, GoEmotions | Ekman)
        p_corr = partial_correlation(data_vec, theory_vecs['go_emotions'], theory_vecs['ekman_basic_emotions'])
        layer_complex_partial.append(p_corr)
        
        # 5. Noise Ceiling (Model Consistency - Upper Bound)
        # Comparamos la RDM real contra una RDM con un mínimo ruido (0.5%)
        # Esto mide qué tan estable es la geometría de esta capa.
        act_subset = X_layer_tensor[idx]
        noise = torch.randn_like(act_subset) * 0.005  # Ruido gaussiano muy leve
        rdm_noisy = compute_rdm_torch(act_subset + noise)
        
        # Correlación entre el modelo y sí mismo (con ruido)
        layer_noise_ceiling.append(spearman_correlation(rdm_full, rdm_noisy))

    # Aggregate
    for target in SENTIMENT_TARGETS:
        all_layer_metrics.append({
            'layer': layer_num, 
            'taxonomy': target,
            'mean_corr': np.mean(layer_stats[target]), 
            'std_corr': np.std(layer_stats[target]),
            'shuffle_corr': np.mean(layer_shuffle[target]),
            'lexical_partial': np.mean(layer_lexical_partial[target]),
            'complex_partial': np.mean(layer_complex_partial) if target == 'go_emotions' else np.nan,
            'noise_ceiling': np.mean(layer_noise_ceiling)
        })

# --- 3. Save Results and Plotting ---
results_df = pd.DataFrame(all_layer_metrics)
results_df.to_csv(f"{OUTPUT_DIR}/rsa_robust_metrics.csv", index=False)

colors = {'ekman_basic_emotions': 'tab:blue', 'plutchik_wheel': 'tab:orange', 'go_emotions': 'tab:green'}

# PLOT 1: Robust Comparison (Actual vs Shuffle)
plt.figure(figsize=(14, 8))
for target in SENTIMENT_TARGETS:
    sub = results_df[results_df['taxonomy'] == target]
    plt.plot(sub['layer'], sub['mean_corr'], label=f"{target}", color=colors[target], marker='o')
    plt.plot(sub['layer'], sub['shuffle_corr'], linestyle=':', color=colors[target], alpha=0.6)
    plt.fill_between(sub['layer'], sub['mean_corr']-sub['std_corr'], sub['mean_corr']+sub['std_corr'], color=colors[target], alpha=0.15)

ceilings = results_df.groupby('layer')['noise_ceiling'].mean()
#plt.fill_between(ceilings.index, ceilings.values, 1.0, color='gray', alpha=0.1, label='Noise Ceiling (Upper Bound)')
plt.title("RSA: Theories vs Chance Level (Shuffle Control)")
plt.ylabel("Spearman Correlation")
plt.xlabel("Layer")
plt.legend(title="Dashed = Shuffle Control")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_robust_comparison.png")
plt.close()

# PLOT 2: Lexical Control
if text_col:
    plt.figure(figsize=(14, 8))
    for target in SENTIMENT_TARGETS:
        sub = results_df[results_df['taxonomy'] == target]
        plt.plot(sub['layer'], sub['lexical_partial'], label=f"{target}", color=colors[target], marker='s')
        plt.fill_between(sub['layer'], sub['mean_corr'], sub['lexical_partial'], color=colors[target], alpha=0.1) # Area lost to semantics
    plt.title("Lexical Control: Unique Emotional Info beyond Words")
    plt.ylabel("Partial Correlation (Data, Emotion | Words)")
    plt.xlabel("Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_lexical_control.png")
    plt.close()

# PLOT 3: Complexity Analysis (GoEmotions Partial vs Total) - THE MISSING PLOT
plt.figure(figsize=(14, 8))
go_data = results_df[results_df['taxonomy'] == 'go_emotions']
plt.plot(go_data['layer'], go_data['mean_corr'], label='GoEmotions (Total)', color='tab:green', marker='o')
plt.plot(go_data['layer'], go_data['complex_partial'], label='GoEmotions (Partial | Ekman)', color='tab:red', linestyle='--', marker='s')
plt.fill_between(go_data['layer'], go_data['mean_corr'], go_data['complex_partial'], color='yellow', alpha=0.1, label='Redundancy with Ekman')

plt.title("Complexity Analysis: Does GoEmotions add value over Ekman?")
plt.ylabel("Correlation")
plt.xlabel("Layer")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_complexity_comparison.png")
plt.close()

# PLOT 4: Percentage Analysis (Dual Axis)
plot_complexity_percentage(results_df, f"{OUTPUT_DIR}/04_complexity_percentage.png")

print(f"\nSUCCESS. All 4 analysis plots saved in {OUTPUT_DIR}")