import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

LLM_USED = "Llama-2-7b-chat-hf" 
MODELS_DIR_BASE = "models"

DATASET_A = "generated_prompts"
DATASET_B = "human_centric"

HEATMAPS_DIR = os.path.join("figures", f"cross_dataset_features_{DATASET_A}_vs_{DATASET_B}_{LLM_USED}")

TAXONOMIES = ["ekman_basic_emotions", "go_emotions", "plutchik_wheel"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(HEATMAPS_DIR, exist_ok=True)

def load_probe_weights(dataset_name, taxonomy, layer):
    filename = f"{dataset_name}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
    path = os.path.join(MODELS_DIR_BASE, filename)
    
    if not os.path.exists(path):
        return None, None
    
    try:
        pipeline = joblib.load(path)
        clf = pipeline.named_steps["logisticregression"]
        weights = clf.coef_
        classes = clf.classes_
        return weights, classes
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def align_weights_and_classes(w_a, c_a, w_b, c_b):
    """
    Find the classes common between A and B, and reorder the weights
    so that they match perfectly. Discard classes that are not in both.
    """
    # Encontrar intersección de clases y ordenarlas alfabéticamente
    common_classes = sorted(list(set(c_a) & set(c_b)))
    
    if len(common_classes) == 0:
        return None, None, None

    indices_a = [np.where(c_a == cls)[0][0] for cls in common_classes]
    indices_b = [np.where(c_b == cls)[0][0] for cls in common_classes]
    
    w_a_aligned = w_a[indices_a]
    w_b_aligned = w_b[indices_b]
    
    return w_a_aligned, w_b_aligned, common_classes

def get_cosine_similarity_matrix(weights_a, weights_b):
    t_a = torch.tensor(weights_a, dtype=torch.float32, device=device)
    t_b = torch.tensor(weights_b, dtype=torch.float32, device=device)
    
    t_a_norm = t_a / t_a.norm(dim=1, keepdim=True)
    t_b_norm = t_b / t_b.norm(dim=1, keepdim=True)
    
    similarity_matrix = torch.mm(t_a_norm, t_b_norm.t())
    return similarity_matrix.cpu().numpy()

def plot_and_save_heatmap(sim_matrix, classes, taxonomy, layer):
    plt.figure(figsize=(12, 10))
    
    annot_kws_size = 7 if len(classes) > 20 else 10
    
    sns.heatmap(
        sim_matrix,
        xticklabels=classes, 
        yticklabels=classes, 
        cmap="RdBu_r", 
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_kws_size},
        square=True 
    )
    
    plt.title(f"Feature Similarity: {taxonomy}\n{DATASET_A} vs {DATASET_B} (Layer {layer})")
    plt.xlabel(f"Features from {DATASET_B}")
    plt.ylabel(f"Features from {DATASET_A}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    save_path = os.path.join(HEATMAPS_DIR, f"heatmap_L{layer:02d}_{taxonomy}_cross_dataset.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

MAX_LAYERS = 33 if "7b" in LLM_USED else 49 
if "Qwen" in LLM_USED: MAX_LAYERS = 49

print(f"Starting comparison {DATASET_A} vs {DATASET_B} (with class alignment)...")

for layer in tqdm(range(MAX_LAYERS), desc="Layers"):
    
    for taxonomy in TAXONOMIES:
        w_a, c_a = load_probe_weights(DATASET_A, taxonomy, layer)
        w_b, c_b = load_probe_weights(DATASET_B, taxonomy, layer)
        
        if w_a is None or w_b is None:
            continue
            
        w_a_new, w_b_new, common_classes = align_weights_and_classes(w_a, c_a, w_b, c_b)
        
        if w_a_new is None:
            print(f"Warning: No common classes for {taxonomy} layer {layer}")
            continue
            
        if len(common_classes) < len(c_a) or len(common_classes) < len(c_b):
            if layer == 20: 
                print(f"\n[Info] {taxonomy} Layer {layer}: Aligning {len(c_a)} vs {len(c_b)} classes -> {len(common_classes)} common.")

        sim_matrix = get_cosine_similarity_matrix(w_a_new, w_b_new)
        
        plot_and_save_heatmap(sim_matrix, common_classes, taxonomy, layer)

print(f"\nProcess completed. Heatmaps saved in: {HEATMAPS_DIR}")