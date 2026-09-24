import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # cwd-independent
from paper_style import (apply_paper_style, pretty, heatmap_figsize,
                         annot_size, save_paper_figure)

apply_paper_style()

LLM_USED = os.environ.get("PROBE_LLM", "Llama-2-7b-chat-hf")  # "Qwen2.5-14B-Instruct"
DATASET = os.environ.get("PROBE_DATASET", "generated_prompts")
MODELS_DIR_BASE = "models"
HEATMAPS_DIR = os.path.join("figures", f"probe_features_heatmaps_{DATASET}_{LLM_USED}")

# Taonomies to be compared
TAXONOMIES = ["ekman_basic_emotions", "go_emotions", "plutchik_wheel"]
MODEL_DIM = int(os.environ.get("PROBE_MODEL_DIM", "4096"))  # 5120 for Qwen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(HEATMAPS_DIR, exist_ok=True)

def load_probe_weights(taxonomy, layer):
    """
    Load the model, extract the Logistic Regression weights
    and their class labels.
    """
    filename = f"{DATASET}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
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
        print(f"Error cargando {filename}: {e}")
        return None, None

def get_cosine_similarity_matrix(weights_a, weights_b):
    """
    Calculates the cosine similarity matrix between two sets of vectors using GPU.
    Returns a numpy matrix [n_classes_a, n_classes_b].
    """
    t_a = torch.tensor(weights_a, dtype=torch.float32, device=device) # [N_a, Dim]
    t_b = torch.tensor(weights_b, dtype=torch.float32, device=device) # [N_b, Dim]
    
    # Normalizar vectores (L2 norm) para que su magnitud sea 1
    # keepdim=True es importante para poder dividir correctamente
    t_a_norm = t_a / t_a.norm(dim=1, keepdim=True)
    t_b_norm = t_b / t_b.norm(dim=1, keepdim=True)
    
    # [N_a, Dim] @ [Dim, N_b] -> [N_a, N_b]
    similarity_matrix = torch.mm(t_a_norm, t_b_norm.t())
    
    return similarity_matrix.cpu().numpy()

def plot_and_save_heatmap(sim_matrix, classes_a, classes_b, name_a, name_b, layer):
    """Generate and save the heatmap, sized for the printed page.

    Drawn at its on-page width so the declared point sizes survive into the PDF;
    no title (the caption carries it) and taxonomy names spelled out.
    """
    n_rows, n_cols = len(classes_a), len(classes_b)
    fig, ax = plt.subplots(figsize=heatmap_figsize(n_rows, n_cols),
                           layout="constrained")

    sns.heatmap(
        sim_matrix,
        xticklabels=classes_b,
        yticklabels=classes_a,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_size(n_rows, n_cols)},
        square=False,
        ax=ax,
        cbar_kws={"label": "cosine similarity", "shrink": 0.85},
    )

    ax.set_xlabel(f"{pretty(name_b)} categories")
    ax.set_ylabel(f"{pretty(name_a)} categories")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")

    save_path = os.path.join(HEATMAPS_DIR, f"heatmap_L{layer:02d}_{name_a}_vs_{name_b}")
    save_paper_figure(fig, save_path)
    plt.close(fig)


MAX_LAYERS = 33 if "7b" in LLM_USED else 49
if "Qwen" in LLM_USED: MAX_LAYERS = 49

print(f"Starting Heatmap generation for {LLM_USED}...")
pairs = list(combinations(TAXONOMIES, 2))

for layer in tqdm(range(MAX_LAYERS), desc="Layers"):
    loaded_models = {} # Just a cache
    
    valid_layer = True
    for tax in TAXONOMIES:
        w, c = load_probe_weights(tax, layer)
        
        if w is None:
            valid_layer = False
            break
        
        loaded_models[tax] = (w, c)
    
    if not valid_layer:
        continue
        
    for tax_a, tax_b in pairs:
        w_a, c_a = loaded_models[tax_a]
        w_b, c_b = loaded_models[tax_b]
        
        sim_matrix = get_cosine_similarity_matrix(w_a, w_b)
        
        plot_and_save_heatmap(sim_matrix, c_a, c_b, tax_a, tax_b, layer)

print(f"\nProcess completed. Heatmaps saved in: {HEATMAPS_DIR}.")