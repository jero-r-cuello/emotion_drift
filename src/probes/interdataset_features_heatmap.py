import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURACIÓN ---
LLM_USED = "Llama-2-7b-chat-hf" 
BASE_DIR = "/home/jcuello/emotion_drift"
MODELS_DIR_BASE = os.path.join(BASE_DIR, "models")

DATASET_A = "generated_prompts"
DATASET_B = "human_centric"

HEATMAPS_DIR = os.path.join(
    BASE_DIR, 
    "figures", 
    f"cross_dataset_features_{DATASET_A}_vs_{DATASET_B}_{LLM_USED}"
)

TAXONOMIES = ['ekman_basic_emotions', 'go_emotions', 'plutchik_wheel']

# Configurar dispositivo (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

os.makedirs(HEATMAPS_DIR, exist_ok=True)

def load_probe_weights(dataset_name, taxonomy, layer):
    filename = f"{dataset_name}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
    path = os.path.join(MODELS_DIR_BASE, filename)
    
    if not os.path.exists(path):
        return None, None
    
    try:
        pipeline = joblib.load(path)
        clf = pipeline.named_steps['logisticregression']
        weights = clf.coef_
        classes = clf.classes_
        return weights, classes
    except Exception as e:
        print(f"Error cargando {filename}: {e}")
        return None, None

def align_weights_and_classes(w_a, c_a, w_b, c_b):
    """
    Encuentra las clases comunes entre A y B, y reordena los pesos
    para que coincidan perfectamente. Descarta clases que no estén en ambos.
    """
    # 1. Encontrar intersección de clases y ordenarlas alfabéticamente
    common_classes = sorted(list(set(c_a) & set(c_b)))
    
    if len(common_classes) == 0:
        return None, None, None

    # 2. Encontrar los índices correspondientes en cada array original
    # np.where devuelve una tupla, tomamos el [0][0] para sacar el índice entero
    indices_a = [np.where(c_a == cls)[0][0] for cls in common_classes]
    indices_b = [np.where(c_b == cls)[0][0] for cls in common_classes]
    
    # 3. Filtrar y reordenar los pesos usando numpy fancy indexing
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
    
    # Ajustar tamaño si son muchas clases
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
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_path = os.path.join(HEATMAPS_DIR, f"heatmap_L{layer:02d}_{taxonomy}_cross_dataset.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# MAIN LOOP
# =============================================================================

MAX_LAYERS = 33 if "7b" in LLM_USED else 49 
if "Qwen" in LLM_USED: MAX_LAYERS = 49

print(f"Iniciando comparación {DATASET_A} vs {DATASET_B} (con alineación de clases)...")

for layer in tqdm(range(MAX_LAYERS), desc="Layers"):
    
    for taxonomy in TAXONOMIES:
        
        # 1. Cargar pesos
        w_a, c_a = load_probe_weights(DATASET_A, taxonomy, layer)
        w_b, c_b = load_probe_weights(DATASET_B, taxonomy, layer)
        
        if w_a is None or w_b is None:
            continue
            
        # 2. ALINEACIÓN DE CLASES (Nuevo paso crítico)
        # Esto soluciona si GoEmotions tiene 'neutral' en uno y en otro no, o distinto orden.
        w_a_new, w_b_new, common_classes = align_weights_and_classes(w_a, c_a, w_b, c_b)
        
        if w_a_new is None:
            print(f"Warning: No hay clases en común para {taxonomy} layer {layer}")
            continue
            
        # Debug opcional para ver si perdemos clases
        if len(common_classes) < len(c_a) or len(common_classes) < len(c_b):
            # Solo imprimimos esto una vez para no llenar la consola, por ejemplo en capa 20
            if layer == 20: 
                print(f"\n[Info] {taxonomy} Layer {layer}: Alineando {len(c_a)} vs {len(c_b)} clases -> {len(common_classes)} comunes.")

        # 3. Calcular Heatmap con los pesos alineados
        sim_matrix = get_cosine_similarity_matrix(w_a_new, w_b_new)
        
        # 4. Graficar usando las clases comunes
        plot_and_save_heatmap(sim_matrix, common_classes, taxonomy, layer)

print(f"\nProceso terminado. Heatmaps guardados en: {HEATMAPS_DIR}")