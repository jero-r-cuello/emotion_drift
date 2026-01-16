import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations

# --- CONFIGURACIÓN (Debe coincidir con tu script de entrenamiento) ---
LLM_USED = "Qwen2.5-14B-Instruct" # "Llama2-7b-chat-hf"
DATASET = "generated_prompts"
BASE_DIR = "/home/jcuello/emotion_drift"
MODELS_DIR_BASE = os.path.join(BASE_DIR, "models")
HEATMAPS_DIR = os.path.join(BASE_DIR, "figures", f"probe_features_heatmaps_{DATASET}_{LLM_USED}")

# Las taxonomías que compararemos
TAXONOMIES = ['ekman_basic_emotions', 'go_emotions', 'plutchik_wheel']
MODEL_DIM = 5120 # 4096 #

# Configurar dispositivo (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

os.makedirs(HEATMAPS_DIR, exist_ok=True)

def load_probe_weights(taxonomy, layer):
    """
    Carga el modelo, extrae los pesos de la Regresión Logística
    y sus etiquetas de clase.
    """
    filename = f"{DATASET}_{LLM_USED}_{taxonomy}_layer_{layer}.joblib"
    path = os.path.join(MODELS_DIR_BASE, filename)
    
    if not os.path.exists(path):
        return None, None
    
    try:
        # Cargamos el pipeline
        pipeline = joblib.load(path)
        
        # Accedemos al paso de regresión logística (asumiendo que es el segundo paso o se llama 'logisticregression')
        # Tu pipeline es: StandardScaler -> LogisticRegression
        # Normalmente sklearn pone nombres en minúsculas
        clf = pipeline.named_steps['logisticregression']
        
        # Extraer pesos (Shape: [n_classes, hidden_size])
        weights = clf.coef_
        classes = clf.classes_
        
        return weights, classes
    except Exception as e:
        print(f"Error cargando {filename}: {e}")
        return None, None

def get_cosine_similarity_matrix(weights_a, weights_b):
    """
    Calcula la matriz de similitud de coseno entre dos conjuntos de vectores usando GPU.
    Retorna una matriz numpy [n_classes_a, n_classes_b].
    """
    # 1. Convertir a Tensores y mover a GPU
    t_a = torch.tensor(weights_a, dtype=torch.float32, device=device) # [N_a, Dim]
    t_b = torch.tensor(weights_b, dtype=torch.float32, device=device) # [N_b, Dim]
    
    # 2. Normalizar vectores (L2 norm) para que su magnitud sea 1
    # keepdim=True es importante para poder dividir correctamente
    t_a_norm = t_a / t_a.norm(dim=1, keepdim=True)
    t_b_norm = t_b / t_b.norm(dim=1, keepdim=True)
    
    # 3. Producto punto (Matmul)
    # [N_a, Dim] @ [Dim, N_b] -> [N_a, N_b]
    similarity_matrix = torch.mm(t_a_norm, t_b_norm.t())
    
    # 4. Devolver a CPU como numpy
    return similarity_matrix.cpu().numpy()

def plot_and_save_heatmap(sim_matrix, classes_a, classes_b, name_a, name_b, layer):
    """Genera y guarda el gráfico."""
    plt.figure(figsize=(12, 10))
    
    # Ajustar tamaño de fuente dinámicamente si hay muchas clases (como GoEmotions)
    annot_kws_size = 8 if (len(classes_a) > 20 or len(classes_b) > 20) else 10
    
    sns.heatmap(
        sim_matrix,
        xticklabels=classes_b,
        yticklabels=classes_a,
        cmap="RdBu_r", # Rojo = Positivo, Azul = Negativo
        center=0,
        vmin=-1, vmax=1,
        annot=True, # Poner números
        fmt=".2f",
        annot_kws={"size": annot_kws_size},
        square=False
    )
    
    plt.title(f"Cosine Similarity: {name_a} vs {name_b} (Layer {layer})")
    plt.xlabel(f"{name_b} Classes")
    plt.ylabel(f"{name_a} Classes")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_path = os.path.join(HEATMAPS_DIR, f"heatmap_L{layer:02d}_{name_a}_vs_{name_b}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# MAIN LOOP
# =============================================================================

# Asumimos un número máximo de capas basado en Llama/Qwen (o detectar dinámicamente)
MAX_LAYERS = 33 if "7b" in LLM_USED else 49 # Ajusta esto según necesites o haz un loop con try
if "Qwen" in LLM_USED: MAX_LAYERS = 49 # Ejemplo aproximado

print(f"Iniciando generación de Heatmaps para {LLM_USED}...")

# Generar pares de comparaciones (Ekman vs Go, Ekman vs Plutchik, Go vs Plutchik)
pairs = list(combinations(TAXONOMIES, 2))

for layer in tqdm(range(MAX_LAYERS), desc="Layers"):
    
    # Cache para no cargar el mismo modelo muchas veces en el mismo loop
    loaded_models = {} 
    
    # Cargar todos los modelos de esta capa primero
    valid_layer = True
    for tax in TAXONOMIES:
        w, c = load_probe_weights(tax, layer)
        if w is None:
            # Si falta algun modelo (ej. capa 0 a veces falla o no se entrenó), saltamos la capa
            valid_layer = False
            break
        loaded_models[tax] = (w, c)
    
    if not valid_layer:
        continue
        
    # Comparar pares
    for tax_a, tax_b in pairs:
        w_a, c_a = loaded_models[tax_a]
        w_b, c_b = loaded_models[tax_b]
        
        # Calcular Heatmap en GPU
        sim_matrix = get_cosine_similarity_matrix(w_a, w_b)
        
        # Graficar
        plot_and_save_heatmap(sim_matrix, c_a, c_b, tax_a, tax_b, layer)

print(f"\nProceso terminado. Heatmaps guardados en: {HEATMAPS_DIR}")