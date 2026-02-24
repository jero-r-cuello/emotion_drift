import os
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURACIÓN ---
LLM_USED = "Llama-2-7b-chat-hf"
DATASET = "generated_prompts"
BASE_DIR = "/home/jcuello/emotion_drift"

# Directorio donde se guardaron los modelos (según tu script de entrenamiento)
# Estructura: models/dataset_split_testing/{DATASET}_{LLM}_{TAXONOMY}/...
MODELS_ROOT = os.path.join(BASE_DIR, "models", "dataset_split_testing", f"{DATASET}_{LLM_USED}")

# Directorio de salida para las figuras
HEATMAPS_DIR = os.path.join(BASE_DIR, "figures", f"split_consistency_heatmaps_{DATASET}_{LLM_USED}")

# Taxonomías a evaluar (puedes agregar más a la lista)
TAXONOMIES = ['ekman_basic_emotions'] 

# Configuración de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

os.makedirs(HEATMAPS_DIR, exist_ok=True)

def load_split_model(taxonomy, layer, split_name):
    """
    Carga el modelo de un split específico (A o B).
    Ruta esperada: MODELS_ROOT/taxonomy/filename
    """
    # Construir la ruta específica de la taxonomía
    model_dir = os.path.join(MODELS_ROOT, taxonomy)
    
    # Nombre del archivo según tu script de entrenamiento:
    # "{DATASET}_{LLM_USED}_{TARGET_TAXONOMY}_layer_{layer_num}_split_{split}.joblib"
    filename = f"{DATASET}_{LLM_USED}_{taxonomy}_layer_{layer}_split_{split_name}.joblib"
    path = os.path.join(model_dir, filename)
    
    if not os.path.exists(path):
        return None, None
    
    try:
        pipeline = joblib.load(path)
        # Extraer el clasificador (asumiendo LogisticRegression dentro de pipeline)
        # sklearn suele poner el nombre de la clase en minúsculas como paso del pipeline
        clf = pipeline.named_steps['logisticregression']
        
        weights = clf.coef_
        classes = clf.classes_
        return weights, classes
    except Exception as e:
        print(f"Error cargando {filename}: {e}")
        return None, None

def get_cosine_similarity_matrix(weights_a, weights_b):
    """Calcula similitud coseno en GPU."""
    t_a = torch.tensor(weights_a, dtype=torch.float32, device=device)
    t_b = torch.tensor(weights_b, dtype=torch.float32, device=device)
    
    # Normalización L2
    t_a_norm = t_a / t_a.norm(dim=1, keepdim=True)
    t_b_norm = t_b / t_b.norm(dim=1, keepdim=True)
    
    # Producto punto
    similarity_matrix = torch.mm(t_a_norm, t_b_norm.t())
    
    return similarity_matrix.cpu().numpy()

def plot_consistency_heatmap(sim_matrix, classes, taxonomy, layer):
    """
    Genera el heatmap comparando Split A vs Split B.
    Se asume que las clases son idénticas en orden (verificación hecha en el loop).
    """
    plt.figure(figsize=(10, 8))
    
    # Ajuste de fuente para taxonomías con muchas clases (GoEmotions)
    annot_size = 8 if len(classes) > 15 else 10
    
    sns.heatmap(
        sim_matrix,
        xticklabels=classes, # Split B
        yticklabels=classes, # Split A
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_size},
        square=True # Cuadrado porque comparamos lo mismo contra lo mismo
    )
    
    plt.title(f"Consistency Check: Split A vs Split B\n{taxonomy} - Layer {layer}")
    plt.xlabel("Split B Classes")
    plt.ylabel("Split A Classes")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Guardar
    # Crear subcarpeta por taxonomía para orden
    save_dir = os.path.join(HEATMAPS_DIR, taxonomy)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"consistency_L{layer:02d}_{taxonomy}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# MAIN LOOP
# =============================================================================

# Detección de capas (asumida por el modelo, igual que tu script anterior)
MAX_LAYERS = 33 if "7b" in LLM_USED else 49

print(f"Iniciando análisis de consistencia (Split A vs B) para {LLM_USED}...")
print(f"Modelos buscados en: {MODELS_ROOT}")

for taxonomy in TAXONOMIES:
    print(f"\nProcesando taxonomía: {taxonomy}")
    
    for layer in tqdm(range(MAX_LAYERS), desc=f"Layers ({taxonomy})"):
        
        # 1. Cargar Split A
        w_a, c_a = load_split_model(taxonomy, layer, "A")
        # 2. Cargar Split B
        w_b, c_b = load_split_model(taxonomy, layer, "B")
        
        if w_a is None or w_b is None:
            # Si falta alguno de los dos, no se puede comparar
            continue
            
        # 3. Verificación de seguridad de clases
        # Aunque el entrenamiento usó stratify, verificamos que las clases sean 
        # exactamente las mismas y en el mismo orden para que la diagonal tenga sentido.
        if not np.array_equal(c_a, c_b):
            # Caso borde: si las clases difieren (ej. una clase muy rara cayó solo en split A),
            # tendríamos que alinear las matrices.
            # Por ahora, simplemente avisamos y saltamos o hacemos un manejo básico.
            # (Si tus datos están bien estratificados, esto no debería pasar a menudo).
            
            # Intento de intersección rápida para no fallar
            common_classes = sorted(list(set(c_a) & set(c_b)))
            if len(common_classes) < 2:
                continue
            
            # Índices para reordenar y filtrar
            idx_a = [np.where(c_a == c)[0][0] for c in common_classes]
            idx_b = [np.where(c_b == c)[0][0] for c in common_classes]
            
            w_a = w_a[idx_a]
            w_b = w_b[idx_b]
            classes_to_plot = common_classes
        else:
            classes_to_plot = c_a

        # 4. Calcular Similitud
        sim_matrix = get_cosine_similarity_matrix(w_a, w_b)
        
        # 5. Graficar
        # La diagonal principal debería ser roja intensa (similitud ~1.0)
        plot_consistency_heatmap(sim_matrix, classes_to_plot, taxonomy, layer)

print(f"\nProceso terminado. Heatmaps guardados en: {HEATMAPS_DIR}")