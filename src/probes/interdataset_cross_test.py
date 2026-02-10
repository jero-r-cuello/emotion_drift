import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.utils import resample

# --- CONFIGURACIÓN "ULTRA-ROBUSTA" ---
LLM_USED = "Llama-2-7b-chat-hf"
BASE_DIR = "/home/jcuello/emotion_drift"
MODELS_DIR_BASE = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", f"cross_testing_performance{LLM_USED}")

PATH_HUMAN = "/home/jcuello/emotion_drift/data/03_activations/MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
PATH_GEN = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"

NAME_GEN = "generated_prompts"
NAME_HUMAN = "human_centric"

TAXONOMIES = ['ekman_basic_emotions', 'plutchik_wheel']
LAYERS = range(33)
ACTIVATION_COL = 'last_token_activation'

# CONFIGURACIÓN ESTADÍSTICA SUPERIOR
BOOTSTRAP_ITERATIONS = 10000  # Máxima robustez (tardará más, pero tienes potencia)
CONFIDENCE_LEVEL = 0.95       # Estándar científico
RANDOM_SEED = 42              # Reproducibilidad

os.makedirs(FIGURES_DIR, exist_ok=True)

# --- CARGA DE DATOS ---
print("--- CARGANDO DATASETS ---")
df_human = pd.read_pickle(PATH_HUMAN)
df_gen = pd.read_pickle(PATH_GEN)
print(f"Datos cargados.")

def get_data_for_layer_strict(df, layer, taxonomy):
    """Extrae X e y (primer label)."""
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

def stratified_bootstrap_metric(y_true, y_pred, n_iterations=10000, ci=0.95, seed=42):
    """
    Realiza Bootstrapping ESTRATIFICADO.
    Mantiene la proporción de clases en cada sub-muestra, evitando
    inestabilidad en el F1-Score para clases desbalanceadas.
    """
    stats = []
    
    # Pre-calcular alpha para percentiles
    alpha = (1.0 - ci) / 2.0
    
    # Fijar semilla localmente para este proceso
    rng = np.random.RandomState(seed)
    
    for i in range(n_iterations):
        # Muestreo con reemplazo ESTRATIFICADO (stratify=y_true)
        # Usamos una semilla diferente por iteración derivada del estado global
        iter_seed = rng.randint(0, 2**32 - 1)
        
        y_t_boot, y_p_boot = resample(
            y_true, y_pred, 
            replace=True, 
            stratify=y_true,  # <--- LA MEJORA CLAVE
            random_state=iter_seed
        )
        
        # Zero_division=0 es importante en bootstrap si una clase desaparece (aunque stratify lo minimiza)
        score = f1_score(y_t_boot, y_p_boot, average='macro', zero_division=0)
        stats.append(score)
    
    lower = np.percentile(stats, alpha * 100)
    upper = np.percentile(stats, (1.0 - alpha) * 100)
    mean_score = np.mean(stats)
    
    return mean_score, lower, upper

# --- MAIN LOOP ---

results = []

print(f"\n--- INICIANDO CROSS-TESTING ULTRA-ROBUSTO ({BOOTSTRAP_ITERATIONS} iters, Stratified) ---")

for taxonomy in TAXONOMIES:
    print(f"\nEvaluando Taxonomía: {taxonomy}")
    
    for layer in tqdm(LAYERS, desc=f"Layers ({taxonomy})"):
        
        X_human, y_human = get_data_for_layer_strict(df_human, layer, taxonomy)
        X_gen, y_gen = get_data_for_layer_strict(df_gen, layer, taxonomy)
        
        if X_human is None or X_gen is None:
            continue
            
        # Comprobación de seguridad: Stratified requiere al menos 2 ejemplos por clase
        # Si una clase tiene 1 solo ejemplo, 'stratify' fallará.
        # En ese caso, hacemos fallback a bootstrap simple.
        try:
            unique, counts = np.unique(y_human, return_counts=True)
            stratify_possible_human = np.all(counts >= 2)
            
            unique_gen, counts_gen = np.unique(y_gen, return_counts=True)
            stratify_possible_gen = np.all(counts_gen >= 2)
        except:
            stratify_possible_human = False
            stratify_possible_gen = False

        # ---------------------------------------------------------
        # Test A: Gen -> Human
        # ---------------------------------------------------------
        model_gen = load_model(NAME_GEN, taxonomy, layer)
        if model_gen is not None:
            y_pred_gh = model_gen.predict(X_human)
            
            # Decisión inteligente de estrategia
            if stratify_possible_human:
                mean, low, high = stratified_bootstrap_metric(y_human, y_pred_gh, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, RANDOM_SEED)
            else:
                # Fallback silencioso si hay clases con 1 solo ejemplo
                y_t_boot, y_p_boot = resample(y_human, y_pred_gh, replace=True, random_state=RANDOM_SEED)
                # (Recalcular con bootstrap simple...) - Simplificación: Usamos la función normal si falla
                # Para mantener el código limpio, asumimos que stratified funciona o sklearn lanza error
                # Aquí re-implemento bootstrap simple inline por si acaso:
                stats = [f1_score(*resample(y_human, y_pred_gh, replace=True), average='macro', zero_division=0) for _ in range(BOOTSTRAP_ITERATIONS)]
                alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
                mean, low, high = np.mean(stats), np.percentile(stats, alpha*100), np.percentile(stats, (1-alpha)*100)

            results.append({
                'taxonomy': taxonomy,
                'layer': layer,
                'train_source': 'Generated',
                'test_source': 'Human',
                'f1_mean': mean,
                'f1_lower': low,
                'f1_upper': high
            })

        # ---------------------------------------------------------
        # Test B: Human -> Gen
        # ---------------------------------------------------------
        model_human = load_model(NAME_HUMAN, taxonomy, layer)
        if model_human is not None:
            y_pred_hg = model_human.predict(X_gen)
            
            if stratify_possible_gen:
                mean, low, high = stratified_bootstrap_metric(y_gen, y_pred_hg, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, RANDOM_SEED)
            else:
                stats = [f1_score(*resample(y_gen, y_pred_hg, replace=True), average='macro', zero_division=0) for _ in range(BOOTSTRAP_ITERATIONS)]
                alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
                mean, low, high = np.mean(stats), np.percentile(stats, alpha*100), np.percentile(stats, (1-alpha)*100)
            
            results.append({
                'taxonomy': taxonomy,
                'layer': layer,
                'train_source': 'Human',
                'test_source': 'Generated',
                'f1_mean': mean,
                'f1_lower': low,
                'f1_upper': high
            })

# --- VISUALIZACIÓN ---

if not results:
    exit()

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(FIGURES_DIR, "cross_test_bootstrap_results.csv"), index=False)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

for tax in TAXONOMIES:
    subset = results_df[results_df['taxonomy'] == tax]
    if subset.empty: continue
        
    plt.figure(figsize=(12, 7))
    
    # SERIE 1: Gen -> Human
    data_gh = subset[subset['train_source'] == 'Generated'].sort_values('layer')
    
    plt.plot(data_gh['layer'], data_gh['f1_mean'], 
             label="Train: Generated $\\to$ Test: Human", 
             color='#1f77b4', linewidth=2)
    
    plt.fill_between(data_gh['layer'], data_gh['f1_lower'], data_gh['f1_upper'], 
                     color='#1f77b4', alpha=0.25)

    # SERIE 2: Human -> Gen
    data_hg = subset[subset['train_source'] == 'Human'].sort_values('layer')
    
    plt.plot(data_hg['layer'], data_hg['f1_mean'], 
             label="Train: Human $\\to$ Test: Generated", 
             color='#ff7f0e', linewidth=2, linestyle='--')
    
    plt.fill_between(data_hg['layer'], data_hg['f1_lower'], data_hg['f1_upper'], 
                     color='#ff7f0e', alpha=0.25)
    
    plt.title(f"Cross-Dataset Robustness (Stratified Bootstrap 10k)\nTaxonomy: {tax} | Metric: Macro F1", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Macro F1-Score (with 95% CI)", fontsize=12)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(FIGURES_DIR, f"cross_test_{tax}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"\n¡Análisis Ultra-Robusto Completado! Gráficos guardados en: {FIGURES_DIR}")