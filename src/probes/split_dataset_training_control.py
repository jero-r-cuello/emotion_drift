#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Configuración ---
LLM_USED = "Llama-2-7b-chat-hf"
MODEL_DIM = 4096
DATASET = "generated_prompts"
TARGET_TAXONOMY = 'ekman_basic_emotions' # Variable solicitada

# Path del archivo (Usando el que pasaste en el ejemplo)
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"

# Directorios de salida
BASE_DIR = "/home/jcuello/emotion_drift"
# Carpeta específica solicitada
PROBES_DIR = os.path.join(BASE_DIR, "models", "dataset_split_testing", f"{DATASET}_{LLM_USED}", TARGET_TAXONOMY)

os.makedirs(PROBES_DIR, exist_ok=True)

print(f"--- Configuración ---")
print(f"Dataset: {DATASET}")
print(f"Taxonomy: {TARGET_TAXONOMY}")
print(f"Output Dir: {PROBES_DIR}")

# --- Carga de Datos ---
if not os.path.exists(DATA_PATH): 
    raise FileNotFoundError(f'{DATA_PATH} not found')

print(f"Cargando datos desde {DATA_PATH}...")
nested_df_original = pd.read_pickle(DATA_PATH)
if nested_df_original.empty: 
    raise ValueError("Dataframe vacío.")
print("Datos cargados correctamente.")

# --- Preprocesamiento Inicial ---
nested_df = nested_df_original.copy()

# Verificar que la taxonomía existe
if TARGET_TAXONOMY not in nested_df.columns:
    raise ValueError(f"La taxonomía {TARGET_TAXONOMY} no existe en el dataframe.")

# Filtrar filas sin etiquetas válidas y obtener la primera etiqueta de la lista
mask = nested_df[TARGET_TAXONOMY].apply(lambda x: isinstance(x, list) and len(x) > 0)
nested_df = nested_df[mask].copy()
nested_df[TARGET_TAXONOMY] = nested_df[TARGET_TAXONOMY].str[0]

if nested_df.empty:
    raise ValueError("El dataframe se quedó vacío tras filtrar las etiquetas.")

labels_present = sorted(nested_df[TARGET_TAXONOMY].unique())
print(f"Clases encontradas ({len(labels_present)}): {labels_present}")

try:
    num_layers = len(nested_df['activations'].iloc[0])
    print(f"Número de capas detectadas: {num_layers}")
except:
    num_layers = 0
    print("Error detectando número de capas.")

# =========================================================================
# BUCLE DE CAPAS
# =========================================================================
for layer_num in tqdm(range(num_layers), desc=f"Entrenando Splits ({TARGET_TAXONOMY})"):
    
    # --- 1. Extracción de Activaciones ---
    try:
        X_list = []
        y_list = []
        
        # Iteramos para extraer la activación de la capa específica
        for act_row, label in zip(nested_df['activations'], nested_df[TARGET_TAXONOMY]):
            try:
                act = act_row.iloc[layer_num]['last_token_activation']
                if not isinstance(act, np.ndarray): continue
                if act.ndim > 1: act = act.squeeze()
                
                if act.shape == (MODEL_DIM,): 
                    X_list.append(act)
                    y_list.append(label)
            except: 
                continue

        if len(X_list) == 0: 
            print(f"[Layer {layer_num}] No activations found.")
            continue
            
        X = np.stack(X_list)
        y_real = np.array(y_list)

    except Exception as e:
        print(f"Error extrayendo datos en capa {layer_num}: {e}")
        continue

    # --- 2. Filtrado de Clases Escasas ---
    # Necesitamos al menos 2 ejemplos por clase para poder hacer un split 50/50 estratificado
    # (1 para el split A, 1 para el split B)
    class_counts = Counter(y_real)
    min_samples_required = 2 
    classes_to_drop = [cls for cls, count in class_counts.items() if count < min_samples_required]
    
    if classes_to_drop:
        mask_valid = ~np.isin(y_real, classes_to_drop)
        X = X[mask_valid]
        y_real = y_real[mask_valid]
    
    if len(X) == 0 or len(np.unique(y_real)) < 2:
        print(f"   [Layer {layer_num}] Skipping: Not enough data/classes after filtering.")
        continue

    # --- 3. Split 50/50 Estratificado ---
    try:
        # test_size=0.5 divide el dataset exactamente en dos mitades
        # stratify=y_real asegura que ambas mitades tengan la misma proporción de clases
        X_split_A, X_split_B, y_split_A, y_split_B = train_test_split(
            X, y_real, 
            test_size=0.5, 
            random_state=42, 
            stratify=y_real
        )
    except ValueError as e:
        print(f"   [Layer {layer_num}] Split error: {e}")
        continue

    # Definición del modelo base
    def get_probe_model():
        return make_pipeline(
            StandardScaler(), 
            LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, solver='lbfgs', n_jobs=-1)
        )

    # --- 4. Entrenar y Guardar Split A ---
    try:
        clf_A = get_probe_model()
        clf_A.fit(X_split_A, y_split_A)
        
        filename_A = f"{DATASET}_{LLM_USED}_{TARGET_TAXONOMY}_layer_{layer_num}_split_A.joblib"
        path_A = os.path.join(PROBES_DIR, filename_A)
        joblib.dump(clf_A, path_A)
    except Exception as e:
        print(f"Error entrenando/guardando Split A capa {layer_num}: {e}")

    # --- 5. Entrenar y Guardar Split B ---
    try:
        clf_B = get_probe_model()
        clf_B.fit(X_split_B, y_split_B)
        
        filename_B = f"{DATASET}_{LLM_USED}_{TARGET_TAXONOMY}_layer_{layer_num}_split_B.joblib"
        path_B = os.path.join(PROBES_DIR, filename_B)
        joblib.dump(clf_B, path_B)
    except Exception as e:
        print(f"Error entrenando/guardando Split B capa {layer_num}: {e}")

print("\nProceso finalizado. Modelos guardados en:", PROBES_DIR)
#%%