# umap_centered_layer_23_neighbors_sweep
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap

# --- Configuración Estética ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# --- Configuración ---
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
TAXONOMY_TARGET = 'ekman_basic_emotions'
BASE_DIR = "/home/jcuello/emotion_drift"

# DEFINICIÓN DEL EXPERIMENTO
TARGET_LAYER = 23
NEIGHBORS_VALUES = [1000, 2000, 5000]#[15, 30, 50, 100, 200, 500]

# Carpeta de salida específica para este experimento
PLOTS_DIR = os.path.join(BASE_DIR, "figures", f"umap_centered_layer_{TARGET_LAYER}_neighbors_sweep")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. Carga y Limpieza de Datos ---
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f'{DATA_PATH} not found')
print(f"Cargando datos desde {DATA_PATH}...")
nested_df_original = pd.read_pickle(DATA_PATH)

# Filtrar filas vacías o sin label
df = nested_df_original.copy()
mask = df[TAXONOMY_TARGET].apply(lambda x: isinstance(x, list) and len(x) > 0)
df = df[mask].copy()

# Extraer labels (flatten)
df['label_str'] = df[TAXONOMY_TARGET].str[0]

# --- 2. Preparación Global ---
unique_labels = sorted(df['label_str'].unique())
print(f"Etiquetas encontradas ({len(unique_labels)}): {unique_labels}")

# Crear paleta de colores
palette = sns.color_palette("bright", n_colors=len(unique_labels))
color_map = dict(zip(unique_labels, palette))

# --- 3. Extracción de Activaciones (SOLO LAYER 23) ---
print(f"\nExtrayendo activaciones para la Layer {TARGET_LAYER}...")
X_list = []
y_list = []

for act_row, label in zip(df['activations'], df['label_str']):
    try:
        # Lógica de extracción para la capa específica
        if isinstance(act_row, (pd.Series, pd.DataFrame)):
            act = act_row.iloc[TARGET_LAYER]['last_token_activation']
        else:
            act = act_row[TARGET_LAYER]['last_token_activation']
        
        if not isinstance(act, np.ndarray): continue
        if act.ndim > 1: act = act.squeeze()
        
        # Validación de forma (Llama-2 = 4096)
        if act.shape[0] == 4096:
            X_list.append(act)
            y_list.append(label)
    except Exception:
        continue

if not X_list:
    raise ValueError(f"No valid data extracted for layer {TARGET_LAYER}")

X = np.stack(X_list)
y = np.array(y_list)

print(f"Data shape: {X.shape} (Points: {X.shape[0]}, Dim: {X.shape[1]})")

# --- 4. Lógica de Centrado Emocional (Se hace UNA sola vez) ---
print("Aplicando centrado (Neutral Mean Subtraction)...")

# Limpieza de seguridad
if np.isnan(X).any() or np.isinf(X).any():
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Centrado relativo al NEUTRO
neutral_indices = np.where(y == 'Neutral')[0]

if len(neutral_indices) > 0:
    neutral_mean = np.mean(X[neutral_indices], axis=0)
    X_processed = X - neutral_mean
    print(f"Centrado realizado usando {len(neutral_indices)} muestras 'Neutral'.")
else:
    X_processed = X - X.mean(axis=0)
    print("WARNING: No se encontraron muestras 'Neutral'. Usando media global.")

# --- 5. Bucle por n_neighbors ---
print(f"\nIniciando generación de UMAPs variando n_neighbors: {NEIGHBORS_VALUES}")

for n_neighbors in tqdm(NEIGHBORS_VALUES, desc="Testing Neighbors"):
    
    # --- A. UMAP ---
    # Usamos los datos YA procesados y centrados (X_processed)
    # n_jobs=-1 es crucial aquí con +21k puntos
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,        
        min_dist=0.1,
        n_components=2,
        metric='cosine',       
        random_state=42,       
        n_jobs=-1              
    )
    
    embedding = reducer.fit_transform(X_processed)
    
    # Crear DF temporal para plot
    plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
    plot_df['label'] = y

    # --- B. Plotting ---
    plt.figure(figsize=(12, 10))
    
    ax = sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='label',
        palette=color_map,     
        style='label',         
        s=60,                  
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Mover la leyenda afuera
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    # Título dinámico
    plt.title(
        f"Centered UMAP - Layer {TARGET_LAYER} - Neighbors: {n_neighbors}\n"
        f"Metric: Cosine | Points: {X.shape[0]}", 
        fontsize=15
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    
    # Guardar con el n_neighbors en el nombre
    filename = f"{TAXONOMY_TARGET}_L{TARGET_LAYER}_neighbors_{n_neighbors:03d}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300) 
    plt.show()
    plt.close()

print(f"\nProceso finalizado. Imágenes guardadas en:\n{PLOTS_DIR}")
# %%
