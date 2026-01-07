#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap  # pip install umap-learn

# --- Configuración Estética ---
sns.set_theme(style="whitegrid")
# Aumentar tamaño de fuentes para legibilidad
plt.rcParams.update({'font.size': 12})

# --- Configuración ---
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
TAXONOMY_TARGET = 'plutchik_wheel' # La columna que usaremos para pintar
BASE_DIR = "/home/jcuello/emotion_drift"
PLOTS_DIR = os.path.join(BASE_DIR, "figures", "umap_evolution_cosine")

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
# Asumimos que es una lista y tomamos el primer elemento
df['label_str'] = df[TAXONOMY_TARGET].str[0]

# --- 2. Preparación Global (Colores y Dimensiones) ---
unique_labels = sorted(df['label_str'].unique())
print(f"Etiquetas encontradas ({len(unique_labels)}): {unique_labels}")

# Crear una paleta de colores consistente para todas las capas
palette = sns.color_palette("bright", n_colors=len(unique_labels))
color_map = dict(zip(unique_labels, palette))

# Detectar número de capas inspeccionando la primera fila
# Asumiendo que 'activations' es una serie de objetos indexables por capa
first_activations = df['activations'].iloc[0]
try:
    # Si es una lista o DF, intentamos ver el largo
    num_layers = len(first_activations)
    print(f"Detectadas {num_layers} capas.")
except:
    print("No se pudo detectar el número de capas automáticamente. Usando default 33.")
    num_layers = 33

# --- 3. Bucle por Capas ---
print(f"\nIniciando generación de UMAP (Metric=Cosine, No PCA) para {num_layers} capas...")

for layer_idx in tqdm(range(num_layers), desc="Processing Layers"):
    
    # --- A. Extracción de Activaciones ---
    X_list = []
    y_list = []
    
    # Iteramos fila por fila para extraer la capa específica
    # Esto es necesario porque la estructura está anidada
    for act_row, label in zip(df['activations'], df['label_str']):
        try:
            # Lógica de extracción (ajustar si la estructura interna varía)
            # En tu script anterior usabas .iloc[layer_num] sobre act_row
            act = act_row.iloc[layer_idx]['last_token_activation']
            
            if not isinstance(act, np.ndarray): continue
            if act.ndim > 1: act = act.squeeze()
            
            # Validación de forma (Llama-2 = 4096)
            if act.shape[0] == 4096:
                X_list.append(act)
                y_list.append(label)
        except Exception:
            continue
            
    if not X_list:
        print(f"Skipping layer {layer_idx} (no valid data extracted)")
        continue

    X = np.stack(X_list)
    y = np.array(y_list)

    # --- B. UMAP (Sin PCA, Métrica Coseno) ---
    # n_neighbors: Balance entre local (bajo) y global (alto). 15-30 es estándar.
    # min_dist: Qué tan apretados están los puntos. 0.1 permite clusters densos.
    reducer = umap.UMAP(
        n_neighbors=30,        # Aumentado ligeramente para preservar más estructura global
        min_dist=0.1,
        n_components=2,
        metric='cosine',       
        random_state=42,       # Para reproducibilidad (relativa en UMAP)
        n_jobs=-1              
    )
    
    embedding = reducer.fit_transform(X)
    
    # Crear DF temporal para plot
    plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
    plot_df['label'] = y

    # --- C. Plotting ---
    plt.figure(figsize=(12, 10))
    
    ax = sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='label',
        palette=color_map,     # Usar el mapa de colores fijo
        style='label',         # Formas distintas para ayudar a la diferenciación
        s=60,                  # Tamaño del punto
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Mover la leyenda afuera
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.title(f"UMAP Projection (Cosine) - {TAXONOMY_TARGET}\nLayer {layer_idx} (Dim: {X.shape[1]})", fontsize=15)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    
    # Guardar
    filename = f"{TAXONOMY_TARGET}_umap_layer_{layer_idx:02d}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300) # dpi 150 es buen balance calidad/peso
    plt.show()
    plt.close() # Cerrar figura para liberar memoria RAM

print(f"\nProceso finalizado. Imágenes guardadas en:\n{PLOTS_DIR}")
# %%
