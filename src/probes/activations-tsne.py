#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Configuración de Datos ---
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
TAXONOMY_TARGET = 'ekman_basic_emotions' # 'go_emotions', 'plutchik_wheel'
BASE_DIR = "/home/jcuello/emotion_drift"
PLOTS_DIR = os.path.join(BASE_DIR, "figures", f"tsne_pca_evolution_{TAXONOMY_TARGET}")

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Configuración Estética ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.max_open_warning': 0})

# --- 1. Carga y Limpieza ---
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f'{DATA_PATH} not found')
print(f"Cargando datos desde {DATA_PATH}...")
nested_df_original = pd.read_pickle(DATA_PATH)

# Filtrar datos válidos
df = nested_df_original.copy()
mask = df[TAXONOMY_TARGET].apply(lambda x: isinstance(x, list) and len(x) > 0)
df = df[mask].copy()

# Aplanar labels
df['label_str'] = df[TAXONOMY_TARGET].str[0]

# --- 2. Preparación Global ---
unique_labels = sorted(df['label_str'].unique())
print(f"Etiquetas: {unique_labels}")

# Mapa de colores fijo para consistencia entre capas
palette = sns.color_palette("bright", n_colors=len(unique_labels))
color_map = dict(zip(unique_labels, palette))

# Detectar número de capas
try:
    first_activations = df['activations'].iloc[0]
    num_layers = len(first_activations)
    print(f"Detectadas {num_layers} capas.")
except:
    num_layers = 33
    print(f"No se detectaron capas, asumiendo {num_layers}.")

# --- 3. Pipeline de Reducción ---
# Definimos la estructura del pipeline, pero se entrena en cada vuelta del loop
def get_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50, random_state=42)),
        ('tsne', TSNE(
            n_components=2, 
            perplexity=30, 
            random_state=42, 
            init='pca', 
            learning_rate='auto',
            n_jobs=-1
        ))
    ])

# --- 4. Bucle Principal ---
print(f"\nGenerando plots PCA(50)->t-SNE para {num_layers} capas...")

for layer_idx in tqdm(range(num_layers), desc="Processing Layers"):
    # Layer 0 crashes the kernel for some reason
    if layer_idx == 0:
        continue

    # --- A. Extracción ---
    X_list = []
    y_list = []
    
    for act_row, label in zip(df['activations'], df['label_str']):
        try:
            # Extraer capa específica
            act = act_row.iloc[layer_idx]['last_token_activation']
            
            if not isinstance(act, np.ndarray): continue
            if act.ndim > 1: act = act.squeeze()
            
            if act.shape[0] == 4096:
                X_list.append(act)
                y_list.append(label)
        except:
            continue
            
    if not X_list:
        continue

    X = np.stack(X_list)
    y = np.array(y_list)

    # --- B. Transformación (PCA -> t-SNE) ---
    pipeline = get_pipeline()
    X_embedded = pipeline.fit_transform(X)
    
    # --- C. Plotting ---
    plot_df = pd.DataFrame(X_embedded, columns=['x', 'y'])
    plot_df['label'] = y

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
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    plt.title(f"t-SNE (via PCA-50) - {TAXONOMY_TARGET}\nLayer {layer_idx}", fontsize=15)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    
    # Guardar
    filename = f"{TAXONOMY_TARGET}_tsne_pca_layer_{layer_idx:02d}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

print(f"\nFinalizado. Imágenes en: {PLOTS_DIR}")
# %%
