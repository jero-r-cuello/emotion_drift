#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import gc # Garbage Collector para gestión de memoria

# --- Configuración de Datos ---
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
TAXONOMY_TARGET = 'ekman_basic_emotions' 
BASE_DIR = "/home/jcuello/emotion_drift"
# Actualizamos el nombre del directorio para reflejar la nueva metodología
PLOTS_DIR = os.path.join(BASE_DIR, "figures", f"tsne_centered_evolution_{TAXONOMY_TARGET}")

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

# Mapa de colores fijo
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

# --- 3. Bucle Principal ---
print(f"\nGenerando plots (Neutral Centering + L2 Norm + PCA + t-SNE) para {num_layers} capas...")

for layer_idx in tqdm(range(num_layers), desc="Processing Layers"):
    
    # --- A. Extracción ---
    X_list = []
    y_list = []
    
    for act_row, label in zip(df['activations'], df['label_str']):
        try:
            # Manejo robusto de extracción (por si es dict o series)
            if isinstance(act_row, (pd.Series, pd.DataFrame)):
                item = act_row.iloc[layer_idx]
            else:
                item = act_row[layer_idx]
            
            # Obtener array
            if isinstance(item, dict):
                act = item.get('last_token_activation')
            elif hasattr(item, 'last_token_activation'):
                act = item['last_token_activation']
            else:
                act = item 

            if not isinstance(act, np.ndarray): continue
            if act.ndim > 1: act = act.squeeze()
            
            # Chequeo de dimensión (Llama-2 = 4096)
            if act.shape[0] == 4096:
                X_list.append(act)
                y_list.append(label)
        except Exception:
            continue
            
    if not X_list:
        print(f"Layer {layer_idx}: No valid data found.")
        continue

    X = np.stack(X_list)
    y = np.array(y_list)

    # --- B. Pre-Procesamiento "Perspectiva Correcta" ---
    
    # 1. Limpieza de Seguridad (Evita el Crash en Capa 0)
    # Reemplaza NaNs o Infinitos con 0 antes de cualquier cálculo
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Centrado Relativo al NEUTRO (Eliminar ruido sintáctico)
    # Buscamos los índices que son 'Neutral'
    neutral_indices = np.where(y == 'Neutral')[0]
    
    if len(neutral_indices) > 0:
        # Calculamos el vector promedio SOLO de los neutros
        neutral_mean = np.mean(X[neutral_indices], axis=0)
        # Restamos ese promedio a TODOS los datos
        # Resultado: Vectores de "diferencia pura" respecto a la neutralidad
        X_processed = X - neutral_mean
    else:
        # Fallback si no hay neutros en el batch
        X_processed = X - X.mean(axis=0)
        print(f"Warning Layer {layer_idx}: No 'Neutral' samples. Using global mean.")

    # 3. Filtrado de Neuronas Muertas (Varianza 0)
    # Si después de restar, hay columnas que son todo 0, PCA fallará
    selector = (X_processed.var(axis=0) > 1e-6)
    if selector.sum() < 2:
        print(f"Layer {layer_idx}: Not enough variance.")
        continue
    X_processed = X_processed[:, selector]

    # --- C. Pipeline de Reducción ---
    
    # Definimos el Pipeline Específico para esta capa
    # Normalizer: Proyecta a la esfera unitaria (mira ángulos, no magnitudes)
    # StandardScaler: Normaliza la varianza de las dimensiones resultantes
    # PCA: Reduce ruido
    # t-SNE: Visualiza
    
    n_comps = min(X_processed.shape[0], X_processed.shape[1], 50)
    
    pipeline = Pipeline([
        ('norm', Normalizer()), 
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comps, random_state=42)),
        ('tsne', TSNE(
            n_components=2, 
            perplexity=min(30, len(X_processed)-1), # Ajuste dinámico
            random_state=42, 
            init='pca', 
            learning_rate='auto',
            n_jobs=4 # FIJO en 4 para evitar Kernel Crash con -1
        ))
    ])

    try:
        X_embedded = pipeline.fit_transform(X_processed)
        
        # --- D. Plotting ---
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
            s=80, # Puntos un poco más grandes
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )
        
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        
        plt.title(f"Centered t-SNE (Neutral Mean Subtracted)\nLayer {layer_idx} | {TAXONOMY_TARGET}", fontsize=15)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        
        # Guardar
        filename = f"centered_tsne_layer_{layer_idx:02d}.png"
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close() # Importante cerrar para liberar memoria de matplotlib

    except Exception as e:
        print(f"Error in Layer {layer_idx}: {e}")

    # --- Gestión de Memoria ---
    # Limpiamos variables grandes explícitamente
    del X, X_processed, X_embedded, plot_df, X_list, y_list
    gc.collect()

print(f"\nFinalizado. Imágenes guardadas en: {PLOTS_DIR}")
# %%