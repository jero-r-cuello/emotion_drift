import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURACIÓN =================
BASE_DIR = "/home/jcuello/emotion_drift"
RUN_TO_LOAD = "Qwen2.5-14B-Instruct_20251220_225401"#"Llama-2-7b-chat-hf_20251014_203636"
DATASET_USED = "generated_prompts"
ANNOTATIONS_FILE = "qwen_annotated_results.jsonl"

INPUT_PKL_PATH = os.path.join(BASE_DIR, "data", "03_activations", f"{DATASET_USED}_{RUN_TO_LOAD}.pkl")
ANNOTATIONS_JSONL_PATH = os.path.join(BASE_DIR, "data", "04_annotated",ANNOTATIONS_FILE) 
OUTPUT_PKL_PATH = os.path.join(BASE_DIR, "data", "03_activations", f"{DATASET_USED}_{RUN_TO_LOAD}_FINAL.pkl")

# Carpeta para guardar los gráficos
PLOTS_DIR = os.path.join(BASE_DIR, "figures", f"annotations_count_plots_{RUN_TO_LOAD}")

TARGET_COLUMNS = ["ekman_basic_emotions", "go_emotions", "plutchik_wheel"]

# ================= FUNCIONES DE PARSEO =================

def extract_emotions_from_entry(record):
    try:
        output_list = record.get('response', {}).get('body', {}).get('output', [])
        message_content = None
        for item in output_list:
            if item.get('type') == 'message':
                content_list = item.get('content', [])
                if content_list and isinstance(content_list, list):
                    message_content = content_list[0].get('text')
                break
        
        if not message_content:
            return None

        clean_json_str = message_content.replace("```json", "").replace("```", "").strip()
        inner_data = json.loads(clean_json_str)
        return inner_data.get('emotions', [])

    except (json.JSONDecodeError, AttributeError, IndexError):
        return None

def load_annotations_map(jsonl_path):
    print(f"--> Cargando y procesando anotaciones desde: {jsonl_path}")
    annotation_map = {}
    id_pattern = re.compile(r"response-(\d+)-(.+)")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"No se encontró el archivo: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Indexando JSONL"):
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
                custom_id = record.get("custom_id", "")
                match = id_pattern.match(custom_id)
                if match:
                    prompt_id = int(match.group(1))
                    annotation_type = match.group(2)
                    if annotation_type in TARGET_COLUMNS:
                        emotions_list = extract_emotions_from_entry(record)
                        if emotions_list is not None:
                            if prompt_id not in annotation_map:
                                annotation_map[prompt_id] = {}
                            annotation_map[prompt_id][annotation_type] = emotions_list
            except json.JSONDecodeError:
                continue
    return annotation_map

# ================= FUNCIONES DE PLOTTING =================

def plot_emotion_dist(df, col_name, mode='primary', output_dir='.'):
    """
    Genera un bar plot horizontal.
    mode: 'primary' (toma solo el 1er elemento) o 'multilabel' (explode de la lista).
    """
    # 1. Preparar datos
    clean_series = df[col_name].dropna()
    
    if len(clean_series) == 0:
        print(f"No hay datos para graficar {col_name}")
        return

    n_prompts = len(df) # Total de filas originales
    
    if mode == 'primary':
        # Tomar primer elemento si la lista no está vacía
        data_to_plot = clean_series.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None).dropna()
        title_prefix = "Primary Emotion Frequency"
        palette_choice = "viridis" # Azul/Morado a Amarillo
    else:
        # Multilabel: Explode
        data_to_plot = clean_series.explode().dropna()
        title_prefix = "Multilabel Frequency"
        palette_choice = "viridis" 

    # Contar frecuencias
    counts = data_to_plot.value_counts()
    
    if len(counts) == 0:
        return

    # 2. Configuración del Plot
    plt.figure(figsize=(12, 8))
    sns.set_style("white") # Fondo limpio como en la imagen
    
    # Crear paleta: Viridis invertido (o normal dependiendo de la preferencia). 
    # En tus imagenes: Alto valor = Morado Oscuro, Bajo valor = Verde/Amarillo.
    # La paleta 'viridis' por defecto va de Morado (0) a Amarillo (1).
    # Como barplot asigna colores en orden, y nuestros datos están ordenados descendente:
    # El primero (mayor) tomará el primer color de viridis (morado). Perfecto.
    colors = sns.color_palette(palette_choice, n_colors=len(counts))
    
    ax = sns.barplot(x=counts.values, y=counts.index, palette=colors, hue=counts.index, legend=False)
    
    # 3. Estética
    pretty_name = col_name.replace("_", " ").title()
    plt.title(f"{title_prefix}: {pretty_name}", fontsize=16)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Emotion Label", fontsize=12)
    
    # Quitar bordes superior y derecho (spine)
    sns.despine()

    # 4. Anotaciones al final de las barras
    max_x = counts.values.max()
    offset = max_x * 0.01 # 1% de padding
    
    for i, v in enumerate(counts.values):
        ax.text(v + offset, i, str(v), color='black', va='center', fontsize=10)

    # 5. Disclaimer Box (Solo para Multilabel como en tu ejemplo)
    if mode == 'multilabel':
        total_occurrences = counts.sum()
        avg_per_prompt = total_occurrences / n_prompts
        
        disclaimer_text = (
            f"Disclaimer: Total frequencies ({total_occurrences}) exceed the number of prompts ({n_prompts}) "
            f"due to multilabel classification.\nAverage labels per prompt: {avg_per_prompt:.2f}"
        )
        
        # Añadir cuadro de texto centrado abajo
        plt.figtext(0.5, 0.01, disclaimer_text, ha="center", fontsize=9,
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Ajustar márgenes para que quepa el texto
        plt.subplots_adjust(bottom=0.15)
    else:
        plt.subplots_adjust(right=0.95)

    # 6. Guardar
    filename = f"{col_name}_{mode}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Gráfico guardado: {filename}")

# ================= MAIN =================

def main():
    print("=== UNIFICACIÓN DE DATOS Y GENERACIÓN DE GRÁFICOS ===")
    
    # 1. Carga PKL
    if not os.path.exists(INPUT_PKL_PATH):
        raise FileNotFoundError(f"No existe: {INPUT_PKL_PATH}")
    
    print(f"--> Cargando PKL original: {INPUT_PKL_PATH}")
    df = pd.read_pickle(INPUT_PKL_PATH)
    original_len = len(df)

    print(f"--> Eliminando archivo original: {INPUT_PKL_PATH}")
    os.remove(INPUT_PKL_PATH)
    
    # 2. Carga Anotaciones
    annotations = load_annotations_map(ANNOTATIONS_JSONL_PATH)
    
    # 3. Merge de Columnas
    def get_annotation(pid, col_name):
        if pid in annotations and col_name in annotations[pid]:
            return annotations[pid][col_name]
        return np.nan # <--- CAMBIO SOLICITADO AQUI (np.nan)

    print("--> Agregando columnas...")
    for col in TARGET_COLUMNS:
        df[col] = df['prompt_id'].map(lambda pid: get_annotation(pid, col))
        print(f"    '{col}': {df[col].notna().sum()} filas completas.")

    if len(df) != original_len:
        print(f"[ALERTA] Filas cambiaron de {original_len} a {len(df)}.")
    
    # 4. Guardar PKL
    print(f"--> Guardando PKL: {OUTPUT_PKL_PATH}")
    df.to_pickle(OUTPUT_PKL_PATH)
    
    # 5. Generar Gráficos
    print(f"--> Generando gráficos en: {PLOTS_DIR}")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    for col in TARGET_COLUMNS:
        # Gráfico 1: Emoción Primaria (Primera de la lista)
        plot_emotion_dist(df, col, mode='primary', output_dir=PLOTS_DIR)
        # Gráfico 2: Multilabel (Todas las emociones)
        plot_emotion_dist(df, col, mode='multilabel', output_dir=PLOTS_DIR)

    print("¡Proceso terminado exitosamente!")

if __name__ == "__main__":
    main()