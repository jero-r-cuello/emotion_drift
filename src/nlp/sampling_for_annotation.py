# %%
import json
import random
import pandas as pd
from pathlib import Path

# --- Configuración ---
INPUT_FILENAME = "/home/jcuello/emotion_drift/data/02_generated/outputs_Llama-2-7b-chat-hf_20250830_215905.jsonl"
SAMPLES_PER_EMOTION = 12
OUTPUT_FILENAME = "sampled_texts_for_annotation.csv"
RANDOM_SEED = 42

def stratified_sample_by_emotion(input_path, num_samples_per_category):
    """
    Lee un archivo JSON Lines, agrupa los textos por 'emotion_considered',
    y devuelve una muestra aleatoria estratificada.

    Args:
        input_path (Path): La ruta al archivo de entrada .jsonl.
        num_samples_per_category (int): El número de muestras a tomar de cada categoría de emoción.

    Returns:
        list: Una lista de diccionarios, donde cada dict contiene 'emotion_considered' y 'text_to_annotate'.
    """
    if not input_path.exists():
        print(f"Error: El archivo de entrada '{input_path}' no fue encontrado.")
        return None

    print(f"Leyendo y agrupando textos por emoción desde: {input_path}...")
    
    # Usamos un diccionario de sets para agrupar textos únicos por emoción
    texts_by_emotion = {}
    line_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                text = data.get("generated_text")
                emotion = data.get("emotion_considered")
                
                # Solo procesar si tenemos tanto el texto como la emoción
                if text and emotion:
                    # Si es la primera vez que vemos esta emoción, inicializamos su set
                    if emotion not in texts_by_emotion:
                        texts_by_emotion[emotion] = set()
                    # Añadir el texto al set de la emoción correspondiente
                    texts_by_emotion[emotion].add(text.strip())
                    
            except json.JSONDecodeError:
                print(f"Advertencia: Se omitió la línea {line_count} por no ser un JSON válido.")

    print("\nAgrupación completada. Resumen de textos únicos por emoción:")
    for emotion, texts in texts_by_emotion.items():
        print(f"- {emotion}: {len(texts)} textos únicos encontrados.")

    print(f"\nRealizando muestreo estratificado (tomando {num_samples_per_category} por emoción)...")
    
    final_samples = []
    # Iterar sobre cada grupo de emoción para tomar las muestras
    for emotion, unique_texts_set in sorted(texts_by_emotion.items()):
        unique_texts_list = list(unique_texts_set)
        
        # Determinar cuántas muestras tomar (el objetivo o el máximo disponible)
        num_to_sample = min(num_samples_per_category, len(unique_texts_list))
        
        if len(unique_texts_list) < num_samples_per_category:
            print(f"  Advertencia para '{emotion}': Solo hay {len(unique_texts_list)} textos disponibles. Se tomarán todos.")
        
        # Tomar la muestra aleatoria del grupo actual
        sampled_texts = random.sample(unique_texts_list, num_to_sample)
        
        # Añadir los resultados a nuestra lista final con su estructura de diccionario
        for text in sampled_texts:
            final_samples.append({
                "emotion_considered": emotion,
                "text_to_annotate": text
            })

    print("\nMuestreo completado con éxito.")
    return final_samples

def save_samples_to_csv_robust(samples, output_path):
    """
    Guarda la lista de diccionarios muestreados en un archivo CSV.
    Se asume que los textos ya han sido procesados para no tener saltos de línea.
    """
    if not samples:
        print("No hay muestras para guardar.")
        return

    print(f"Guardando {len(samples)} muestras en '{output_path}'...")
    try:
        df = pd.DataFrame(samples)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print("Archivo CSV guardado con éxito.")
        
    except Exception as e:
        print(f"Ocurrió un error al guardar el archivo CSV: {e}")


if __name__ == "__main__":
    input_file_path = Path(INPUT_FILENAME)
    output_file_path = Path(OUTPUT_FILENAME)
    
    # 1. Configurar la semilla de aleatoriedad para obtener resultados replicables
    print(f"Usando semilla aleatoria: {RANDOM_SEED} para replicabilidad.")
    random.seed(RANDOM_SEED)

    # 2. Obtener las muestras estratificadas con el formato original (con \n)
    original_samples_structured = stratified_sample_by_emotion(input_file_path, SAMPLES_PER_EMOTION)
    
    if original_samples_structured:
        # 3. Preprocesar las muestras para reemplazar \n por su versión literal \\n
        print("\nReemplazando saltos de línea con '\\n' para un guardado robusto en CSV...")
        processed_samples = []
        for sample_dict in original_samples_structured:
            processed_text = sample_dict['text_to_annotate'].replace('\n', '\\n')
            processed_samples.append({
                'emotion_considered': sample_dict['emotion_considered'],
                'text_to_annotate': processed_text
            })

        # 4. Guardar las muestras ya procesadas y aplanadas
        save_samples_to_csv_robust(processed_samples, output_file_path)
        
        # Opcional: Imprimir una vista previa de los datos
        print("\n--- Vista previa de las primeras 5 filas del DataFrame a guardar ---")
        df_preview = pd.DataFrame(processed_samples)
        print(df_preview.head())
# %%
