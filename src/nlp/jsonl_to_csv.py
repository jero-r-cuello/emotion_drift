#%%
"""
Helper script to convert the JSONL files with annotations
into a consolidated CSV file ready for the performance analysis
"""

import json
import csv
from collections import defaultdict
import sys
import pandas as pd

# --- CONFIGURACIÓN ---

# Nombres de los archivos de entrada y salida
INPUT_JSONL_FILE = '/home/jcuello/emotion_drift/src/nlp/gpt-5-nano-all-annotations.jsonl'
OUTPUT_CSV_FILE = '/home/jcuello/emotion_drift/src/nlp/gpt-5-nano-consolidated_annotations.csv'

# Ruta al archivo CSV que contiene los textos de los prompts
PROMPTS_CSV_FILE = "/home/jcuello/emotion_drift/data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv"

# Nombres de las columnas clave en el archivo CSV de prompts.
# Basado en tu muestra, el ID es la columna 'id' y el texto es 'response_text'.
PROMPT_ID_COLUMN = 'id'
PROMPT_TEXT_COLUMN = 'response_text'

# Nombres de las taxonomías que se esperan encontrar en el 'custom_id'
TAXONOMY_1 = 'ekman_basic_emotions'
TAXONOMY_2 = 'go_emotions'


def load_prompts_from_csv(csv_path, id_col, text_col):
    """
    Carga los textos de los prompts desde un archivo CSV a un diccionario.
    Transforma el ID numérico del CSV (ej: 0) al formato esperado
    por el JSONL (ej: 'request-0') para poder mapearlos.
    """
    print(f"Cargando prompts desde el archivo: {csv_path}")
    prompts_dict = {}
    try:
        df = pd.read_csv(csv_path)
        # Verificar que las columnas necesarias existen en el DataFrame
        if id_col not in df.columns or text_col not in df.columns:
            print(f"ERROR: El archivo CSV '{csv_path}' debe contener las columnas '{id_col}' y '{text_col}'.", file=sys.stderr)
            sys.exit(1)
        
        # Iterar sobre el DataFrame para construir el diccionario con el formato de clave correcto
        for index, row in df.iterrows():
            # Clave: transforma el ID numérico (0) en un string ('request-0')
            key = f"request-{row[id_col]}"
            # Valor: el texto del prompt
            value = row[text_col]
            if pd.notna(value): # Asegurarse de que el texto no sea nulo
                prompts_dict[key] = value

        print(f"Se cargaron {len(prompts_dict)} prompts exitosamente.")
        return prompts_dict

    except FileNotFoundError:
        print(f"ERROR: El archivo CSV de prompts '{csv_path}' no fue encontrado.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ocurrió un error inesperado al leer el archivo CSV de prompts: {e}", file=sys.stderr)
        sys.exit(1)


def parse_jsonl_to_csv(input_path, output_path, prompts_dict):
    """
    Lee un archivo JSONL, agrupa los datos por prompt y configuración de modelo,
    y escribe el resultado en un archivo CSV.
    """
    grouped_data = defaultdict(dict)

    print(f"Leyendo y procesando el archivo JSONL: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    response_body = data.get('response', {}).get('body', {})
                    if not custom_id or not response_body:
                        print(f"ADVERTENCIA: Se omitió la línea {i+1} por falta de 'custom_id' o 'response.body'.")
                        continue

                    model_name = response_body.get('model')
                    effort = response_body.get('reasoning', {}).get('effort')
                    verbosity = response_body.get('text', {}).get('verbosity')
                    output_list = response_body.get('output', [])
                    
                    if len(output_list) < 2 or 'content' not in output_list[1] or not output_list[1]['content']:
                         print(f"ADVERTENCIA: Estructura de 'output' inesperada en línea {i+1}. Se omite.")
                         continue
                    output_text_str = output_list[1]['content'][0].get('text')

                    if not all([model_name, effort, verbosity, output_text_str]):
                        print(f"ADVERTENCIA: Se omitió la línea {i+1} por falta de datos esenciales (model, effort, verbosity, output).")
                        continue

                    # Extraer el ID del prompt (ej. 'request-0') del custom_id para usarlo como clave
                    prompt_id = "-".join(custom_id.split("-")[:2])
                    
                    # Extraer la taxonomía del custom_id
                    suffix_to_remove = f"-{verbosity}-{effort}"
                    if custom_id.endswith(suffix_to_remove):
                        base_id = custom_id[:-len(suffix_to_remove)]
                        taxonomy = base_id.rpartition('-')[-1]
                    else:
                        print(f"ADVERTENCIA: El formato de 'custom_id' en la línea {i+1} ('{custom_id}') es inesperado. Se omite.")
                        continue

                    annotation_data = json.loads(output_text_str)
                    labels = annotation_data.get('emotions', [])
                    justification = annotation_data.get('justification', '')

                    grouping_key = (prompt_id, model_name, verbosity, effort)
                    grouped_data[grouping_key][taxonomy] = {'labels': labels, 'justification': justification}

                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                    print(f"ADVERTENCIA: Error procesando la línea {i+1}: {e}. Se omite.")
    
    except FileNotFoundError:
        print(f"ERROR: El archivo de entrada '{input_path}' no fue encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Procesamiento completado. Se agruparon {len(grouped_data)} filas únicas para el CSV.")
    print(f"Escribiendo datos en el archivo: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'response_text', 'model',
            f'{TAXONOMY_1.replace("_basic_emotions", "")}_labels', f'{TAXONOMY_1.replace("_basic_emotions", "")}_justification',
            f'{TAXONOMY_2}_labels', f'{TAXONOMY_2}_justification'
        ])

        for key, taxonomies in sorted(grouped_data.items()): # sorted() para un orden predecible
            prompt_id, model_name, verbosity, effort = key
            tax1_data = taxonomies.get(TAXONOMY_1, {})
            tax2_data = taxonomies.get(TAXONOMY_2, {})

            row = [
                prompts_dict.get(prompt_id, f"Texto no encontrado para el ID: {prompt_id}"),
                f"{model_name}-verbosity-{verbosity}-effort-{effort}",
                str(tax1_data.get('labels', '[]')),
                tax1_data.get('justification', ''),
                str(tax2_data.get('labels', '[]')),
                tax2_data.get('justification', '')
            ]
            writer.writerow(row)

    print(f"¡El archivo '{output_path}' ha sido generado exitosamente!")


if __name__ == '__main__':
    # 1. Cargar los prompts desde el archivo CSV, creando un diccionario con las claves correctas (ej: 'request-0').
    prompts_dictionary = load_prompts_from_csv(PROMPTS_CSV_FILE, PROMPT_ID_COLUMN, PROMPT_TEXT_COLUMN)
    
    # 2. Procesar el archivo JSONL y generar el CSV final usando los prompts cargados.
    parse_jsonl_to_csv(INPUT_JSONL_FILE, OUTPUT_CSV_FILE, prompts_dictionary)
# %%
