#%%
import pandas as pd
import os
import json
import re

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Cambia estas rutas para que coincidan con la ubicación de tus archivos.

# Ruta a tu archivo .pkl
ruta_pkl = '/home/jcuello/emotion_drift/data/03_activations/andyzou_situations_microsoft_Phi-3-medium-128k-instruct_20250728_173814.pkl'

# Ruta a la carpeta que contiene los archivos emocion.json
ruta_json_folder = '/home/jcuello/emotion_drift/data/01_stimuli/andyzou_situations_dataset'

# --- 2. PRE-CARGA DE LOS ESCENARIOS DESDE LOS ARCHIVOS JSON ---
# Se cargan todos los escenarios en un diccionario para una búsqueda más eficiente.
# La estructura será: {'happiness': ['escenario1', 'escenario2'], 'sadness': [...]}

emotion_scenarios = {}
print("Cargando escenarios desde los archivos JSON...")

try:
    # Itera sobre cada archivo en la carpeta especificada
    for filename in os.listdir(ruta_json_folder):
        if filename.endswith('.json'):
            # Extrae el nombre de la emoción del nombre del archivo (ej: 'happiness.json' -> 'happiness')
            emotion_name = os.path.splitext(filename)[0]
            
            # Construye la ruta completa al archivo
            filepath = os.path.join(ruta_json_folder, filename)
            
            # Abre y carga el contenido del JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
                # Almacena la lista de escenarios en el diccionario
                emotion_scenarios[emotion_name] = scenarios
    print("Escenarios cargados exitosamente.")
    print(f"Emociones encontradas: {list(emotion_scenarios.keys())}")

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el directorio: {ruta_json_folder}")
    print("Por favor, verifica que la ruta 'ruta_json_folder' sea correcta.")
    exit()


# --- 3. FUNCIÓN PARA EXTRAER Y BUSCAR EL ESCENARIO ---

def encontrar_emocion_del_prompt(prompt_text):
    """
    Extrae el escenario de un prompt y busca a qué emoción pertenece.
    """
    # Patrón para extraer el texto entre "Scenario: " y "Answer:"
    # re.DOTALL se usa por si el escenario ocupa múltiples líneas.
    match = re.search(r"Scenario: (.*?)\nAnswer:", prompt_text, re.DOTALL)
    
    if not match:
        return "No se encontró el escenario en el prompt"
        
    # El texto del escenario extraído (eliminando espacios extra al inicio/final)
    scenario_text = match.group(1).strip()
    
    # Itera sobre las emociones y sus listas de escenarios cargadas previamente
    for emotion, scenarios in emotion_scenarios.items():
        if scenario_text in scenarios:
            return emotion # Devuelve el nombre de la emoción si se encuentra
            
    return "No encontrado en los JSON" # Si no se encuentra en ningún archivo


# --- 4. CARGAR EL DATAFRAME Y APLICAR LA FUNCIÓN ---

print("\nCargando el archivo .pkl...")
try:
    # Carga tu dataframe desde el archivo .pkl
    df = pd.read_pickle(ruta_pkl)

    print("Archivo .pkl cargado. Procesando filas...")
    
    # Crea la nueva columna "emotion_scenario" aplicando la función a cada valor de la columna "prompt"
    df['emotion_scenario'] = df['prompt'].apply(encontrar_emocion_del_prompt)
    
    print("\nProcesamiento completado.")
    
    # --- 5. VERIFICAR LOS RESULTADOS ---
    
    print("\nPrimeras 5 filas del DataFrame con la nueva columna:")
    print(df.head())
    
    print("\nConteo de valores para la nueva columna 'emotion_scenario':")
    print(df['emotion_scenario'].value_counts())

    # Opcional: Guardar el DataFrame modificado en un nuevo archivo
    # df.to_pickle('tu_archivo_modificado.pkl')
    # df.to_csv('tu_archivo_modificado.csv', index=False)
    # print("\nDataFrame modificado guardado.")

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo .pkl en: {ruta_pkl}")
    print("Por favor, verifica que la ruta 'ruta_pkl' sea correcta.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
# %%
