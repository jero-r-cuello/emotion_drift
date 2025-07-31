#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# --- PASO 1: Carga y Preparación de Datos ---

# Define la ruta al archivo de datos. Modifícala si es necesario.
DATA_PATH = "../../data/03_activations/andyzou_situations_microsoft_Phi-3-medium-128k-instruct_20250728_173814.pkl"

# Comprobación de seguridad para asegurar que el archivo existe
if not os.path.exists(DATA_PATH):
    print(f"Error: Archivo de datos no encontrado en la ruta especificada: {DATA_PATH}")
    exit()

print(f"Cargando datos desde {DATA_PATH}...")
nested_df = pd.read_pickle(DATA_PATH)
print("Datos cargados exitosamente.")

# Dividir en conjuntos de entrenamiento y prueba
train_df = nested_df[nested_df['split'] == 'train'].copy()
test_df = nested_df[nested_df['split'] == 'test'].copy()

# Obtener listas de emociones y capas para iterar sobre ellas
unique_emotions = train_df['emotion_considered'].unique()
if not train_df.empty:
    layer_numbers = list(train_df.iloc[0]['activations']['last_token_activation'].keys())
else:
    print("Error: El conjunto de entrenamiento está vacío.")
    exit()

print(f"Se analizarán las emociones: {list(unique_emotions)}")
print(f"Se analizarán {len(layer_numbers)} capas por emoción.")


# --- PASO 2: Entrenamiento y Evaluación de Sondas Lineales ---

print("\n" + "="*50)
print("Iniciando entrenamiento y evaluación de las sondas lineales...")
print("="*50)

# Diccionario para almacenar los resultados de precisión
probe_accuracies_by_layer = {emotion: {} for emotion in unique_emotions}

# Bucle principal por cada emoción
for emotion in unique_emotions:
    print(f"\n--- Procesando probes para la emoción: {emotion.upper()} ---")
    
    # Filtrar los DataFrames una vez por emoción para mayor eficiencia
    emotion_train_df = train_df[train_df['emotion_considered'] == emotion]
    emotion_test_df = test_df[test_df['emotion_considered'] == emotion]
    
    # Extraer las etiquetas binarias (0 o 1) para la tarea de clasificación
    y_train_binary = emotion_train_df['label'].values
    y_test_binary = emotion_test_df['label'].values
    
    # Bucle anidado para entrenar una sonda por cada capa
    for layer in layer_numbers:
        
        # Preparar las features (X) para la capa actual
        X_train_layer = np.array([d['last_token_activation'][layer] for d in emotion_train_df['activations']])
        X_test_layer = np.array([d['last_token_activation'][layer] for d in emotion_test_df['activations']])
        
        # Entrenar la sonda lineal (Regresión Logística)
        # Es un clasificador lineal simple, ideal para esta tarea.
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train_layer, y_train_binary)
        
        # Realizar predicciones en el conjunto de prueba
        predictions = probe.predict(X_test_layer)
        
        # Calcular y almacenar la precisión
        accuracy = accuracy_score(y_test_binary, predictions)
        probe_accuracies_by_layer[emotion][layer] = accuracy
        
    print(f"Se han entrenado y evaluado {len(layer_numbers)} sondas para '{emotion}'.")

print("\nProceso completado.")


# --- PASO 3: Informe de Resultados ---

print("\n" + "="*50)
print("INFORME DE PRECISIÓN DE SONDAS LINEALES")
print("="*50)

# Almacenar los mejores resultados para un resumen final
best_results = {}

for emotion, layer_accuracies in probe_accuracies_by_layer.items():
    
    print(f"\n--- Resultados para: {emotion.upper()} ---")
    
    if not layer_accuracies:
        print("No hay resultados de precisión para esta emoción.")
        continue
        
    # Imprimir la precisión de cada capa para la emoción actual
    for layer, acc in layer_accuracies.items():
        print(f"Capa {layer:<2}: {acc:.2%}")
        
    # Encontrar y guardar la mejor capa para esta emoción
    best_layer = max(layer_accuracies, key=layer_accuracies.get)
    best_accuracy = layer_accuracies[best_layer]
    best_results[emotion] = {'layer': best_layer, 'accuracy': best_accuracy}

print("\n" + "-"*50)
print("RESUMEN - MEJOR CAPA POR EMOCIÓN")
print("-"*50)

for emotion, result in best_results.items():
    print(f"Emoción '{emotion}':")
    print(f"  -> Mejor rendimiento en la Capa {result['layer']} con una precisión del {result['accuracy']:.2%}")

print("\n" + "="*50)
# %%
