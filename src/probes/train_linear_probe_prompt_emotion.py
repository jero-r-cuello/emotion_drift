# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import joblib 
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuración ---
emotion_to_test = "emotion_considered"
LLM_USED = "Llama-2-7b-chat-hf"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636.pkl"
MODELS_DIR = "/home/jcuello/emotion_drift/models"
dataset_used = "generated_prompts"

USE_PCA = False
N_COMPONENTS = 5

pca_suffix = '_pca' if USE_PCA else ''

PLOTS_DIR = "/home/jcuello/emotion_drift/figures"
LEARNING_CURVES_DIR = os.path.join(PLOTS_DIR, f"learning_curves{pca_suffix}")
CONFUSION_MATRICES_DIR = os.path.join(PLOTS_DIR, f"confusion_matrices{pca_suffix}")
ERROR_REPORTS_DIR = os.path.join(PLOTS_DIR, f"error_reports{pca_suffix}")
os.makedirs(LEARNING_CURVES_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRICES_DIR, exist_ok=True)
os.makedirs(ERROR_REPORTS_DIR, exist_ok=True)

MULTICLASS_PROBES_DIR = os.path.join(MODELS_DIR, f"multiclass_probes{pca_suffix}")
os.makedirs(MULTICLASS_PROBES_DIR, exist_ok=True)

if USE_PCA:
    PCA_OBJECTS_DIR = os.path.join(MODELS_DIR, "pca_objects")
    os.makedirs(PCA_OBJECTS_DIR, exist_ok=True)

# Carga de datos
if not os.path.exists(DATA_PATH):
    print(f'Error: Data file not found at the specified path: {DATA_PATH}')
    exit()
print(f'Loading data from {DATA_PATH}...')
nested_df = pd.read_pickle(DATA_PATH)
print("Data loaded successfully.")

# --- Filtrado de clases con 1 solo ejemplo (código anterior) ---
print(f"\nNúmero total de ejemplos antes de filtrar: {len(nested_df)}")
class_counts = nested_df[emotion_to_test].value_counts()
classes_to_remove = class_counts[class_counts <= 1].index.tolist()
if classes_to_remove:
    print(f"\nADVERTENCIA: Se encontraron {len(classes_to_remove)} clases con 1 solo ejemplo que serán descartadas:")
    print(classes_to_remove)
    nested_df = nested_df[~nested_df[emotion_to_test].isin(classes_to_remove)].copy()
    print(f"\nNúmero total de ejemplos después de filtrar: {len(nested_df)}\n")
else:
    print("\nTodas las clases tienen suficientes miembros para la división estratificada.\n")

final_emotion_classes = sorted(nested_df[emotion_to_test].unique())
print(f"Clases a utilizar en el entrenamiento: {final_emotion_classes}")

if nested_df.empty:
    print("Error: El DataFrame está vacío. No se puede continuar.")
    exit()
num_layers = len(nested_df['activations'].iloc[0])

# --- Función plot_learning_curve (sin cambios) ---
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None: _, axes = plt.subplots(1, 1, figsize=(10, 6));
    axes.set_title(title);
    if ylim is not None: axes.set_ylim(*ylim);
    axes.set_xlabel("Training examples"); axes.set_ylabel("Score (Accuracy)");
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy');
    train_scores_mean = np.mean(train_scores, axis=1); train_scores_std = np.std(train_scores, axis=1);
    test_scores_mean = np.mean(test_scores, axis=1); test_scores_std = np.std(test_scores, axis=1);
    axes.grid();
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r");
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g");
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score");
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score");
    axes.legend(loc="best");
    return plt

# --- Función Generar reporte de errores (sin cambios) ---
def get_top_n_errors(cm, class_labels, n=20):
    np.fill_diagonal(cm, 0)
    flat_cm = cm.flatten()
    top_indices = np.argsort(flat_cm)[-n:]
    report = "Top N Errores de Clasificación (Real -> Predicho: # de veces)\n"
    report += "="*60 + "\n"
    for idx in reversed(top_indices):
        count = flat_cm[idx]
        if count == 0: continue
        true_class_idx, pred_class_idx = np.unravel_index(idx, cm.shape)
        true_label = class_labels[true_class_idx]
        pred_label = class_labels[pred_class_idx]
        report += f"{true_label} -> {pred_label}: {count}\n"
    return report

# ### NUEVO ###: Diccionario para almacenar los accuracies de cada capa
layer_accuracies = {}

# --- Bucle de entrenamiento principal ---
for layer_num in tqdm(range(num_layers), desc="Entrenamiento de Capas"):
    print(f"\n--- Procesando Capa (Layer) {layer_num} ---")
    
    # Carga robusta de datos (sin cambios)
    X_data_list, y_labels_list, expected_dim = [], [], None
    print("Extrayendo y validando activaciones...")
    for index, row in nested_df.iterrows():
        try:
            activation = row['activations'].loc[layer_num, 'last_token_activation']
            if not isinstance(activation, np.ndarray):
                print(f"  - ADVERTENCIA: La activación en el índice {index} no es un array. Saltando.")
                continue
            if expected_dim is None:
                expected_dim = activation.shape[0]
                print(f"  - Dimensión de activación esperada detectada: {expected_dim}")
            if activation.shape[0] == expected_dim:
                X_data_list.append(activation)
                y_labels_list.append(row[emotion_to_test])
            else:
                print(f"  - ADVERTENCIA: Dimensión incorrecta en el índice {index}. Se esperaba {expected_dim}, se encontró {activation.shape}. Saltando.")
        except (KeyError, IndexError):
            print(f"  - ADVERTENCIA: No se encontró la capa {layer_num} en el índice {index}. Saltando.")
            continue
    if not X_data_list:
        print(f"Error: No se encontraron activaciones válidas para la capa {layer_num}. Saltando al siguiente.")
        continue
    X = np.vstack(X_data_list)
    y = pd.Series(y_labels_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if USE_PCA:
        print(f"--- Modo PCA activado (n_components={N_COMPONENTS}) ---")
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        joblib.dump(pca, os.path.join(PCA_OBJECTS_DIR, f'{LLM_USED}_pca_layer_{layer_num}.joblib'))
    
    model = LogisticRegression(random_state=42, max_iter=2000, C=0.1)

    print("Generando curva de aprendizaje...")
    fig_lc = plot_learning_curve(model, f"Learning Curve for Layer {layer_num}{pca_suffix}\n({LLM_USED})", X_train, y_train, cv=3, n_jobs=-1)
    fig_lc.savefig(os.path.join(LEARNING_CURVES_DIR, f'lc_layer_{layer_num}{pca_suffix}.png'))
    fig_lc.show()
    
    print("Entrenando modelo final...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # ### NUEVO ###: Almacenar accuracy para el gráfico final
    layer_accuracies[layer_num] = accuracy

    print(f"\n--- Métricas de Performance para Capa {layer_num}{pca_suffix} ---")
    print(f"Precisión (Accuracy) General: {accuracy:.4f}")
    
    # ### NUEVO ###: Generar e imprimir reporte de clasificación completo
    report = classification_report(y_test, y_pred, labels=final_emotion_classes, zero_division=0)
    print("\nReporte de Clasificación:")
    print(report)

    # --- Visualización y Reportes ---
    print("\nGenerando matriz de confusión y reportes de texto...")
    
    fig_cm, ax = plt.subplots(figsize=(25, 25))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues",
                                            labels=final_emotion_classes,
                                            display_labels=final_emotion_classes,
                                            xticks_rotation='vertical')
    ax.set_title(f'Confusion Matrix for Layer {layer_num}{pca_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFUSION_MATRICES_DIR, f'cm_layer_{layer_num}{pca_suffix}.png'))
    plt.show()

    cm = confusion_matrix(y_test, y_pred, labels=final_emotion_classes)
    error_report = get_top_n_errors(cm, final_emotion_classes, n=20)
    
    print(error_report)
    
    report_filename = f'performance_report_layer_{layer_num}{pca_suffix}.txt'
    with open(os.path.join(ERROR_REPORTS_DIR, report_filename), 'w') as f:
        f.write(f"Reporte de Performance para Capa {layer_num}{pca_suffix}\n")
        f.write(f"================================================\n\n")
        f.write(f"Accuracy General: {accuracy:.4f}\n\n")
        # ### NUEVO ###: Guardar el reporte de clasificación en el archivo
        f.write("--- Reporte de Clasificación Detallado ---\n")
        f.write(report + "\n\n")
        f.write("--- " + error_report)
    
    model_filename = f'{LLM_USED}_multiclass_probe_layer_{layer_num}{pca_suffix}_trained_on_{dataset_used}.joblib'
    model_path = os.path.join(MULTICLASS_PROBES_DIR, model_filename)
    joblib.dump(model, model_path)

print("\n--- Proceso finalizado ---")


# ### NUEVO ###: Bloque para generar el gráfico de resumen de accuracies
if layer_accuracies:
    print("\nGenerando gráfico de resumen de accuracy por capa...")
    
    layers = list(layer_accuracies.keys())
    accuracies = list(layer_accuracies.values())
    
    # Calcular accuracy por azar y promedio
    num_classes = len(final_emotion_classes)
    chance_accuracy = 1 / num_classes
    average_accuracy = np.mean(accuracies)
    
    plt.figure(figsize=(15, 8))
    plt.bar(layers, accuracies, color='skyblue', label='Accuracy por Capa')
    
    # Líneas de referencia
    plt.axhline(y=chance_accuracy, color='r', linestyle='--', label=f'Azar ({chance_accuracy:.3f})')
    plt.axhline(y=average_accuracy, color='g', linestyle='--', label=f'Promedio ({average_accuracy:.3f})')
    
    plt.xlabel('Capa del Modelo')
    plt.ylabel('Accuracy')
    plt.title(f'Rendimiento del Clasificador a través de las Capas de {LLM_USED}{pca_suffix}')
    plt.xticks(layers)
    plt.ylim(bottom=0, top=max(accuracies) * 1.2) # Ajustar el límite Y para mejor visualización
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Guardar y mostrar el gráfico
    summary_plot_path = os.path.join(PLOTS_DIR, f'accuracy_summary_per_layer{pca_suffix}.png')
    plt.savefig(summary_plot_path)
    print(f"Gráfico de resumen guardado en: {summary_plot_path}")
    plt.show()
else:
    print("\nNo se generaron resultados de accuracy, no se puede crear el gráfico de resumen.")

# %%
