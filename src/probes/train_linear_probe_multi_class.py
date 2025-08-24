# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
# --- REMOVED ---
# from sklearn.preprocessing import LabelEncoder 
from sklearn.decomposition import PCA
import joblib 
import matplotlib.pyplot as plt

# --- Configuración ---
emotion_to_test = "emotion_considered"
LLM_USED = "Llama-2-7b-chat-hf"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/llm_focused_Llama-2-7b-chat-hf_20250811_143357.pkl"
MODELS_DIR = "/home/jcuello/emotion_drift/models"
dataset_used = "llm_focused"

# --- NUEVO: Parámetro para controlar el uso de PCA ---
USE_PCA = False  # Cámbialo a False para ejecutar el script sin PCA
N_COMPONENTS = 5 # Conservar el 95% de la varianza. Puedes usar un entero (ej: 100) para un número fijo de componentes.

pca_suffix = '_pca' if USE_PCA else ''

PLOTS_DIR = "/home/jcuello/emotion_drift/figures"
LEARNING_CURVES_DIR = os.path.join(PLOTS_DIR, f"learning_curves{pca_suffix}")
CONFUSION_MATRICES_DIR = os.path.join(PLOTS_DIR, f"confusion_matrices{pca_suffix}")
os.makedirs(LEARNING_CURVES_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRICES_DIR, exist_ok=True)

MULTICLASS_PROBES_DIR = os.path.join(MODELS_DIR, f"multiclass_probes{pca_suffix}")
os.makedirs(MULTICLASS_PROBES_DIR, exist_ok=True)

# --- NUEVO: Directorio para guardar los objetos PCA ajustados ---
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

# --- CHANGE: Use string labels directly ---
# Scikit-learn models can handle string labels natively.
# We get the labels directly from the dataframe column.
y = nested_df[emotion_to_test]

# --- CHANGE: Get class names from the unique values in the label column ---
# We sort them to ensure consistent order in plots.
emotion_classes = sorted(y.unique())
print(f"Found emotion classes: {emotion_classes}")

num_layers = len(nested_df['activations'].iloc[0])

# Función plot_learning_curve (sin cambios)
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

# --- Bucle de entrenamiento principal (modificado para incluir PCA) ---
results = {}

for layer_num in range(num_layers):
    print(f"\n--- Procesando Capa (Layer) {layer_num} ---")
    if USE_PCA: print(f"--- Modo PCA activado (n_components={N_COMPONENTS}) ---")
    
    try:
        X_data_list = [row['activations'].loc[layer_num, 'last_token_activation'] for _, row in nested_df.iterrows()]
        X = np.vstack(X_data_list)
        # --- REMOVED ---
        # The `y` variable is already defined outside the loop with string labels.
        # y = y_encoded
    except (KeyError, IndexError):
        print(f"Error: No se encontró la capa {layer_num}. Saltando.")
        continue

    # The `y` variable here is the pandas Series of string labels.
    # train_test_split and stratify work correctly with string labels.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if USE_PCA:
        print("Ajustando PCA en los datos de entrenamiento...")
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        n_components_found = X_train.shape[1]
        print(f"PCA ajustado. Dimensión original: {X.shape[1]}, dimensión reducida: {n_components_found}")

        pca_filename = f'{LLM_USED}_pca_layer_{layer_num}_trained_on_{dataset_used}.joblib'
        pca_path = os.path.join(PCA_OBJECTS_DIR, pca_filename)
        joblib.dump(pca, pca_path)
    
    
    model = LogisticRegression(random_state=42, max_iter=2000, C=0.1)

    # Generar curva de aprendizaje
    # `plot_learning_curve` will pass the string labels in `y_train` to scikit-learn's `learning_curve`, which handles them correctly.
    print("Generando curva de aprendizaje...")
    title = f"Learning Curve for Layer {layer_num}{pca_suffix}\n({LLM_USED})"
    fig = plot_learning_curve(model, title, X_train, y_train, cv=3, n_jobs=-1)
    
    plot_filename = f'lc_layer_{layer_num}{pca_suffix}.png'
    plot_path = os.path.join(LEARNING_CURVES_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.show()
    fig.close()
    
    # Entrenar modelo
    # `model.fit` will automatically handle the string labels in `y_train`.
    print("Entrenando modelo final...")
    model.fit(X_train, y_train)
    
    # Evaluar
    # `model.predict` will return predictions as strings.
    # `accuracy_score` compares the true and predicted string labels.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precisión (Accuracy) en la capa {layer_num}{pca_suffix}: {accuracy:.4f}")

    # Generar matriz de confusión
    print("Generando matriz de confusión...")
    fig, ax = plt.subplots(figsize=(8, 8))
    # `from_predictions` works perfectly with string labels.
    # We pass `emotion_classes` (our sorted list of strings) to `display_labels` for clear plotting.
    ConfusionMatrixDisplay.from_predictions(y_test,
                                            y_pred,
                                            ax=ax,
                                            cmap="Blues",
                                            display_labels=emotion_classes,
                                            xticks_rotation='vertical')
    ax.set_title(f'Confusion Matrix for Layer {layer_num}{pca_suffix}')
    plt.tight_layout()
    
    cm_filename = f'cm_layer_{layer_num}{pca_suffix}.png'
    cm_path = os.path.join(CONFUSION_MATRICES_DIR, cm_filename)
    plt.savefig(cm_path)
    plt.show()
    plt.close()

    # Guardar modelo
    # The saved model will now remember the string class labels.
    # If you load it later and call .predict(), it will output strings directly.
    model_filename = f'{LLM_USED}_multiclass_probe_layer_{layer_num}{pca_suffix}_trained_on_{dataset_used}.joblib'
    model_path = os.path.join(MULTICLASS_PROBES_DIR, model_filename)
    joblib.dump(model, model_path)

print("\n--- Proceso finalizado ---")
# %%
