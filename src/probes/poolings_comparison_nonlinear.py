#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Cambiado: SVC en lugar de LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ... Carga de datos ...
print("Cargando datos para comparación de poolings...")
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
nested_df = pd.read_pickle(DATA_PATH)
LLM_USED = "Llama-2-7b-chat-hf"

# Definir qué agregaciones vamos a comparar
AGGREGATION_TYPES = ['concat_mmm_activation', 'last_token_activation',
                     'mean_activation', 'max_activation',
                     'min_activation', 'amp_activation']

num_layers = 32
# SOLUCIÓN PUNTO 5: Pre-inicializar con NaN
results_agg = {agg: [np.nan] * num_layers for agg in AGGREGATION_TYPES}

target_col = 'ekman_basic_emotions' 

# --- PREPROCESAMIENTO ---
nested_df[target_col] = nested_df[target_col].str[0]

# Limpiar etiquetas vacías
original_len = len(nested_df)
nested_df = nested_df.dropna(subset=[target_col])
print(f"Se eliminaron {original_len - len(nested_df)} filas sin etiquetas (NaNs o listas vacías).")

# Reiniciar índices para asegurar alineación perfecta 0..N
nested_df = nested_df.reset_index(drop=True)
y = nested_df[target_col].values

print(f"Comparando aggregations para: {target_col}")

# SOLUCIÓN PUNTO 6: Split fuera del bucle y estratificado
# Definimos los índices globales que DEBEN usarse para train y test
indices_global = np.arange(len(y))
train_idx_global, test_idx_global, y_train_global, y_test_global = train_test_split(
    indices_global, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
)

# --- BUCLE PRINCIPAL ---
for agg_type in AGGREGATION_TYPES:
    print(f"--- Procesando: {agg_type} ---")
    
    for layer in tqdm(range(num_layers)):
        try:
            X_list = []
            valid_indices_curr = [] # Guardaremos qué filas fueron válidas en esta capa
            
            # Iteramos con enumerate para saber el índice original de la fila
            for idx, row_act in enumerate(nested_df['activations']):
                try:
                    # LOGICA DE EXTRACCIÓN ROBUSTA
                    act = row_act.iloc[layer][agg_type]
                    
                    # 1. Validar tipo
                    if not isinstance(act, np.ndarray): continue
                    
                    # 2. Corregir dimensiones (Squeeze)
                    if act.ndim > 1: act = act.squeeze()
                    
                    # 3. Validar shape
                    expected_dim = 4096 * 3 if 'concat' in agg_type else 4096
                    
                    if act.shape == (expected_dim,): 
                        X_list.append(act)
                        valid_indices_curr.append(idx)
                except Exception:
                    continue

            # Si no hay datos válidos en esta capa, pasamos
            if len(X_list) == 0:
                print(f"Advertencia: Capa {layer} vacía para {agg_type}")
                continue

            X = np.stack(X_list)
            
            # --- CRUCE DE ÍNDICES ---
            valid_indices_arr = np.array(valid_indices_curr)
            
            mask_train = np.isin(valid_indices_arr, train_idx_global)
            mask_test = np.isin(valid_indices_arr, test_idx_global)
            
            X_train_curr = X[mask_train]
            y_train_curr = y[valid_indices_arr[mask_train]]
            
            X_test_curr = X[mask_test]
            y_test_curr = y[valid_indices_arr[mask_test]]
            
            # Verificación de seguridad
            if len(np.unique(y_train_curr)) < 2:
                continue

            # Entrenamiento - Cambiado a SVC No Lineal
            clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
            clf.fit(X_train_curr, y_train_curr)
            
            # Predicción
            score = f1_score(y_test_curr, clf.predict(X_test_curr), average='macro')
            results_agg[agg_type][layer] = score
            
        except KeyError:
            print(f"Key {agg_type} no encontrada en layer {layer}.")
            break
        except ValueError as ve:
            print(f"Error de valor en layer {layer}: {ve}")
            continue

# --- CÁLCULO DE BASELINES ---

# 1. Dummy Stratified
dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
dummy_clf.fit(np.zeros((len(y_train_global), 1)), y_train_global)
dummy_score = f1_score(y_test_global, dummy_clf.predict(np.zeros((len(y_test_global), 1))), average='macro')
print(f"Stratified Dummy Score: {dummy_score:.4f}")

# 2. Permuted SVM Baseline
avg_scores = {agg: np.nanmean(scores) for agg, scores in results_agg.items()}
if all(np.isnan(v) for v in avg_scores.values()):
    print("No hay scores válidos.")
    best_agg = AGGREGATION_TYPES[0]
else:
    best_agg = max(avg_scores, key=avg_scores.get)

print(f"Calculando Permuted Baseline con: {best_agg}")
permuted_scores = [np.nan] * num_layers

np.random.seed(42)
y_train_permuted_global = np.random.permutation(y_train_global)
idx_to_permuted_label = dict(zip(train_idx_global, y_train_permuted_global))

for layer in tqdm(range(num_layers), desc="Permuted Dummy"):
    try:
        X_list = []
        valid_indices_curr = []
        
        expected_dim = 4096 * 3 if 'concat' in best_agg else 4096
        
        for idx, row_act in enumerate(nested_df['activations']):
            try:
                act = row_act.iloc[layer][best_agg]
                if not isinstance(act, np.ndarray): continue
                if act.ndim > 1: act = act.squeeze()
                if act.shape == (expected_dim,):
                    X_list.append(act)
                    valid_indices_curr.append(idx)
            except: continue
            
        if not X_list: continue
        
        X = np.stack(X_list)
        valid_indices_arr = np.array(valid_indices_curr)
        
        mask_train = np.isin(valid_indices_arr, train_idx_global)
        mask_test = np.isin(valid_indices_arr, test_idx_global)
        
        X_train_curr = X[mask_train]
        indices_train_curr = valid_indices_arr[mask_train]
        
        y_train_curr_perm = np.array([idx_to_permuted_label[i] for i in indices_train_curr])
        
        X_test_curr = X[mask_test]
        y_test_curr = y[valid_indices_arr[mask_test]]
        
        # Entrenamiento - Cambiado a SVC No Lineal
        clf_perm = SVC(kernel='rbf', C=1.0, class_weight='balanced')
        clf_perm.fit(X_train_curr, y_train_curr_perm)
        
        score_perm = f1_score(y_test_curr, clf_perm.predict(X_test_curr), average='macro')
        permuted_scores[layer] = score_perm
        
    except KeyError:
        break

# --- GUARDAR LA TABLA ---
df_final_results = pd.DataFrame(results_agg)
df_final_results[f'permuted_svm_{best_agg}'] = permuted_scores
df_final_results['dummy_stratified'] = dummy_score
# Insertar la columna de capa al principio para mayor claridad
df_final_results.insert(0, 'layer', range(num_layers))

output_table_path = f"/home/jcuello/emotion_drift/results/non_linear_pooling_table_{target_col}.csv"
df_final_results.to_csv(output_table_path, index=False)

print(f"Results table saved: {output_table_path}")

# --- PLOT ---
plt.figure(figsize=(12, 6))

for agg_type, scores in results_agg.items():
    valid_idxs = [i for i, s in enumerate(scores) if not np.isnan(s)]
    valid_scores = [scores[i] for i in valid_idxs]
    if valid_scores:
        plt.plot(valid_idxs, valid_scores, label=agg_type, marker='o', alpha=0.7)

valid_idxs_perm = [i for i, s in enumerate(permuted_scores) if not np.isnan(s)]
valid_scores_perm = [permuted_scores[i] for i in valid_idxs_perm]
if valid_scores_perm:
    # Cambiado: Etiqueta actualizada a SVM
    plt.plot(valid_idxs_perm, valid_scores_perm, label=f'Permuted SVM ({best_agg})', 
             linestyle='--', color='gray', marker='x')

plt.axhline(y=dummy_score, color='r', linestyle=':', label='Dummy Stratified', linewidth=2)

plt.title(f"Token Aggregation Impact in Decoding for ({LLM_USED} - Nonlinear SVM)")
plt.xlabel("Layer")
plt.ylabel("Macro F1")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(f"/home/jcuello/emotion_drift/figures/non_linear_pooling_comparison_{target_col}.png", dpi=300)
plt.show()
plt.close()
#%%