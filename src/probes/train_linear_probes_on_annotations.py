#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import math
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, 
    classification_report, 
    accuracy_score, 
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Configuración Estética ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# --- Configuración ---
LLM_USED = "Llama-2-7b-chat-hf" #"Qwen2.5-14B-Instruct"  #
MODEL_DIM = 4096 # 5120 # 
DATASET = "human_centric"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl" # "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl" #"/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Qwen2.5-14B-Instruct_20251220_225401_FINAL.pkl" #
SENTIMENT_TARGETS = ['ekman_basic_emotions', 'go_emotions', 'plutchik_wheel']
N_CONTROL = 3
min_samples_required = 5 # Puedes subir esto a 3 o 5 si sigue fallando

# Directorios
BASE_DIR = "/home/jcuello/emotion_drift"
PLOTS_DIR_BASE = os.path.join(BASE_DIR, "figures", f"probes_{DATASET}_{LLM_USED}")
RESULTS_DIR_BASE = os.path.join(BASE_DIR, "results", f"probes_{DATASET}_{LLM_USED}")
CM_BASE_DIR = os.path.join(PLOTS_DIR_BASE, "probes_confusion_matrices") # Carpeta para Matrices de Confusión
MODELS_DIR_BASE = os.path.join(BASE_DIR, "models")

os.makedirs(PLOTS_DIR_BASE, exist_ok=True)
os.makedirs(RESULTS_DIR_BASE, exist_ok=True)
os.makedirs(CM_BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR_BASE, exist_ok=True)

def get_normalized_entropy(y_labels):
    """Calculates Normalized Shannon Entropy (0=Imbalanced, 1=Balanced)."""
    counts = Counter(y_labels)
    total = sum(counts.values())
    n_classes = len(counts)
    if n_classes <= 1: return 0.0
    
    # H = -sum(p * log2(p))
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    
    # Normalize by max possible entropy log2(K)
    return entropy / math.log2(n_classes)

# --- Carga de Datos ---
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f'{DATA_PATH} not found')
print(f"Cargando datos desde {DATA_PATH}...")
nested_df_original = pd.read_pickle(DATA_PATH)
if nested_df_original.empty: raise ValueError("Dataframe vacío.")
print("Datos cargados correctamente.")

# Lista para acumular TODAS las métricas
all_metrics_rows = []

# =============================================================================
# BUCLE PRINCIPAL (Taxonomías)
# =============================================================================
for sentiment_target in SENTIMENT_TARGETS:
    print(f"\n{'='*60}\n   PROCESANDO TAXONOMÍA: {sentiment_target}\n{'='*60}")
    
    # Preprocesamiento
    nested_df = nested_df_original.copy()
    if sentiment_target not in nested_df.columns: continue

    mask = nested_df[sentiment_target].apply(lambda x: isinstance(x, list) and len(x) > 0)
    nested_df = nested_df[mask].copy()
    nested_df[sentiment_target] = nested_df[sentiment_target].str[0]
        
    if nested_df.empty: continue
    
    final_labels = sorted(nested_df[sentiment_target].unique())
    print(f"Clases ({len(final_labels)}): {final_labels}")

    # Check Entropy
    all_current_labels = nested_df[sentiment_target].tolist()
    imbalance_score = get_normalized_entropy(all_current_labels)
    print(f"Metrics Check -> Normalized Entropy: {imbalance_score:.3f}")

    # Crear directorio para Matrices de Confusión de esta taxonomía
    cm_target_dir = os.path.join(CM_BASE_DIR, sentiment_target)
    os.makedirs(cm_target_dir, exist_ok=True)

    try:
        num_layers = len(nested_df['activations'].iloc[0])
    except:
        num_layers = 0

    # =========================================================================
    # BUCLE DE CAPAS
    # =========================================================================
    for layer_num in tqdm(range(num_layers), desc=f"Layers ({sentiment_target})"):
        
        # --- 1. Extracción Robusta ---
        try:
            X_list = []
            y_list = []
            for act_row, label in zip(nested_df['activations'], nested_df[sentiment_target]):
                try:
                    act = act_row.iloc[layer_num]['last_token_activation']
                    if not isinstance(act, np.ndarray): continue
                    if act.ndim > 1: act = act.squeeze()
                    if act.shape == (MODEL_DIM,): 
                        X_list.append(act)
                        y_list.append(label)
                except: continue

            if len(X_list) == 0: continue
            X = np.stack(X_list)
            y_real = np.array(y_list)

        except Exception as e:
            print(f"Error capa {layer_num}: {e}")
            continue

        class_counts = Counter(y_real)
        
        # Identificar clases que no cumplen el mínimo
        classes_to_drop = [cls for cls, count in class_counts.items() if count < min_samples_required]
        print(f"Classes to drop (freq. below {min_samples_required}: {classes_to_drop}")
        
        if classes_to_drop:
            # Crear máscara booleana: True si la clase es válida, False si se debe borrar
            mask_valid = ~np.isin(y_real, classes_to_drop)
            
            # Filtrar X e y
            X = X[mask_valid]
            y_real = y_real[mask_valid]
            
            # (Opcional) Imprimir aviso solo una vez o si es crítico
            # print(f"   [Layer {layer_num}] Filtered out rare classes: {classes_to_drop}")

        # Verificación de seguridad: Si después de filtrar nos quedamos sin datos o con 1 sola clase
        if len(X) == 0 or len(np.unique(y_real)) < 2:
            print(f"   [Layer {layer_num}] Skipping: Not enough data/classes after filtering.")
            continue
        # =====================================================================

        # --- 2. Entrenamiento REAL ---
        # Ahora el split no fallará por clases con 1 solo miembro
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_real, test_size=0.2, random_state=42, stratify=y_real)
        except ValueError as e:
            # Captura extra por si quedan clases con muy pocos ejemplos para el split 80/20
            print(f"   [Layer {layer_num}] Split error (not enough samples for stratification): {e}")
            continue
        
        clf = make_pipeline(
            StandardScaler(), 
            LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, solver='lbfgs', n_jobs=-1)
        )

        clf.fit(X_train, y_train)

        try:
            model_filename = f"{DATASET}_{LLM_USED}_{sentiment_target}_layer_{layer_num}.joblib"
            model_path = os.path.join(MODELS_DIR_BASE, model_filename)
            joblib.dump(clf, model_path)
        except Exception as e:
            print(f"Warning: No se pudo guardar el modelo en la capa {layer_num}: {e}")

        y_pred = clf.predict(X_test)
        
        # --- 3. Métricas Exhaustivas ---
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report_dict['macro avg']['f1-score']
        mcc_score = matthews_corrcoef(y_test, y_pred) # Mantenemos cálculo MCC por completitud en CSV

        metrics_row = {
            'taxonomy': sentiment_target,
            'layer': layer_num,
            'accuracy': report_dict['accuracy'],
            'macro_f1': macro_f1,
            'mcc': mcc_score,
            'imbalance_entropy': imbalance_score
        }        
        
        for label, metrics in report_dict.items():
            if label in ['accuracy', 'macro avg', 'weighted avg']: continue
            metrics_row[f'{label}_f1'] = metrics['f1-score']

        # --- 4. Tareas de Control (Permutation Test) ---
        control_f1_scores = []
        control_mcc_scores = []
        
        clf_control = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, n_jobs=-1)
        )        

        for i in range(N_CONTROL): 
            y_shuffled = np.random.permutation(y_real)
            X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_shuffled, test_size=0.2, random_state=i)
            clf_control.fit(X_tr_c, y_tr_c)
            y_pred_c = clf_control.predict(X_te_c)
            
            control_f1_scores.append(f1_score(y_te_c, y_pred_c, average='macro'))
            control_mcc_scores.append(matthews_corrcoef(y_te_c, y_pred_c))

        avg_control_f1 = np.mean(control_f1_scores)
        avg_control_mcc = np.mean(control_mcc_scores)

        metrics_row['control_macro_f1'] = avg_control_f1
        metrics_row['control_mcc'] = avg_control_mcc
        
        # --- 5. Métrica Normalizada (AHORA CALCULAMOS PARA F1 TAMBIÉN) ---
        # MCC Normalized (Lo guardamos por si acaso, pero no lo graficamos)
        if avg_control_mcc < 0.99:
            norm_mcc = (mcc_score - avg_control_mcc) / (1 - avg_control_mcc)
        else:
            norm_mcc = 0.0
            
        # Macro F1 Normalized (Esta es la que graficaremos)
        if avg_control_f1 < 0.99:
            norm_f1 = (macro_f1 - avg_control_f1) / (1 - avg_control_f1)
        else:
            norm_f1 = 0.0

        metrics_row['normalized_mcc'] = norm_mcc
        metrics_row['normalized_macro_f1'] = norm_f1
        
        all_metrics_rows.append(metrics_row)
        
        # --- 6. GENERAR Y GUARDAR MATRIZ DE CONFUSIÓN ---
        try:
            # Tamaño dinámico según número de clases
            fig_size = 10 if len(final_labels) < 10 else 20
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            
            # Plot
            # Cambiamos título para reflejar F1
            disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
            ax.set_title(f"Confusion Matrix: {sentiment_target} - Layer {layer_num}\nMacro F1: {macro_f1:.3f}")
            
            # Save
            cm_filename = f"cm_layer_{layer_num:02d}.png"
            cm_path = os.path.join(cm_target_dir, cm_filename)
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close(fig)

            fig_norm, ax_norm = plt.subplots(figsize=(fig_size, fig_size))
            
            # Calcular matriz normalizada sobre las filas (True Labels)
            cm_norm = confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize='true')
            disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=clf.classes_)
            
            # Plot
            disp_norm.plot(cmap='Blues', ax=ax_norm, xticks_rotation='vertical', values_format='.2f')
            ax_norm.set_title(f"Normalized CM: {sentiment_target} - Layer {layer_num}\nMacro F1: {macro_f1:.3f}")
            ax_norm.grid(False) 
            
            cm_norm_filename = f"cm_norm_layer_{layer_num:02d}.png"
            cm_norm_path = os.path.join(cm_target_dir, cm_norm_filename)
            plt.tight_layout()
            plt.savefig(cm_norm_path)
            plt.close(fig_norm)
    
        except Exception as e:
            print(f"Warning: Could not save Confusion Matrix for L{layer_num}: {e}")

        # --- 7. Imprimir Resumen ---
        print(f"\n>> Layer {layer_num} | {sentiment_target}")
        display_cols = ['accuracy', 'macro_f1', 'control_macro_f1', 'normalized_macro_f1']
        row_str = " | ".join([f"{k}: {metrics_row[k]:.4f}" for k in display_cols])
        print(f"   GENERAL: {row_str}")
        
        # Imprimir top 3 mejores y peores clases
        class_f1s = {k.replace('_f1',''): v for k,v in metrics_row.items() if '_f1' in k and 'macro' not in k}
        sorted_classes = sorted(class_f1s.items(), key=lambda item: item[1], reverse=True)
        print(f"   MEJORES: {sorted_classes[:3]}")
        print(f"   PEORES:  {sorted_classes[-3:]}")

# =============================================================================
# GUARDADO Y VISUALIZACIÓN FINAL
# =============================================================================

df_results = pd.DataFrame(all_metrics_rows)
csv_path = os.path.join(RESULTS_DIR_BASE, f'full_probing_metrics_{LLM_USED}_final_F1.csv')
df_results.to_csv(csv_path, index=False)
print(f"\nDataFrame completo guardado en: {csv_path}")

# Definir colores fijos para cada taxonomía
palette = sns.color_palette("tab10", n_colors=len(SENTIMENT_TARGETS))
color_map = {target: color for target, color in zip(SENTIMENT_TARGETS, palette)}

# --- PLOT 0: Normalized Entropy Comparison (Bar Plot) ---
# (Este se mantiene igual ya que es sobre los datos, no sobre la métrica)
plt.figure(figsize=(10, 6))
entropy_df = df_results[['taxonomy', 'imbalance_entropy']].drop_duplicates().sort_values('taxonomy')

ax = sns.barplot(
    data=entropy_df, 
    x='taxonomy', 
    y='imbalance_entropy', 
    hue='taxonomy', 
    palette=color_map, 
    dodge=False
)

plt.title('Class Balance Comparison (Normalized Entropy)', fontsize=15)
plt.xlabel('Taxonomy', fontsize=12)
plt.ylabel('Normalized Entropy\n(0 = Highly Imbalanced, 1 = Balanced)', fontsize=12)
plt.ylim(0, 1.15) 
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')

plt.legend([], [], frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR_BASE, '00_taxonomy_imbalance.png'))
plt.show()

# --- PLOT 1: Absolute Performance (Macro F1) ---
plt.figure(figsize=(14, 8))
for target in SENTIMENT_TARGETS:
    subset = df_results[df_results['taxonomy'] == target]
    if subset.empty: continue
    
    c = color_map[target]
    # Real Model (Solid line, markers)
    plt.plot(subset['layer'], subset['macro_f1'], marker='o', label=f'{target}', color=c)
    # Control Model (Dashed line, same color)
    plt.plot(subset['layer'], subset['control_macro_f1'], linestyle='--', alpha=0.6, color=c)

plt.title('Absolute Performance (Macro F1) vs Control', fontsize=16)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Macro F1 Score', fontsize=12)
plt.legend(title="Taxonomy (Dashed = Control)")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR_BASE, '01_absolute_performance_f1.png'))
plt.show()

# --- PLOT 2: Normalized Macro F1 (Comparison) ---
# MODIFICADO: Ahora usa 'normalized_macro_f1' en lugar de 'normalized_mcc'
plt.figure(figsize=(14, 8))
for target in SENTIMENT_TARGETS:
    subset = df_results[df_results['taxonomy'] == target]
    if subset.empty: continue
    
    c = color_map[target]
    ent = subset['imbalance_entropy'].iloc[0]
    
    # Normalized Score (Usando F1)
    plt.plot(subset['layer'], subset['normalized_macro_f1'], marker='o', linewidth=2, 
             label=f"{target} (Ent: {ent:.2f})", color=c)

plt.title('Normalized Macro F1 (Selectivity over Memorization)', fontsize=16)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Normalized Macro F1 Score', fontsize=12)
plt.ylim(-0.1, 1.05) 
plt.axhline(0, color='black', linewidth=0.5)
plt.legend(fontsize=12, title="Taxonomy (Entropy: 1=Bal, 0=Imbal)")
plt.grid(True, linestyle='--', alpha=0.7)

# Nota explicativa actualizada
text_str = "Score = (F1_Real - F1_Control) / (1 - F1_Control)"
plt.text(0.02, 0.02, text_str, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR_BASE, '02_comparison_normalized_f1.png'))
plt.show()

print("\n--- ALL PLOTS (BASED ON MACRO F1) AND MATRICES GENERATED SUCCESSFULLY ---")
#%%