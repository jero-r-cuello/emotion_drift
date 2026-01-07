#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import math

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
LLM_USED = "Llama-2-7b-chat-hf"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
SENTIMENT_TARGETS = ['ekman_basic_emotions', 'go_emotions', 'plutchik_wheel']
N_CONTROL = 1

# Directorios
BASE_DIR = "/home/jcuello/emotion_drift"
PLOTS_DIR_BASE = os.path.join(BASE_DIR, "figures")
RESULTS_DIR_BASE = os.path.join(BASE_DIR, "results")
CM_BASE_DIR = os.path.join(PLOTS_DIR_BASE, "probes_confusion_matrices") # Carpeta para Matrices de Confusión

os.makedirs(PLOTS_DIR_BASE, exist_ok=True)
os.makedirs(RESULTS_DIR_BASE, exist_ok=True)
os.makedirs(CM_BASE_DIR, exist_ok=True)

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
                    if act.shape == (4096,): 
                        X_list.append(act)
                        y_list.append(label)
                except: continue

            if len(X_list) == 0: continue
            X = np.stack(X_list)
            y_real = np.array(y_list)

        except Exception as e:
            print(f"Error capa {layer_num}: {e}")
            continue

        # --- 2. Entrenamiento REAL ---
        X_train, X_test, y_train, y_test = train_test_split(X, y_real, test_size=0.2, random_state=42, stratify=y_real)
        
        clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # --- 3. Métricas Exhaustivas ---
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        mcc_score = matthews_corrcoef(y_test, y_pred)

        metrics_row = {
            'taxonomy': sentiment_target,
            'layer': layer_num,
            'accuracy': report_dict['accuracy'],
            'macro_f1': report_dict['macro avg']['f1-score'],
            'mcc': mcc_score,
            'imbalance_entropy': imbalance_score
        }        
        
        for label, metrics in report_dict.items():
            if label in ['accuracy', 'macro avg', 'weighted avg']: continue
            metrics_row[f'{label}_f1'] = metrics['f1-score']

        # --- 4. Tareas de Control (Permutation Test) ---
        control_f1_scores = []
        control_mcc_scores = []
        
        clf_control = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, n_jobs=-1)
        
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
        
        # --- 5. Métrica Normalizada ---
        if avg_control_mcc < 0.99:
            norm_mcc = (mcc_score - avg_control_mcc) / (1 - avg_control_mcc)
        else:
            norm_mcc = 0.0

        metrics_row['normalized_mcc'] = norm_mcc
        
        all_metrics_rows.append(metrics_row)
        
        # --- 6. GENERAR Y GUARDAR MATRIZ DE CONFUSIÓN (Requerimiento 4) ---
        try:
            # Tamaño dinámico según número de clases
            fig_size = 10 if len(final_labels) < 10 else 20
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            
            # Plot
            disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
            ax.set_title(f"Confusion Matrix: {sentiment_target} - Layer {layer_num}\nMCC: {mcc_score:.3f}")
            
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
            
            # Plot (notar values_format='.2f' para mostrar decimales)
            disp_norm.plot(cmap='Blues', ax=ax_norm, xticks_rotation='vertical', values_format='.2f')
            ax_norm.set_title(f"Normalized CM: {sentiment_target} - Layer {layer_num}\nMCC: {mcc_score:.3f}")
            ax_norm.grid(False) # Desactivar grid interno para limpieza visual
            
            # Guardar con nombre distinto
            cm_norm_filename = f"cm_norm_layer_{layer_num:02d}.png"
            cm_norm_path = os.path.join(cm_target_dir, cm_norm_filename)
            plt.tight_layout()
            plt.savefig(cm_norm_path)
            plt.close(fig_norm)
    
        except Exception as e:
            print(f"Warning: Could not save Confusion Matrix for L{layer_num}: {e}")

        # --- 7. Imprimir Resumen ---
        print(f"\n>> Layer {layer_num} | {sentiment_target}")
        display_cols = ['accuracy', 'macro_f1', 'control_macro_f1', 'mcc', 'control_mcc', 'normalized_mcc']
        # Convertimos a string formateado
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
csv_path = os.path.join(RESULTS_DIR_BASE, f'full_probing_metrics_{LLM_USED}_final.csv')
df_results.to_csv(csv_path, index=False)
print(f"\nDataFrame completo guardado en: {csv_path}")

# Definir colores fijos para cada taxonomía
palette = sns.color_palette("tab10", n_colors=len(SENTIMENT_TARGETS))
color_map = {target: color for target, color in zip(SENTIMENT_TARGETS, palette)}

# --- PLOT 0: Normalized Entropy Comparison (Bar Plot) ---
plt.figure(figsize=(10, 6))

# 1. Extract unique entropy values per taxonomy
entropy_df = df_results[['taxonomy', 'imbalance_entropy']].drop_duplicates().sort_values('taxonomy')

# 2. Create Bar Plot
# We use hue=taxonomy to strictly apply your 'color_map' dictionary
ax = sns.barplot(
    data=entropy_df, 
    x='taxonomy', 
    y='imbalance_entropy', 
    hue='taxonomy', 
    palette=color_map, 
    dodge=False
)

# 3. Styling
plt.title('Class Balance Comparison (Normalized Entropy)', fontsize=15)
plt.xlabel('Taxonomy', fontsize=12)
plt.ylabel('Normalized Entropy\n(0 = Highly Imbalanced, 1 = Balanced)', fontsize=12)
plt.ylim(0, 1.15) # Extra headroom for labels
plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# 4. Add numeric labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')

# Clean up
plt.legend([], [], frameon=False) # Remove redundant legend
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

# --- PLOT 2: Absolute Performance (MCC) ---
plt.figure(figsize=(14, 8))
for target in SENTIMENT_TARGETS:
    subset = df_results[df_results['taxonomy'] == target]
    if subset.empty: continue
    
    c = color_map[target]
    # Real Model
    plt.plot(subset['layer'], subset['mcc'], marker='o', label=f'{target}', color=c)
    # Control Model
    plt.plot(subset['layer'], subset['control_mcc'], linestyle='--', alpha=0.6, color=c)

plt.title('Absolute Performance (MCC) vs Control', fontsize=16)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12)
plt.legend(title="Taxonomy (Dashed = Control)")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR_BASE, '02_absolute_performance_mcc.png'))
plt.show()

# --- PLOT 3: Normalized MCC (Comparison) ---
plt.figure(figsize=(14, 8))
for target in SENTIMENT_TARGETS:
    subset = df_results[df_results['taxonomy'] == target]
    if subset.empty: continue
    
    c = color_map[target]
    ent = subset['imbalance_entropy'].iloc[0]
    
    # Normalized Score
    plt.plot(subset['layer'], subset['normalized_mcc'], marker='o', linewidth=2, 
             label=f"{target} (Ent: {ent:.2f})", color=c)

plt.title('Normalized MCC (Selectivity over Memorization)', fontsize=16)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Normalized MCC Score', fontsize=12)
plt.ylim(-0.1, 1.05) 
plt.axhline(0, color='black', linewidth=0.5)
plt.legend(fontsize=12, title="Taxonomy (Entropy: 1=Bal, 0=Imbal)")
plt.grid(True, linestyle='--', alpha=0.7)

# Nota explicativa
text_str = "Score = (MCC_Real - MCC_Control) / (1 - MCC_Control)"
plt.text(0.02, 0.02, text_str, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR_BASE, '03_comparison_normalized_mcc.png'))
plt.show()

print("\n--- ALL PLOTS AND MATRICES GENERATED SUCCESSFULLY ---")
#%%