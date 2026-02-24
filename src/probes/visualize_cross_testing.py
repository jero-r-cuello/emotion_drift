import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
LLM_USED = "Llama-2-7b-chat-hf"
BASE_DIR = "/home/jcuello/emotion_drift"
FIGURES_DIR = os.path.join(BASE_DIR, "figures", f"cross_testing_performance_{LLM_USED}")

# Nombre del archivo CSV generado por el script 1
INPUT_CSV = os.path.join(FIGURES_DIR, "cross_test_bootstrap_results.csv")

TAXONOMIES = ['ekman_basic_emotions', 'plutchik_wheel']

# Rutas a los baselines (Tal cual estaban en tu script original)
# Ajusta estas rutas si es necesario
PATH_BASELINE_GEN = "results/probes_generated_prompts_Llama-2-7b-chat-hf/full_probing_metrics_Llama-2-7b-chat-hf_final_F1.csv"
PATH_BASELINE_HUMAN = "results/probes_human_centric_Llama-2-7b-chat-hf/full_probing_metrics_Llama-2-7b-chat-hf_final_F1.csv"

# --- VISUALIZACIÓN ---

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"Error: No se encontró el archivo de resultados en {INPUT_CSV}")
        print("Ejecuta primero el script de procesamiento.")
        exit()

    print(f"Cargando resultados de: {INPUT_CSV}")
    results_df = pd.read_csv(INPUT_CSV)

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    for tax in TAXONOMIES:
        subset = results_df[results_df['taxonomy'] == tax]
        if subset.empty: 
            print(f"No hay datos para la taxonomía: {tax}")
            continue
            
        plt.figure(figsize=(12, 7))
        
        # SERIE 1: Gen -> Human
        data_gh = subset[subset['train_source'] == 'Generated'].sort_values('layer')
        
        plt.plot(data_gh['layer'], data_gh['f1_mean'], 
                 label="Train: Generated $\\to$ Test: Human", 
                 color='#1f77b4', linewidth=2)
        
        plt.fill_between(data_gh['layer'], data_gh['f1_lower'], data_gh['f1_upper'], 
                         color='#1f77b4', alpha=0.25)

        # SERIE 2: Human -> Gen
        data_hg = subset[subset['train_source'] == 'Human'].sort_values('layer')
        
        plt.plot(data_hg['layer'], data_hg['f1_mean'], 
                 label="Train: Human $\\to$ Test: Generated", 
                 color='#ff7f0e', linewidth=2)
        
        plt.fill_between(data_hg['layer'], data_hg['f1_lower'], data_hg['f1_upper'], 
                         color='#ff7f0e', alpha=0.25)
        
        # SERIE 3: Gen -> Gen (baseline)
        if os.path.exists(PATH_BASELINE_GEN):
            data_gg = pd.read_csv(PATH_BASELINE_GEN)
            data_gg = data_gg[data_gg["taxonomy"] == tax]
            # Filtro básico por si el CSV tiene todas las taxonomías mezcladas (asumiendo que tiene la columna correcta)
            # Si los CSVs de baseline ya están separados por carpetas, esto no afecta.
            plt.plot(data_gg['layer'], data_gg['macro_f1'], 
                     label="Train: Generated $\\to$ Test: Generated", 
                     color='#1f77b4', linewidth=2, linestyle='--')
        else:
            print(f"Advertencia: No se encontró baseline Gen->Gen en {PATH_BASELINE_GEN}")
        
        # SERIE 4: Human -> Human (baseline)
        if os.path.exists(PATH_BASELINE_HUMAN):
            data_hh = pd.read_csv(PATH_BASELINE_HUMAN)
            data_hh = data_hh[data_hh["taxonomy"] == tax]
            plt.plot(data_hh['layer'], data_hh['macro_f1'], 
                     label="Train: Human $\\to$ Test: Human", 
                     color='#ff7f0e', linewidth=2, linestyle='--')
        else:
             print(f"Advertencia: No se encontró baseline Human->Human en {PATH_BASELINE_HUMAN}")

        plt.title(f"Cross-Dataset Robustness (Stratified Bootstrap 10k)\nTaxonomy: {tax} | Metric: Macro F1", fontsize=14)
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("Macro F1-Score (with 95% CI)", fontsize=12)
        plt.legend(loc='lower right')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(FIGURES_DIR, f"cross_test_{tax}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico guardado: {save_path}")
        plt.close()

    print("\nVisualización completada.")