# testing_svm_multiclass_out_of_domain.py
#%%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = "/home/jcuello/emotion_drift/models/multiclass_svm_probes" 
NEW_DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/out_of_domain_microsoft_Phi-3-medium-128k-instruct_20250804_165130.pkl"
EMOTION_COLUMN = "emotion_scenario"
FIGURES_DIR = "/home/jcuello/emotion_drift/figures/multi_class_svm_evaluation"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Kernels to evaluate
KERNELS_TO_TEST = ["linear", "rbf"]

print(f'Loading data from {NEW_DATA_PATH}')
new_df = pd.read_pickle(NEW_DATA_PATH)
print(f'Data loaded successfully')

layer_numbers = list(new_df.iloc[0]["activations"]["last_token_activation"].keys())
print(f'Found activations for {len(layer_numbers)} layers to test.')


for svm_kernel in KERNELS_TO_TEST:
    print("\n" + "="*60)
    print(f'EVALUATING SVM MODELS WITH KERNEL: "{svm_kernel.upper()}"')
    print("="*60)

    layer_accuracies = {}
    for layer in layer_numbers:
        model_filename = f'multiclass_svm_{svm_kernel}_probe_layer_{layer}_trained_on_{EMOTION_COLUMN}.joblib'
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f'Layer {layer:<2}: WARNING - Model not found in "{model_path}". Skipping this layer.')
            continue

        probe_model = joblib.load(model_path)

        X_test_layer = np.array([d["last_token_activation"][layer] for d in new_df["activations"]])
        y_test_layer = new_df[EMOTION_COLUMN].values

        predictions = probe_model.predict(X_test_layer)
        accuracy = accuracy_score(y_test_layer, predictions)
        layer_accuracies[layer] = accuracy

        print(f'Layer {layer:<2} (Kernel: {svm_kernel}): Accuracy = {accuracy:.2%}')

    # Report accuracy
    print("\n" + "="*50)
    print(f'ACCURACY REPORT (KERNEL: {svm_kernel.upper()})')
    print("="*50)

    if not layer_accuracies:
        print("Any model found for this kernel.")
        continue

    # Best and avg accuracy
    average_accuracy = np.mean(list(layer_accuracies.values()))
    best_layer = max(layer_accuracies, key=layer_accuracies.get)
    best_accuracy = layer_accuracies[best_layer]

    print(f'Mean accuracy over all layers is: {average_accuracy:.2%}')
    print("-" * 50)
    print(f'The best performing layer was layer {best_layer} with an accuracy of {best_accuracy:.2%}.')
    print("-" * 50)

    # Bar plot of accuracy per layer
    layers = list(layer_accuracies.keys())
    accuracies = list(layer_accuracies.values())
    chance_level = 1 / new_df[EMOTION_COLUMN].nunique()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ["#4682B4" if layer != best_layer else "#FF6347" for layer in layers]
    bars = ax.bar(layers, accuracies, color=colors)
    
    ax.axhline(average_accuracy, color="darkgreen", linestyle="--", linewidth=2, label=f'Mean over layers ({average_accuracy:.2%})')
    ax.axhline(chance_level, color="firebrick", linestyle=":", linewidth=2, label=f'Chance level ({chance_level:.2%})')
    
    ax.set_title(f'Accuracy of SVM ({svm_kernel.upper()}) by Layer (Out-of-Domain)', fontsize=18, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(layers)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, max(accuracies) * 1.15)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()

    output_filename = os.path.join(FIGURES_DIR, f'accuracy_per_layer_{svm_kernel}.png')
    plt.savefig(output_filename, dpi=300)
    print(f'\nAcc plot saved at: {output_filename}')
    plt.show()

    # Confusion matrices for best layer
    print(f'\nGenerating confusion matrix for the best performing layer (Layer {best_layer}, Kernel: {svm_kernel})...')
    
    best_model_filename = f'multiclass_svm_{svm_kernel}_probe_layer_{best_layer}_trained_on_{EMOTION_COLUMN}.joblib'
    best_model_path = os.path.join(MODELS_DIR, best_model_filename)
    best_probe_model = joblib.load(best_model_path)

    X_test_best_layer = np.array([d["last_token_activation"][best_layer] for d in new_df["activations"]])
    y_true = new_df[EMOTION_COLUMN].values
    y_pred = best_probe_model.predict(X_test_best_layer)
    class_labels = best_probe_model.classes_

    # Absolut confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_cm, cmap="Blues", xticks_rotation="vertical", display_labels=class_labels)
    ax_cm.set_title(f'Confusion Matrix (Kernel: {svm_kernel.upper()}) - Layer {best_layer}\n(Accuracy: {best_accuracy:.2%})', fontsize=15)
    plt.tight_layout()
    cm_filename = os.path.join(FIGURES_DIR, f'confusion_matrix_{svm_kernel}_layer_{best_layer}.png')
    plt.savefig(cm_filename, dpi=300)
    print(f'Confusion matrix saved at: "{os.path.basename(cm_filename)}"')
    plt.show()

    # Normalized confusion matrix
    fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_norm, cmap="Greens", xticks_rotation="vertical", display_labels=class_labels, normalize="true")
    ax_norm.set_title(f'Normalized Confusion Matrix (Kernel: {svm_kernel.upper()}) - Layer {best_layer}', fontsize=15)
    plt.tight_layout()
    norm_cm_filename = os.path.join(FIGURES_DIR, f'confusion_matrix_normalized_{svm_kernel}_layer_{best_layer}.png')
    plt.savefig(norm_cm_filename, dpi=300)
    print(f'Normalized confusion matrix saved at: "{os.path.basename(norm_cm_filename)}"')
    plt.show()

print("\nEvaluation process completed for all specified kernels!")
# %%
