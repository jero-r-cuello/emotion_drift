# testing_binaryclass_out_of_domain.py
#%%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODELS_DIR = "/home/jcuello/emotion_drift/models/binary_probes"
NEW_DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/out_of_domain_microsoft_Phi-3-medium-128k-instruct_20250804_165130.pkl"
EMOTION_COLUMN = "emotion_scenario"
FIGURES_DIR = "/home/jcuello/emotion_drift/figures/binary_probe_evaluation"

os.makedirs(FIGURES_DIR, exist_ok=True)

print(f'Loading data from {NEW_DATA_PATH}...')
if not os.path.exists(NEW_DATA_PATH):
    print(f'Error: Data file not found at "{NEW_DATA_PATH}"')
    exit()

new_df = pd.read_pickle(NEW_DATA_PATH)
print("Data loaded successfully.")

if new_df.empty:
    print("Error: The loaded DataFrame is empty.")
    exit()
    
layer_numbers = list(new_df.iloc[0]["activations"]["last_token_activation"].keys())
unique_emotions = new_df[EMOTION_COLUMN].unique()

print(f'\nFound {len(layer_numbers)} layers to evaluate.')
print(f'Found {len(unique_emotions)} emotions to test: {list(unique_emotions)}')


print("\n" + "="*50)
print("Evaluating binary probes and storing predictions...")
print("="*50)

evaluation_results = {emotion: {"layers": {}} for emotion in unique_emotions}

for emotion in unique_emotions:
    print(f'\n--- Evaluating probes for emotion: {emotion.upper()} ---')
    y_true_binary = (new_df[EMOTION_COLUMN] == emotion).astype(int)

    for layer in layer_numbers:
        model_filename = f'binary_probe_{emotion}_layer_{layer}_trained_on_{EMOTION_COLUMN}.joblib'
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f'Layer {layer:<2}: WARNING - Model not found. Skipping.')
            continue

        probe_model = joblib.load(model_path)
        X_test_layer = np.array([d["last_token_activation"][layer] for d in new_df["activations"]])

        predictions = probe_model.predict(X_test_layer)
        accuracy = accuracy_score(y_true_binary, predictions)

        evaluation_results[emotion]["layers"][layer] = {
            "accuracy": accuracy,
            "predictions": predictions 
        }

    if not evaluation_results[emotion]["layers"]:
        print(f'Could not find any matching models for emotion "{emotion}".')
    else:
        print(f'Finished evaluating {len(evaluation_results[emotion]["layers"])} probes for "{emotion}".')

# Accuracy report
print("\n" + "="*50)
print("FINAL REPORT & VISUALIZATIONS")
print("="*50)

for emotion, data in evaluation_results.items():
    if not data["layers"]:
        print(f'\n--- No results for: {emotion.upper()} ---')
        continue

    print(f'\n--- Results for: {emotion.upper()} ---')
    
    # Extract the layer accuracies for finding the best one
    layer_accuracies = {layer: results["accuracy"] for layer, results in data["layers"].items()}
    
    best_layer = max(layer_accuracies, key=layer_accuracies.get)
    best_accuracy = layer_accuracies[best_layer]
    
    print(f'Best performance on Layer {best_layer} with an accuracy of {best_accuracy:.2%}')

    # Create accuracy bar plot
    layers = list(layer_accuracies.keys())
    acc_values = list(layer_accuracies.values())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ["#4682B4" if layer != best_layer else "#FF6347" for layer in layers]
    bars = ax.bar(layers, acc_values, color=colors)
    
    ax.set_title(f'Binary Probe Accuracy for "{emotion.upper()}" by Layer', fontsize=18, fontweight="bold")
    ax.set_xlabel("Layer Number", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(layers)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1.05)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()
    plot_filename = os.path.join(FIGURES_DIR, f'accuracy_plot_{emotion}.png')
    plt.savefig(plot_filename, dpi=300)
    print(f'Saved accuracy plot: {plot_filename}')
    plt.show()
    plt.close(fig)

    # Generate confusion matrices for best layer
    print(f'Generating confusion matrix for "{emotion}" at its best layer ({best_layer})...')
    
    y_pred_best = data["layers"][best_layer]["predictions"]
    y_true_binary = (new_df[EMOTION_COLUMN] == emotion).astype(int)
    
    display_labels = [f'Not {emotion}', emotion]
    
    # Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true_binary, y_pred_best, ax=ax_cm, cmap="Blues", display_labels=display_labels)
    ax_cm.set_title(f'Confusion Matrix for "{emotion.upper()}" (Layer {best_layer})\nAccuracy: {best_accuracy:.2%}')
    plt.tight_layout()
    cm_filename = os.path.join(FIGURES_DIR, f'confusion_matrix_{emotion}_layer_{best_layer}.png')
    plt.savefig(cm_filename, dpi=300)
    print(f'Saved confusion matrix: {cm_filename}')
    plt.show()
    plt.close(fig_cm)
    
    # Plot Normalized Confusion Matrix
    fig_norm, ax_norm = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true_binary, y_pred_best, ax=ax_norm, cmap="Greens", normalize="true", display_labels=display_labels)
    ax_norm.set_title(f'Normalized Confusion Matrix for "{emotion.upper()}" (Layer {best_layer})')
    plt.tight_layout()
    cm_norm_filename = os.path.join(FIGURES_DIR, f'confusion_matrix_normalized_{emotion}_layer_{best_layer}.png')
    plt.savefig(cm_norm_filename, dpi=300)
    print(f'Saved normalized confusion matrix: {cm_norm_filename}')
    plt.show()
    plt.close(fig_norm)

print("\n\nProcess completed successfully.")
# %%
