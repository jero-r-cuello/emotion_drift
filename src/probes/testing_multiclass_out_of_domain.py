# testing_out_of_domain.py
#%%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


MODELS_DIR = "/home/jcuello/emotion_drift/models/multiclass_probes"
NEW_DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/out_of_domain_Meta-Llama-3-8B_20250806_155308.pkl"
EMOTION_COLUMN = "emotion_scenario"
LLM_USED = "Meta-Llama-3-8B"
os.makedirs("/home/jcuello/emotion_drift/figures/multi_class_probe_evaluation", exist_ok=True)

print(f'Loading data from {NEW_DATA_PATH}')
new_df = pd.read_pickle(NEW_DATA_PATH)
print(f'Data loaded succesfuly')

layer_numbers = list(new_df.iloc[0]["activations"]["last_token_activation"].keys())
print(f'Found {len(layer_numbers)} models')

print("\n" + "="*50)
print(f'Evaluating in "out-of-domain" dataset...')
print("="*50)


layer_accuracies = {}
for layer in layer_numbers:
    model_filename = f'{LLM_USED}_multiclass_probe_layer_{layer}_trained_on_{EMOTION_COLUMN}.joblib'
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

    print(f'Layer {layer:<2}: Accuracy = {accuracy:.2%}')

# Accuracy report
print("\n" + "="*50)
print("ACCURACY REPORT")
print("="*50)

if not layer_accuracies:
    print("No models found")

if layer_accuracies:
    layers = list(layer_accuracies.keys())
    accuracies = list(layer_accuracies.values())

    average_accuracy = np.mean(accuracies)
    best_layer = max(layer_accuracies, key=layer_accuracies.get)
    best_accuracy = layer_accuracies[best_layer]

    print(f'Mean accuracy over all layers is: {average_accuracy:.2%}')
    print("\n" + "-"*50)
    print(f'The best performing layer was layer {best_layer}')
    print(f'with an accuracy of {best_accuracy:.2%}.')
    print("\nEvaluation process completed!")

    num_emociones = new_df[EMOTION_COLUMN].nunique()
    chance_level = 1 / num_emociones if num_emociones > 0 else 0

    # Accuracy per layer bar plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ["#4682B4" if layer != best_layer else "#FF6347" for layer in layers]
    
    bars = ax.bar(layers, accuracies, color=colors)

    ax.axhline(average_accuracy, color="darkgreen", linestyle="--", linewidth=2, label=f'Mean over layers ({average_accuracy:.2%})')
    ax.axhline(chance_level, color="firebrick", linestyle=":", linewidth=2, label=f'Chance level ({chance_level:.2%})')
    
    ax.set_title("Accuracy of multi-class model by layer (Testing Out-of-Domain)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(layers)
    ax.tick_params(axis="x", rotation=45)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, max(accuracies) * 1.1) 

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()

    output_filename = f"/home/jcuello/emotion_drift/figures/multi_class_probe_evaluation/{LLM_USED}_accuracy_per_layer.png"
    plt.savefig(output_filename, dpi=300)
    print(f'\nBar plot saved at: {output_filename}')
    plt.show()

print(f'\nGenerating confusion matrix for best layer ({best_layer})...')

best_model_path = os.path.join(MODELS_DIR, f'{LLM_USED}_multiclass_probe_layer_{best_layer}_trained_on_{EMOTION_COLUMN}.joblib')
best_probe_model = joblib.load(best_model_path)

X_test_best_layer = np.array([d["last_token_activation"][best_layer] for d in new_df["activations"]])
y_true = new_df[EMOTION_COLUMN].values
y_pred = best_probe_model.predict(X_test_best_layer)

class_labels = best_probe_model.classes_

# Absolut confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    ax=ax,
    cmap="Blues",
    xticks_rotation="vertical",
    display_labels=class_labels
)
ax.set_title(f'Confusion matrix for layer {best_layer} (best)\n(Accuracy: {best_accuracy:.2%})', fontsize=15)
plt.tight_layout()

plt.savefig(f'/home/jcuello/emotion_drift/figures/multi_class_probe_evaluation/{LLM_USED}_confusion_matrix_layer_{best_layer}.png', dpi=300)
print(f'Confusion matrix saved as "confusion_matrix_layer_{best_layer}.png"')
plt.show()

# Normalized confusion matrix
fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    ax=ax_norm,
    cmap="Greens",
    xticks_rotation="vertical",
    display_labels=class_labels,
    normalize="true"
)
ax_norm.set_title(f'Normalized confusion matrix for layer {best_layer} (best)', fontsize=15)
plt.tight_layout()

plt.savefig(f'/home/jcuello/emotion_drift/figures/multi_class_probe_evaluation/{LLM_USED}_confusion_matrix_normalized_layer_{best_layer}.png', dpi=300)
print(f'Confusion matrix saved as "confusion_matrix_normalized_layer_{best_layer}.png"')
plt.show()

# %%
