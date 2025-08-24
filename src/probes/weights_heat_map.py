# weights_heat_map.py
#%%
import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


MODEL_DIR = "/home/jcuello/emotion_drift/models/multiclass_probes"
FIGURE_DIR = "/home/jcuello/emotion_drift/figures/weight_heat_maps"
os.makedirs(FIGURE_DIR, exist_ok=True)

LLM_USED = "Phi-3-"
EMOTION_SCENARIO = "emotion_scenario"
NUM_LAYERS = 32
BIN_SIZE = 32

first_model_path = os.path.join(MODEL_DIR, f"{LLM_USED}_multiclass_probe_layer_0_trained_on_{EMOTION_SCENARIO}.joblib")
if not os.path.exists(first_model_path):
    raise FileNotFoundError(f"Could not find model file at {first_model_path}. Please check your EMOTION_SCENARIO and file paths.")

probe_layer_0 = joblib.load(first_model_path)
EMOTIONS = probe_layer_0.classes_
EMBEDDING_DIM = probe_layer_0.coef_.shape[1]

weights_by_emotion = {emotion: [] for emotion in EMOTIONS}

# Loop for weights extraction
for n_layer in range(NUM_LAYERS):
    model_filename = f"multiclass_probe_layer_{n_layer}_trained_on_{EMOTION_SCENARIO}.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if os.path.exists(model_path):
        probe = joblib.load(model_path)

        # "coef_" shape in a multi-class linear model is (n_classes, n_features).
        # Map the weights to the correct emotion using the "classes_" attribute.
        for i, emotion in enumerate(probe.classes_):
            weights_by_emotion[emotion].append(probe.coef_[i])


for emotion in EMOTIONS:
    weights_by_emotion[emotion] = np.vstack(weights_by_emotion[emotion])

fig, axes = plt.subplots(
    nrows=len(EMOTIONS),
    ncols=1,
    figsize=(20, 24),
    sharex=True,
    sharey=False
)

fig.suptitle(f'Probe Weights Across Layers (Bin Size = {BIN_SIZE})', fontsize=24)
cmap = "coolwarm"

all_weights = np.concatenate(list(weights_by_emotion.values()))

# Generate a heatmap for each emotion
for i, emotion in enumerate(EMOTIONS):
    ax = axes[i]
    weight_matrix = weights_by_emotion[emotion]

    num_neurons = weight_matrix.shape[1]
    num_bins = num_neurons // BIN_SIZE
    trimmed_matrix = weight_matrix[:, :num_bins * BIN_SIZE]
    
    binned_matrix = trimmed_matrix.reshape(weight_matrix.shape[0], num_bins, BIN_SIZE).mean(axis=2)
    transposed_binned_matrix = binned_matrix.T

    sns.heatmap(
        transposed_binned_matrix,
        ax=ax,
        cmap=cmap,
        cbar=False)

    ax.set_title(emotion.capitalize(), fontsize=16)
    ax.set_ylabel("Neuron Bin")

axes[-1].set_xlabel("Model Layer")
fig_filename = os.path.join(FIGURE_DIR, f'{LLM_USED}_weights_heatmap_{EMOTION_SCENARIO}_{BIN_SIZE}_bins.png')


fig.tight_layout(rect=[0, 0.05, 1, 0.96])

mappable = axes[0].get_children()[0]
cbar = fig.colorbar(
    mappable,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.08,
    shrink=0.7,
    aspect=30,
)
cbar.set_label("Learned Weight Value", size=14)

plt.savefig(fig_filename, dpi=300)
plt.show()

print(f"\nHeatmap saved to {fig_filename}")
# %%

import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

LLM_USED = "Meta-Llama-3-8B"
MODEL_DIR = "/home/jcuello/emotion_drift/models/multiclass_probes"
FIGURE_DIR = "/home/jcuello/emotion_drift/figures/weight_heat_maps"
os.makedirs(FIGURE_DIR, exist_ok=True)

EMOTION_SCENARIO = "emotion_scenario"
NUM_LAYERS = 32

# Chose plotting method
PLOT_METHOD = 'top_n_weights'

# Parameters for each method
# For 'binning'
BIN_SIZE = 32
# For 'svd'
N_COMPONENTS = 32 # Number of largest singular vectors (principal components) to show
# For 'top_n_weights'
N_TOP_FEATURES = 32 # Number of neurons with the largest mean absolute weights to show

first_model_path = os.path.join(MODEL_DIR, f"{LLM_USED}_multiclass_probe_layer_0_trained_on_{EMOTION_SCENARIO}.joblib")
if not os.path.exists(first_model_path):
    raise FileNotFoundError(f"Could not find model file at {first_model_path}. Please check your EMOTION_SCENARIO and file paths.")

probe_layer_0 = joblib.load(first_model_path)
EMOTIONS = probe_layer_0.classes_
EMBEDDING_DIM = probe_layer_0.coef_.shape[1]

weights_by_emotion = {emotion: [] for emotion in EMOTIONS}

for n_layer in range(NUM_LAYERS):
    model_filename = f"multiclass_probe_layer_{n_layer}_trained_on_{EMOTION_SCENARIO}.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if os.path.exists(model_path):
        probe = joblib.load(model_path)
        for i, emotion in enumerate(probe.classes_):
            weights_by_emotion[emotion].append(probe.coef_[i])
    else:
        print(f"Warning: Model file not found for layer {n_layer}. Skipping.")
        for emotion in EMOTIONS:
            weights_by_emotion[emotion].append(np.zeros(EMBEDDING_DIM))

for emotion in EMOTIONS:
    weights_by_emotion[emotion] = np.vstack(weights_by_emotion[emotion])

fig, axes = plt.subplots(
    nrows=len(EMOTIONS),
    ncols=1,
    figsize=(20, 24),
    sharex=True,
    sharey=False
)

# Set the main title based on the chosen method
if PLOT_METHOD == 'binning':
    fig_suptitle = f'Binned Probe Weights Across Layers (Bin Size = {BIN_SIZE})'
    fig_filename_suffix = f'binned_{BIN_SIZE}'
elif PLOT_METHOD == 'svd':
    fig_suptitle = f'Top {N_COMPONENTS} Singular Vectors of Probe Weights'
    fig_filename_suffix = f'svd_{N_COMPONENTS}'
elif PLOT_METHOD == 'top_n_weights':
    fig_suptitle = f'Top {N_TOP_FEATURES} features by Absolute Weight'
    fig_filename_suffix = f'top_weights_{N_TOP_FEATURES}'

fig.suptitle(fig_suptitle, fontsize=24)
cmap = "coolwarm"

# Generate a heatmap for each emotion ---
for i, emotion in enumerate(EMOTIONS):
    ax = axes[i]
    weight_matrix = weights_by_emotion[emotion]
    
    # Data transformation based on chosen method
    if PLOT_METHOD == 'binning':
        num_neurons = weight_matrix.shape[1]
        num_bins = num_neurons // BIN_SIZE
        trimmed_matrix = weight_matrix[:, :num_bins * BIN_SIZE]
        plot_matrix = trimmed_matrix.reshape(weight_matrix.shape[0], num_bins, BIN_SIZE).mean(axis=2)
        ax.set_ylabel("Neuron Bin")

    elif PLOT_METHOD == 'svd':
        # Use TruncatedSVD to find the top N principal components (singular vectors) of the neuron weights
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
        svd.fit(weight_matrix.T) # We fit on the transposed matrix to find components of neurons
        plot_matrix = svd.components_.T # Shape: (embedding_dim, n_components) -> we need to plot the components for each layer
        # For visualization, we want to see how each component's importance varies across layers.
        # So we project the original weights onto these components.
        plot_matrix = weight_matrix @ svd.components_.T # Shape: (num_layers, n_components)
        ax.set_ylabel("Principal Component")

    elif PLOT_METHOD == 'top_n_weights':
        # Calculate the mean absolute weight for each neuron across all layers
        mean_abs_weights = np.mean(np.abs(weight_matrix), axis=0)
        # Get the indices of the N neurons with the highest scores
        top_n_indices = np.argsort(mean_abs_weights)[-N_TOP_FEATURES:]
        # Select only these neurons from the original weight matrix
        plot_matrix = weight_matrix[:, top_n_indices]
        ax.set_ylabel("Top Neuron Index")

    # Transpose the final matrix for plotting: Layers on X-axis
    transposed_plot_matrix = plot_matrix.T
    
    sns.heatmap(
        transposed_plot_matrix,
        ax=ax,
        cmap=cmap,
        cbar=False,
    )

    ax.set_title(emotion.capitalize(), fontsize=16)

axes[-1].set_xlabel("Model Layer")
fig_filename = os.path.join(FIGURE_DIR, f'weights_heatmap_{EMOTION_SCENARIO}_{fig_filename_suffix}.png')

fig.tight_layout(rect=[0, 0.05, 1, 0.96])
mappable = axes[0].get_children()[0]
cbar = fig.colorbar(
    mappable,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.08,
    shrink=0.7,
    aspect=30,
)
cbar.set_label("Weight Value", size=14)

plt.show()
# %% Modificación para n_top global y no local de cada categoría

import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

LLM_USED = "Llama-2-7b-chat-hf"
DATASET_USED = "llm_focused"
MODEL_DIR = "/home/jcuello/emotion_drift/models/multiclass_probes"
FIGURE_DIR = "/home/jcuello/emotion_drift/figures/weight_heat_maps"
os.makedirs(FIGURE_DIR, exist_ok=True)

EMOTION_SCENARIO = "emotion_scenario"
NUM_LAYERS = 32

# Chose plotting method
PLOT_METHOD = 'top_n_weights'

# Parameters for each method
# For 'binning'
BIN_SIZE = 32
# For 'svd'
N_COMPONENTS = 32 # Number of largest singular vectors (principal components) to show
# For 'top_n_weights'
N_TOP_NEURONS = 32 # Number of neurons with the largest mean absolute weights to show

first_model_path = os.path.join(MODEL_DIR, f"{LLM_USED}_multiclass_probe_layer_0_trained_on_{DATASET_USED}.joblib")
if not os.path.exists(first_model_path):
    raise FileNotFoundError(f"Could not find model file at {first_model_path}. Please check your EMOTION_SCENARIO and file paths.")

probe_layer_0 = joblib.load(first_model_path)
EMOTIONS = probe_layer_0.classes_
EMBEDDING_DIM = probe_layer_0.coef_.shape[1]

weights_by_emotion = {emotion: [] for emotion in EMOTIONS}

for n_layer in range(NUM_LAYERS):
    model_filename = f"{LLM_USED}_multiclass_probe_layer_{n_layer}_trained_on_{DATASET_USED}.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if os.path.exists(model_path):
        probe = joblib.load(model_path)
        for i, emotion in enumerate(probe.classes_):
            weights_by_emotion[emotion].append(probe.coef_[i])
    else:
        print(f"Warning: Model file not found for layer {n_layer}. Skipping.")
        for emotion in EMOTIONS:
            weights_by_emotion[emotion].append(np.zeros(EMBEDDING_DIM))

for emotion in EMOTIONS:
    weights_by_emotion[emotion] = np.vstack(weights_by_emotion[emotion])

fig, axes = plt.subplots(
    nrows=len(EMOTIONS),
    ncols=1,
    figsize=(20, 24),
    sharex=True,
    sharey=False
)

# Set the main title based on the chosen method
if PLOT_METHOD == 'binning':
    fig_suptitle = f'Binned Probe Weights Across Layers (Bin Size = {BIN_SIZE})'
    fig_filename_suffix = f'binned_{BIN_SIZE}'
elif PLOT_METHOD == 'svd':
    fig_suptitle = f'Top {N_COMPONENTS} Singular Vectors of Probe Weights'
    fig_filename_suffix = f'svd_{N_COMPONENTS}'
elif PLOT_METHOD == 'top_n_weights':
    fig_suptitle = f'Top {N_TOP_NEURONS} Neurons by Absolute Weight Across All Emotions'
    fig_filename_suffix = f'top_weights_all_emotions_{N_TOP_NEURONS}'

fig.suptitle(fig_suptitle, fontsize=24)
cmap = "coolwarm"

# --- MODIFICATION FOR 'top_n_weights' ---
top_n_indices = None
if PLOT_METHOD == 'top_n_weights':
    # Stack all weight matrices to find the most important neurons across all emotions
    all_weights = np.vstack([weights_by_emotion[emotion] for emotion in EMOTIONS])
    # Calculate the mean absolute weight for each neuron across all layers and emotions
    mean_abs_weights_all_emotions = np.mean(np.abs(all_weights), axis=0)
    # Get the indices of the N neurons with the highest scores
    top_n_indices = np.argsort(mean_abs_weights_all_emotions)[-N_TOP_NEURONS:]
# --- END OF MODIFICATION ---


# Generate a heatmap for each emotion ---
for i, emotion in enumerate(EMOTIONS):
    ax = axes[i]
    weight_matrix = weights_by_emotion[emotion]

    # Data transformation based on chosen method
    if PLOT_METHOD == 'binning':
        num_neurons = weight_matrix.shape[1]
        num_bins = num_neurons // BIN_SIZE
        trimmed_matrix = weight_matrix[:, :num_bins * BIN_SIZE]
        plot_matrix = trimmed_matrix.reshape(weight_matrix.shape[0], num_bins, BIN_SIZE).mean(axis=2)
        ax.set_ylabel("Neuron Bin")

    elif PLOT_METHOD == 'svd':
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
        svd.fit(weight_matrix.T)
        plot_matrix = svd.components_.T
        plot_matrix = weight_matrix @ svd.components_.T # Shape: (num_layers, n_components)
        ax.set_ylabel("Principal Component")

    elif PLOT_METHOD == 'top_n_weights':
        # Select the same top N neurons (calculated before the loop) for each emotion
        plot_matrix = weight_matrix[:, top_n_indices]
        # Get the original indices for the labels on the y-axis
        y_tick_labels = top_n_indices
        ax.set_yticks(np.arange(len(y_tick_labels)))
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylabel("Top Feature Index")


    # Transpose the final matrix for plotting: Layers on X-axis
    transposed_plot_matrix = plot_matrix.T

    sns.heatmap(
        transposed_plot_matrix,
        ax=ax,
        cmap=cmap,
        cbar=False,
    )

    ax.set_title(emotion.capitalize(), fontsize=16)

axes[-1].set_xlabel("Model Layer")
fig_filename = os.path.join(FIGURE_DIR, f'weights_heatmap_{EMOTION_SCENARIO}_{fig_filename_suffix}.png')

fig.tight_layout(rect=[0, 0.05, 1, 0.96])
mappable = axes[0].get_children()[0]
cbar = fig.colorbar(
    mappable,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.08,
    shrink=0.7,
    aspect=30,
)
cbar.set_label("Weight Value", size=14)

plt.show()


# %%
