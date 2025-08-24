# activations_heat_map.py
# %% Raw activations plot (the problem is that the neurons are "less" activated in earlier layers)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoConfig

dataset_used = "out_of_domain"
run_to_load = "microsoft_Phi-3-medium-128k-instruct_20250804_165130"
MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct"
bin_size = 1

df = pd.read_pickle(f'/home/jcuello/emotion_drift/data/03_activations/{dataset_used}_{run_to_load}.pkl')
os.makedirs("/home/jcuello/emotion_drift/figures/activation_heat_maps", exist_ok=True)
fig_filename = f'/home/jcuello/emotion_drift/figures/activation_heat_maps/{dataset_used}_activations_{bin_size}_bins_{run_to_load}.png'

# Model configs and data specs
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
embedding_dimension = config.hidden_size
emotions = df["emotion_scenario"].unique()

# Get the mean activation over emotions (grouped by emotion)
summed_activations = {emotion: np.zeros((num_layers, embedding_dimension)) for emotion in emotions}
emotion_counts = {emotion: 0 for emotion in emotions}

for index, row in df.iterrows():
    emotion = row["emotion_scenario"]
    activation_matrix = np.vstack(row["activations"]["last_token_activation"].values)
    summed_activations[emotion] += activation_matrix
    emotion_counts[emotion] += 1

mean_activations = {}
for emotion in emotions:
    if emotion_counts[emotion] > 0:
        mean_activations[emotion] = summed_activations[emotion] / emotion_counts[emotion]

fig, axes = plt.subplots(
    nrows=len(emotions), 
    ncols=1, 
    figsize=(20, 24),
    sharex=True, 
    sharey=False
)

fig.suptitle(f'Mean Binned Activations Across Emotions (Bin Size = {bin_size})', fontsize=24)
cmap = "coolwarm" 

# Binning activations
for i, emotion in enumerate(emotions):
    ax = axes[i]
    if emotion in mean_activations:
        original_matrix = mean_activations[emotion]
        
        num_neurons = original_matrix.shape[1]
        num_bins = num_neurons // bin_size
        trimmed_matrix = original_matrix[:, :num_bins * bin_size]
        binned_matrix = trimmed_matrix.reshape(original_matrix.shape[0], num_bins, bin_size).mean(axis=2)
        
        transposed_binned_matrix = binned_matrix.T
        
        sns.heatmap(
            transposed_binned_matrix,
            ax=ax, 
            cmap=cmap,
            cbar=False 
        )
        
    ax.set_title(emotion.capitalize(), fontsize=16)
    ax.set_ylabel("Neuron Bin")
    
axes[-1].set_xlabel("Model Layer")

# Adjust the main layout to make space at the bottom for the color bar
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
mappable = axes[0].get_children()[0]

# Add the color bar, anchored to the entire figure
cbar = fig.colorbar(
    mappable, 
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.08,
    shrink=0.7,
    aspect=30,
)
cbar.set_label("Mean Activation Value", size=14)

plt.savefig(fig_filename, dpi=300)
plt.show()
# %% Normalized activations plot (each layer activations are normalized)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoConfig
import os

dataset_used = "llm_focused"
run_to_load = "Llama-2-7b-chat-hf_20250811_143357"
MODEL_NAME = "/home/models/Llama-2-7b-chat-hf"
bin_size = 1

df = pd.read_pickle(f'/home/jcuello/emotion_drift/data/03_activations/{dataset_used}_{run_to_load}.pkl')
os.makedirs("/home/jcuello/emotion_drift/figures/activation_heat_maps", exist_ok=True)
fig_filename = f'/home/jcuello/emotion_drift/figures/activation_heat_maps/{dataset_used}_activations_{bin_size}_bins_{run_to_load}.png'

df.rename(columns={"emotion_considered":"emotion_scenario"},inplace=True) #!! Easier somethimes, I have to organise the code

# Model config and data specs
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
embedding_dimension = config.hidden_size
emotions = df["emotion_scenario"].unique()

# Get the mean activation over emotions (grouped by emotion)
summed_activations = {emotion: np.zeros((num_layers, embedding_dimension)) for emotion in emotions}
emotion_counts = {emotion: 0 for emotion in emotions}

for index, row in df.iterrows():
    emotion = row["emotion_scenario"]
    activation_matrix = np.vstack(row["activations"]["last_token_activation"].values)
    summed_activations[emotion] += activation_matrix
    emotion_counts[emotion] += 1

mean_activations = {}
for emotion in emotions:
    if emotion_counts[emotion] > 0:
        mean_activations[emotion] = summed_activations[emotion] / emotion_counts[emotion]

# Normalize by layer
normalized_mean_activations = {}
for emotion, matrix in mean_activations.items():
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        layer_activations = matrix[i, :]
        layer_mean = layer_activations.mean()
        layer_std = layer_activations.std()
        epsilon = 1e-8
        normalized_matrix[i, :] = (layer_activations - layer_mean) / (layer_std + epsilon)
    normalized_mean_activations[emotion] = normalized_matrix


fig, axes = plt.subplots(
    nrows=len(emotions), 
    ncols=1, 
    figsize=(20, 24),
    sharex=True, 
    sharey=False
)

fig.suptitle(f'Layer-Normalized Mean Activations Across Emotions (Bin Size = {bin_size})', fontsize=24)
cmap = "coolwarm" 

# Binning normalized activations
for i, emotion in enumerate(emotions):
    ax = axes[i]
    if emotion in normalized_mean_activations:
        normalized_matrix = normalized_mean_activations[emotion]
        
        num_neurons = normalized_matrix.shape[1]
        num_bins = num_neurons // bin_size
        trimmed_matrix = normalized_matrix[:, :num_bins * bin_size]
        binned_matrix = trimmed_matrix.reshape(normalized_matrix.shape[0], num_bins, bin_size).mean(axis=2)
        
        transposed_binned_matrix = binned_matrix.T
        
        sns.heatmap(
            transposed_binned_matrix,
            ax=ax, 
            cmap=cmap,
            cbar=False
        )
        
    ax.set_title(emotion.capitalize(), fontsize=16)
    ax.set_ylabel("Neuron Bin")
    
axes[-1].set_xlabel("Model Layer")


# Adjust the main layout to make space at the bottom for the color bar
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
mappable = axes[0].get_children()[0]

# Add the color bar, anchored to the entire figure
cbar = fig.colorbar(
    mappable, 
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.08,
    shrink=0.7,
    aspect=30,
)
cbar.set_label("Normalized Activation (Z-score)", size=14)

plt.savefig(fig_filename, dpi=300)
plt.show()
# %%
