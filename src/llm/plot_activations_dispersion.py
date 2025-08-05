# plot_dispersion.py
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoConfig

# Load your data
df = pd.read_pickle("/home/jcuello/emotion_drift/data/03_activations/out_of_domain_microsoft_Phi-3-medium-128k-instruct_20250804_165130.pkl")
MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct"

# Model config and data specs
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
emotions = df["emotion_scenario"].unique()

dispersions_by_emotion = {emotion: [] for emotion in emotions}

# Loop through each row
for index, row in df.iterrows():
    emotion = row["emotion_scenario"]
    
    # Extract the (num_layers, embedding_dimension) activation matrix
    activation_matrix = np.vstack(row["activations"]["last_token_activation"].values)
    
    # Calculate the standard deviation for each layer (along the neuron axis)
    # The result is a 1D array of length num_layers
    layer_dispersions = np.std(activation_matrix, axis=1)
    
    dispersions_by_emotion[emotion].append(layer_dispersions)


# Calculate mean SD
mean_dispersions = {}
for emotion, dispersion_lists in dispersions_by_emotion.items():
    if dispersion_lists:
        mean_dispersions[emotion] = np.mean(np.array(dispersion_lists), axis=0)

# Line plot
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(16, 9))

colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))

for i, emotion in enumerate(emotions):
    if emotion in mean_dispersions:
        plt.plot(
            range(num_layers), 
            mean_dispersions[emotion], 
            label=emotion.capitalize(),
            color=colors[i],
            linewidth=2.5
        )

plt.title("Dispersion of Activations Across Layers", fontsize=20, pad=20)
plt.xlabel("Model Layer", fontsize=14)
plt.ylabel("Average Standard Deviation of Activations", fontsize=14)
plt.xticks(np.arange(0, num_layers, 2)) 
plt.xlim(0, num_layers - 1) 
plt.legend(title='Emotion', fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()

# %%
