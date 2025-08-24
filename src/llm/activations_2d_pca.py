# %%
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Configuration ---
LLM_USED = "Llama-2-7b-chat-hf"
RUN_ID = "Llama-2-7b-chat-hf_20250811_143357"
DATASET_USED = "llm_focused"

# This should be the column with your string labels (e.g., 'joy', 'sadness')
EMOTION_COLUMN = "emotion_considered" 

# Input data path
DATA_PATH = f"/home/jcuello/emotion_drift/data/03_activations/{DATASET_USED}_{RUN_ID}.pkl"

# Output directory for the 2D plots (Updated from 3d to 2d)
PLOTS_DIR = f"/home/jcuello/emotion_drift/figures/pca_2d_scatter_plots/{DATASET_USED}_{RUN_ID}"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. Load Data ---
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file not found at '{DATA_PATH}'")
    exit()

print(f"Loading data from {DATA_PATH}...")
df = pd.read_pickle(DATA_PATH)
print("Data loaded successfully.")

# --- 2. Prepare Labels and Colors ---
if EMOTION_COLUMN not in df.columns:
    print(f"Error: Emotion column '{EMOTION_COLUMN}' not found in the DataFrame.")
    # Attempt to use the other common name as a fallback
    if "emotion_scenario" in df.columns:
        print("Found 'emotion_scenario' instead. Using that column.")
        EMOTION_COLUMN = "emotion_scenario"
    else:
        exit()

# Get the list of emotion labels for all prompts
y_labels = df[EMOTION_COLUMN]
unique_emotions = sorted(y_labels.unique())
print(f"Found {len(unique_emotions)} unique emotions: {unique_emotions}")

# Create a color palette and a mapping from emotion to color
palette = sns.color_palette("tab10", n_colors=len(unique_emotions))
color_map = {emotion: color for emotion, color in zip(unique_emotions, palette)}

# Determine the number of layers
if df.empty:
    print("Error: The DataFrame is empty.")
    exit()
num_layers = len(df['activations'].iloc[0])
print(f"Detected {num_layers} layers. Starting visualization process...")

# --- 3. Loop Through Layers, Perform PCA, and Plot ---
for layer_num in tqdm(range(num_layers), desc="Generating 2D plots for layers"):
    try:
        # a. Extract activations for the current layer from ALL prompts
        activations_list = [row['activations'].loc[layer_num, 'last_token_activation'] for _, row in df.iterrows()]
        X_layer = np.vstack(activations_list)
        
        # b. Apply PCA to reduce to 2 components (CHANGED FROM 3)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_layer)
        
        # c. Create a 2D scatter plot (CHANGED FROM 3D)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # d. Plot points, colored by emotion
        for emotion in unique_emotions:
            # Find the indices corresponding to the current emotion
            indices = (y_labels == emotion)
            
            ax.scatter(
                X_pca[indices, 0],  # PC 1
                X_pca[indices, 1],  # PC 2
                c=[color_map[emotion]],  # Use the mapped color
                label=emotion.capitalize(),
                s=20,          # Point size
                alpha=0.7      # Point transparency
            )
        
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        
        # Update title for 2D plot
        ax.set_title(
            f"PCA of Activations for Layer {layer_num}\n"
            f"({LLM_USED} on {DATASET_USED})\n"
            f"Explained Variance (2 PCs): {explained_variance:.2f}%",
            fontsize=14
        )
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        # Removed Z-axis label
        
        # e. Add a legend
        ax.legend(title="Emotion")
        ax.grid(True)
        
        # f. Save the plot with a new filename
        plot_filename = f"layer_{layer_num}_pca_2d_scatter.png"
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # IMPORTANT: Close the figure to free up memory

    except (KeyError, IndexError) as e:
        print(f"\nWarning: Could not process layer {layer_num}. Error: {e}. Skipping.")
        continue

print(f"\n--- Process complete ---")
print(f"All {num_layers} plots have been saved to: {PLOTS_DIR}")
# %%
