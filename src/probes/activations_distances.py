#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# --- Aesthetic Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# --- Configuration ---
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL_WITH_RATINGS_AND_CATS.pkl"
TAXONOMY_TARGET = 'ekman_basic_emotions' 
BASE_DIR = "/home/jcuello/emotion_drift"
PLOTS_DIR = os.path.join(BASE_DIR, "figures", "cosine_distance_histograms_gpu")

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Logic: Using {device}")
if device.type == 'cpu':
    print("WARNING: Running on CPU. This will be slow for 21k points.")

# --- 1. Load Data ---
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f'{DATA_PATH} not found')
print(f"Loading data from {DATA_PATH}...")
nested_df_original = pd.read_pickle(DATA_PATH)

# Filter empty rows
df = nested_df_original.copy()
mask = df[TAXONOMY_TARGET].apply(lambda x: isinstance(x, list) and len(x) > 0)
df = df[mask].copy()

# Extract labels
df['label_str'] = df[TAXONOMY_TARGET].str[0]

# --- 2. Global Prep ---
unique_labels = sorted(df['label_str'].unique())
print(f"Labels found: {unique_labels}")

# Detect layers
first_activations = df['activations'].iloc[0]
try:
    num_layers = len(first_activations)
    print(f"Detected {num_layers} layers.")
except:
    num_layers = 33
    print(f"Warning: Could not detect layers, defaulting to {num_layers}")

# Paper Colors (Blue for Within, Orange for Between)
COLOR_WITHIN = '#4c72b0' 
COLOR_BETWEEN = '#dd8452' 

# --- 3. Layer Loop ---
print(f"\nStarting GPU-Accelerated Distance Calculation...")

for layer_idx in tqdm(range(num_layers), desc="Processing Layers"):
    
    # --- A. Extract Activations (CPU) ---
    # We do this extraction on CPU first because it requires iterating over objects
    X_list = []
    y_list = []
    
    for act_row, label in zip(df['activations'], df['label_str']):
        try:
            # Extract specific layer
            if isinstance(act_row, (pd.Series, pd.DataFrame)):
                act = act_row.iloc[layer_idx]['last_token_activation']
            else:
                act = act_row[layer_idx]['last_token_activation']
            
            if not isinstance(act, np.ndarray): continue
            if act.ndim > 1: act = act.squeeze()
            
            # Shape validation (e.g., 4096 for Llama-2-7b)
            if act.shape[0] == 4096:
                X_list.append(act)
                y_list.append(label)
        except Exception:
            continue
            
    if not X_list:
        continue

    # Convert to Numpy first
    X_np = np.stack(X_list)
    y_np = np.array(y_list)

    # --- B. GPU Calculation Logic ---
    # We wrap in no_grad to save memory (we aren't training)
    with torch.no_grad():
        
        # 1. Move Data to GPU
        # Convert to float32 tensor
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        
        # 2. Pre-Normalize Vectors
        # Cosine Distance = 1 - (A . B) / (|A|*|B|)
        # If we normalize A -> A / |A|, then Cosine Distance = 1 - (A . B)
        # 1e-8 is for numerical stability
        X_norm = X_tensor / (X_tensor.norm(dim=1, keepdim=True) + 1e-8)
        
        # --- C. Plotting Setup ---
        # Calculate grid size
        n_labels = len(unique_labels)
        cols = 3
        rows = (n_labels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
        axes = axes.flatten()
        fig.suptitle(f"Cosine Distance Distributions - Layer {layer_idx}\n(Blue: Within-Class | Orange: Between-Class)", fontsize=16)

        # --- D. Emotion Loop ---
        for i, target_emotion in enumerate(unique_labels):
            ax = axes[i]
            
            # Create boolean masks (on CPU, used to slice GPU tensor)
            mask_target = (y_np == target_emotion)
            mask_others = (y_np != target_emotion)
            
            # Slice GPU Tensors
            vecs_target = X_norm[mask_target]
            vecs_others = X_norm[mask_others]
            
            if len(vecs_target) < 2:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                ax.set_title(target_emotion)
                continue

            # ---------------------------------------------------------
            # CALCULATION 1: WITHIN-CLASS (Target vs Target)
            # ---------------------------------------------------------
            # Matrix Mult: (N, D) @ (D, N) -> (N, N) similarity matrix
            sim_matrix_within = torch.mm(vecs_target, vecs_target.T)
            
            # Convert to distance: 0 (Same) -> 2 (Opposite)
            dist_matrix_within = 1.0 - sim_matrix_within
            
            # Extract upper triangle (excluding diagonal) to get unique pairs
            # triu_indices returns indices for upper triangle offset by 1
            r_idx, c_idx = torch.triu_indices(len(vecs_target), len(vecs_target), offset=1, device=device)
            
            # Extract values and move to CPU for plotting
            within_dists = dist_matrix_within[r_idx, c_idx].cpu().numpy()

            # ---------------------------------------------------------
            # CALCULATION 2: BETWEEN-CLASS (Target vs Others)
            # ---------------------------------------------------------
            between_dists = []
            if len(vecs_others) > 0:
                # Matrix Mult: (N_target, D) @ (D, N_others) -> (N_target, N_others)
                sim_matrix_between = torch.mm(vecs_target, vecs_others.T)
                dist_matrix_between = 1.0 - sim_matrix_between
                
                # Flatten the matrix to a 1D list of distances
                flat_between = dist_matrix_between.flatten()
                
                # OPTIMIZATION: Subsample if too large
                # Plotting histograms with >1M points is slow on CPU.
                # We subsample randomly ON GPU before moving data.
                MAX_POINTS = 1_000_000
                if flat_between.numel() > MAX_POINTS:
                    # Generate random indices
                    perm = torch.randperm(flat_between.numel(), device=device)[:MAX_POINTS]
                    between_dists = flat_between[perm].cpu().numpy()
                else:
                    between_dists = flat_between.cpu().numpy()

            # ---------------------------------------------------------
            # PLOTTING
            # ---------------------------------------------------------
            # Fixed bins for consistent comparison (0 to 1.5 covers most cosine distances)
            bins = np.linspace(0.0, 1.5, 60)
            
            # Plot Between (Context) first
            if len(between_dists) > 0:
                ax.hist(between_dists, bins=bins, density=True, 
                        color=COLOR_BETWEEN, alpha=0.6, label='Between-Class')
                
            # Plot Within (Focus) second
            if len(within_dists) > 0:
                ax.hist(within_dists, bins=bins, density=True, 
                        color=COLOR_WITHIN, alpha=0.7, label='Within-Class')

            ax.set_title(target_emotion, fontweight='bold')
            ax.set_xlabel("Cosine Distance")
            ax.set_ylabel("Density")
            ax.set_xlim(0, 1.3) # Trim x-axis for visual clarity
            
            if i == 0:
                ax.legend()

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Save Plot
        filename = f"{TAXONOMY_TARGET}_cosine_hist_gpu_layer_{layer_idx:02d}.png"
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=150)
        plt.close() # Free memory

        # Clean up GPU memory for next layer
        del X_tensor, X_norm, vecs_target, vecs_others
        torch.cuda.empty_cache()

print(f"\nProcessing complete. Figures saved to:\n{PLOTS_DIR}")
#%%