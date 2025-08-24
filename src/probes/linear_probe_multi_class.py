# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 
import matplotlib.pyplot as plt

emotion_to_test = "emotion_considered" #!! Can be "emotion_scenario" or "emotion_considered"
LLM_USED = "Meta-Llama-3-8B"
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/llm_focused_Llama-2-7b-chat-hf_20250811_143357.pkl"
MODELS_DIR = "/home/jcuello/emotion_drift/models"

# To store the models
MULTICLASS_PROBES_DIR = os.path.join(MODELS_DIR, "multiclass_probes")
os.makedirs(MULTICLASS_PROBES_DIR, exist_ok=True)

# Safety check to ensure the file exists
if not os.path.exists(DATA_PATH):
    print(f'Error: Data file not found at the specified path: {DATA_PATH}')
    exit()

print(f'Loading data from {DATA_PATH}...')
nested_df = pd.read_pickle(DATA_PATH)
print("Data loaded successfully.")

# Correct 80/20 Train/Test Split from the ORIGINAL 'train' set
print("\n" + "="*50)
print("Creating a new 80/20 split from the original 'train' data pool.")

source_df = nested_df[nested_df["split"] == "train"].copy()
print(f'Source data for split: {len(source_df)} prompts from the original "train" set.')

# Create a unique ID for each pair to ensure they are not split up.
source_df["pair_id"] = source_df.apply(
    lambda row: f'{row[emotion_to_test]}_{row["prompt_id"] // 2}',
    axis=1
)

unique_pair_ids = source_df["pair_id"].unique()

# Split the list of PAIR IDs into training and testing sets
train_pair_ids, test_pair_ids = train_test_split(
    unique_pair_ids,
    test_size=0.20,
    random_state=42
)

# Create the new, clean train and test DataFrames by filtering the source data
train_df = source_df[source_df["pair_id"].isin(train_pair_ids)].copy()
test_df = source_df[source_df["pair_id"].isin(test_pair_ids)].copy()

print("\nSplit verification:")
print(f'Total prompts in the new training set: {len(train_df)} ({len(train_df)/len(source_df):.0%})')
print(f'Total prompts in the new testing set:  {len(test_df)} ({len(test_df)/len(source_df):.0%})')
print("="*50)

unique_emotions = train_df[emotion_to_test].unique()
layer_numbers = list(train_df.iloc[0]["activations"]["last_token_activation"].keys())

print(f'\nTask: Multiclass classification among {len(unique_emotions)} emotions. Testing {emotion_to_test}')
print(f'{len(layer_numbers)} layers will be analyzed.')


print("\n" + "="*50)
print("Starting the training and evaluation of multiclass linear probes...")
print("="*50)

multiclass_accuracies = {}

# The main loop is over layers, as each layer gets one probe.
for layer in layer_numbers:    
    X_train_layer = np.array([d["last_token_activation"][layer] for d in train_df["activations"]])
    X_test_layer = np.array([d["last_token_activation"][layer] for d in test_df["activations"]])
    
    y_train_multiclass = train_df[emotion_to_test].values
    y_test_multiclass = test_df[emotion_to_test].values
    
    probe = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs") #!! Every parameter could be different!
    probe.fit(X_train_layer, y_train_multiclass)
    
    model_filename = f'{LLM_USED}_multiclass_probe_layer_{layer}_trained_on_{emotion_to_test}.joblib'
    model_path = os.path.join(MULTICLASS_PROBES_DIR, model_filename)
    joblib.dump(probe, model_path)

    predictions = probe.predict(X_test_layer)
    
    accuracy = accuracy_score(y_test_multiclass, predictions)
    multiclass_accuracies[layer] = accuracy
    
    print(f'Layer {layer:<2} evaluated')
print("\nProcess completed.")


print("\n" + "="*50)
print("MULTICLASS PROBE ACCURACY REPORT")
print("="*50)
print("This shows how well each layer can distinguish between all 6 emotions.")

# Print the multiclass accuracy of each layer
for layer, acc in multiclass_accuracies.items():
    print(f'Layer {layer:<2}: {acc:.2%}')

# Find and save the best layer overall
best_layer = max(multiclass_accuracies, key=multiclass_accuracies.get)
best_accuracy = multiclass_accuracies[best_layer]

print("\n" + "-"*50)
print(f"The model's internal representations are most discriminative at Layer {best_layer}")
print(f'with an overall accuracy of {best_accuracy:.2%} for distinguishing between all 6 emotions.')

print("\n" + "="*50)
#%% Training stuff. Plots the accuracy in a validation set for each 

print(f'\nGenerating learning curve for the best performing layer: Layer {best_layer}\n')

# 1. Prepare the full dataset for this layer
# We use the full source_df because the learning_curve function handles its own cross-validation splits
X_full = np.array([d["last_token_activation"][best_layer] for d in source_df["activations"]])
y_full = source_df[emotion_to_test].values

# 2. Define the model to be evaluated
# It's crucial to use the exact same model parameters
estimator = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

# 3. Use the learning_curve utility
# We will test the model on training sets of 10%, 30%, 50%, 70%, and 100% of the data.
# `cv=5` means it will use 5-fold cross-validation to get a smoother, more robust result.
train_sizes_abs, train_scores, test_scores = learning_curve(
    estimator,
    X_full,
    y_full,
    cv=5, # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1, # Use all available CPU cores
    train_sizes=np.linspace(0.05, 1.0, 20) # 20 steps from 5% to 100% of the data
)

# 4. Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 5. Plot the learning curve
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 6))

plt.title(f'Learning Curve for Multiclass Probe (Layer {best_layer})', fontsize=16)
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")

# Plot the mean accuracy lines
plt.plot(train_sizes_abs, train_scores_mean, "o-", color="blue", label="Training score")
plt.plot(train_sizes_abs, test_scores_mean, "o-", color="green", label="Cross-validation score")

# Plot the standard deviation as a shaded area (confidence band)
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="green")

plt.legend(loc="best")
plt.grid(True)
plt.savefig("/home/jcuello/emotion_drift/figures/multi_class_probe_evaluation/learning_curve_best_training_layer.png",
            dpi=300)
plt.show()
# %%
