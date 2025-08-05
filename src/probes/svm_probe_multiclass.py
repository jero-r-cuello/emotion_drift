# svm_probe_multiclass.py
# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

emotion_to_test = "emotion_scenario"  # Can be "emotion_scenario" or "emotion_considered"

DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/andyzou_situations_microsoft_Phi-3-medium-128k-instruct_20250728_173814.pkl"
MODELS_DIR = "/home/jcuello/emotion_drift/models"
svm_kernel = "rbf" #!! Has to be manually changed here

# Where to store the models
MULTICLASS_PROBES_DIR = os.path.join(MODELS_DIR, f'multiclass_svm_probes')
os.makedirs(MULTICLASS_PROBES_DIR, exist_ok=True)

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
print("Starting the training and evaluation of multiclass SVM probes...")
print("="*50)

multiclass_accuracies = {}

# The main loop is over layers, as each layer gets one probe.
for layer in layer_numbers:
    X_train_layer = np.array([d["last_token_activation"][layer] for d in train_df["activations"]])
    X_test_layer = np.array([d["last_token_activation"][layer] for d in test_df["activations"]])

    y_train_multiclass = train_df[emotion_to_test].values
    y_test_multiclass = test_df[emotion_to_test].values

    probe = SVC(kernel=svm_kernel, random_state=42)
    probe.fit(X_train_layer, y_train_multiclass)

    model_filename = f'multiclass_svm_{svm_kernel}_probe_layer_{layer}_trained_on_{emotion_to_test}.joblib'
    model_path = os.path.join(MULTICLASS_PROBES_DIR, model_filename)
    joblib.dump(probe, model_path)

    predictions = probe.predict(X_test_layer)

    accuracy = accuracy_score(y_test_multiclass, predictions)
    multiclass_accuracies[layer] = accuracy

    print(f'Layer {layer:<2} evaluated')
print("\nProcess completed.")

print("\n" + "="*50)
print("MULTICLASS SVM PROBE ACCURACY REPORT")
print("="*50)
print("This shows how well each layer can distinguish between all 6 emotions.")

# Print the multiclass accuracy of each layer
for layer, acc in multiclass_accuracies.items():
    print(f'Layer {layer:<2}: {acc:.2%}')

# Find and save the best layer overall
best_layer = max(multiclass_accuracies, key=multiclass_accuracies.get)
best_accuracy = multiclass_accuracies[best_layer]

print("\n" + "-"*50)
print(f'The models internal representations are most discriminative at Layer {best_layer}')
print(f'with an overall accuracy of {best_accuracy:.2%} for distinguishing between all 6 emotions.')

print("\n" + "="*50)
# %%
