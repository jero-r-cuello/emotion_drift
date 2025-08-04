# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 

emotion_to_test = "emotion_scenario" #!! Can be "emotion_scenario" or "emotion_considered"

DATA_PATH = "data/03_activations/andyzou_situations_microsoft_Phi-3-medium-128k-instruct_20250728_173814.pkl"
MODELS_DIR = "/home/jcuello/emotion_drift/models"

# To store the models
BINARY_PROBES_DIR = os.path.join(MODELS_DIR, 'binary_probes')
os.makedirs(BINARY_PROBES_DIR, exist_ok=True)

# Safety check to ensure the file exists
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file not found at the specified path: {DATA_PATH}")
    exit()

print(f"Loading data from {DATA_PATH}...")
nested_df = pd.read_pickle(DATA_PATH)
print("Data loaded successfully.")

# Correct 80/20 Train/Test Split from the ORIGINAL 'train' set
print("\n" + "="*50)
print("Creating a new 80/20 split from the original 'train' data pool.")
print("The original 'test' data will be ignored to prevent data leakage.")

source_df = nested_df[nested_df['split'] == 'train'].copy()
print(f"Source data for split: {len(source_df)} prompts from the original 'train' set.")

# Create a unique ID for each pair to ensure they are not split up.
source_df['pair_id'] = source_df.apply(
    lambda row: f"{row[emotion_to_test]}_{row['prompt_id'] // 2}",
    axis=1
)

unique_pair_ids = source_df['pair_id'].unique()

# Split the list of PAIR IDs into training and testing sets
train_pair_ids, test_pair_ids = train_test_split(
    unique_pair_ids,
    test_size=0.20,
    random_state=42
)

# Create the new, clean train and test DataFrames by filtering the source data
train_df = source_df[source_df['pair_id'].isin(train_pair_ids)].copy()
test_df = source_df[source_df['pair_id'].isin(test_pair_ids)].copy()

print("\nSplit verification:")
print(f"Total prompts in the new training set: {len(train_df)} ({len(train_df)/len(source_df):.0%})")
print(f"Total prompts in the new testing set:  {len(test_df)} ({len(test_df)/len(source_df):.0%})")
print("="*50)

unique_emotions = train_df[emotion_to_test].unique()
if not train_df.empty:
    layer_numbers = list(train_df.iloc[0]['activations']['last_token_activation'].keys())
else:
    print("Error: The training set is empty after the split.")
    exit()

print(f"\nEmotions to be analyzed: {list(unique_emotions)}")
print(f"{len(layer_numbers)} layers will be analyzed for each emotion.")


print("\n" + "="*50)
print("Starting the training and evaluation of linear probes...")
print("="*50)

probe_accuracies_by_layer = {emotion: {} for emotion in unique_emotions}

# Main loop for each emotion
for emotion in unique_emotions:
    print(f"\n--- Processing probes for emotion: {emotion.upper()} ---")

    emotion_train_df = train_df[train_df[emotion_to_test] == emotion]
    emotion_test_df = test_df[test_df[emotion_to_test] == emotion]

    if emotion_train_df.empty or emotion_test_df.empty:
        print(f"Skipping '{emotion}' due to missing data in the train or test set after the split.")
        continue
        
    y_train_binary = emotion_train_df['label'].values
    y_test_binary = emotion_test_df['label'].values

    # Nested loop to train one probe for each layer
    for layer in layer_numbers:
        X_train_layer = np.array([d['last_token_activation'][layer] for d in emotion_train_df['activations']])
        X_test_layer = np.array([d['last_token_activation'][layer] for d in emotion_test_df['activations']])

        probe = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        probe.fit(X_train_layer, y_train_binary)

        model_filename = f"binary_probe_{emotion}_layer_{layer}_trained_on_{emotion_to_test}.joblib"
        model_path = os.path.join(BINARY_PROBES_DIR, model_filename)
        joblib.dump(probe, model_path)

        predictions = probe.predict(X_test_layer)

        accuracy = accuracy_score(y_test_binary, predictions)
        probe_accuracies_by_layer[emotion][layer] = accuracy

    print(f"{len(layer_numbers)} probes have been trained and evaluated for '{emotion}'.")
print("\nProcess completed.")


print("\n" + "="*50)
print("LINEAR PROBE ACCURACY REPORT")
print("="*50)


best_results = {}
for emotion, layer_accuracies in probe_accuracies_by_layer.items():
    print(f"\n--- Results for: {emotion.upper()} ---")

    if not layer_accuracies:
        print("No accuracy results for this emotion.")
        continue

    for layer, acc in layer_accuracies.items():
        print(f"Layer {layer:<2}: {acc:.2%}")

    best_layer = max(layer_accuracies, key=layer_accuracies.get)
    best_accuracy = layer_accuracies[best_layer]
    best_results[emotion] = {'layer': best_layer, 'accuracy': best_accuracy}

print("\n" + "-"*50)
print("SUMMARY - BEST PERFORMING LAYER PER EMOTION")
print("-"*50)

for emotion, result in best_results.items():
    print(f"Emotion '{emotion}':")
    print(f"  -> Best performance on Layer {result['layer']} with an accuracy of {result['accuracy']:.2%}")

print("\n" + "="*50)
