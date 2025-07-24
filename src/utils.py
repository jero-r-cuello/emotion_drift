# utils.py

import os
import json
import numpy as np
import torch 
import pandas as pd

def load_dataset(dataset_name, testing=False):
    """
    Load datasets based on the provided dataset name.
    Currently supports 'andyzou_situations' and 'empathetic_dialogues'.
    If testing is True, it limits the dataset to the first 10 entries for quick runs.
    """
    if dataset_name == "andyzou_situations":
        data_path = os.path.join("data", "01_stimuli", "andyzou_situations_dataset", "situations_emotions.csv")
        df_dataset = pd.read_csv(data_path)

        if testing:
            df_dataset = df_dataset.head(10)
    
    elif dataset_name == "empathetic_dialogues":
        data_path = os.path.join("data", "01_stimuli", "empatheticdialogues", "test_unique_prompts.csv")
        df_dataset = pd.read_csv(data_path)

        if testing:
            df_dataset = df_dataset.head(10)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    # This only uses the situations and emotions from the dataset.
    situations = df_dataset["situation"].tolist()
    emotions = df_dataset["emotion"].tolist()
    prompt_data = [{"prompt_text": situation,
                    "emotion": emotion}
                    for emotion, situation in zip(emotions, situations)]
    
    return prompt_data

        # This creates a combination of prompts and emotions. It's based on the LAT technique from Zou et al. (2023). Added the assessment based on SAM
#        scenarios = df_scenarios["prompt"].tolist()[:2] #!! Limiting to the first 2 prompts for testing
#        unique_emotions = df_scenarios["context"].unique().tolist()[:2] #!! Limiting to the first 2 emotions for testing
#        prompt_data = [{"prompt_text": f'USER: In a range from 1 to 9, consider the amount of {emotion} in the following scenario: "{scenario}". The amount of {emotion} is',
#                        "emotion": emotion}
#                        for emotion, scenario in itertools.product(unique_emotions, scenarios)]


def save_results_to_json(model_name, results_data, project_root_path, timestamp): 
    """
    Saves the results of the model generation to a JSON file.
    The filename is based on the model name and a provided timestamp.
    """
    output_dir = os.path.join(project_root_path, "data", "02_generated")
    os.makedirs(output_dir, exist_ok=True)
    
    safe_model_name = model_name.replace("/", "_")
    
    filename = f"outputs_{safe_model_name}_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)
    
    output_data = {
        "model_name": model_name,
        "generation_date": timestamp,
        "results": results_data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"\n[REPO INFO] Results stored in: {filepath}\n")

def save_activations(prompt_key, activations_data, activations_dir):
    """
    Now is deprecated, because the activations are saved in the vllm llama.py script.
    Just here for reference.
    Saves the activations data for a specific prompt to a compressed .npz file.
    The file is named after the prompt key and stored in the specified directory.
    """
    # Define the full path for the output .npz file
    filepath = os.path.join(activations_dir, f"{prompt_key}.npz")
    
    # Prepare the dictionary for saving, converting tensors to NumPy arrays
    arrays_to_save = {}
    for layer_key, tensor in activations_data.items():
        # Move tensor to CPU and convert to a standard float format before saving
        arrays_to_save[layer_key] = tensor.to(torch.float32).cpu().numpy()
            
    # Save the data for the current prompt to its own compressed file
    np.savez_compressed(filepath, **arrays_to_save)
    print(f"\n[REPO INFO] Activations for {prompt_key} stored in: {filepath}")