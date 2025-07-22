# utils.py

import os
import json
import numpy as np
import torch 


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

def save_activations(activations_data, model_name, project_root_path, timestamp):
    """
    Saves the activations captured during the generation process to a compressed .npz file.
    The filename is based on the model name and a provided timestamp.
    """
    output_dir = os.path.join(project_root_path, "data", "03_activations")
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    
    base_filename = f"activations_{safe_model_name}_{timestamp}"
    
    filepath = os.path.join(output_dir, f"{base_filename}.npz")
    arrays_to_save = {}

    for prompt_key, data in activations_data.items():
        for layer_key, tensor in data['activations'].items():
            save_key = f"{prompt_key}_{layer_key}"
            
            # Convert tensor to float32 before saving to avoid dtype issues
            arrays_to_save[save_key] = tensor.to(torch.float32).numpy()
            
    np.savez_compressed(filepath, **arrays_to_save)
    print(f"\n[REPO INFO] Activations stored in: {filepath}\n")