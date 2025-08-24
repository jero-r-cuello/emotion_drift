# utils.py

import os
import json
import numpy as np
import torch 
import pandas as pd
import random
from tqdm import tqdm
import re


def create_nested_df(group, activation_cols):
    """
    Helper function to transform a group of layer data into a single,
    nested DataFrame indexed by layer number.
    """
    return group[activation_cols].set_index('layer_number').sort_index()


def pre_processing_results(run_to_load, dataset_used, save=True):
    """
    Pre-processes LLM experiment results into a final, nested DataFrame.
    """
    base_data_path = 'data'
    activations_dir = os.path.join(base_data_path, '03_activations', f'activations_{run_to_load}')
    jsonl_file_path = os.path.join(base_data_path, '02_generated', f'outputs_{run_to_load}.jsonl')
    output_pickle_path = os.path.join(base_data_path, '03_activations', f'{dataset_used}_{run_to_load}.pkl')

    print(f"Loading data for run: {run_to_load}")

    # Load generated text and prompts, and data related
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    df_meta = pd.DataFrame(records)

    if "results" in df_meta["prompt_key"].values:
        df_meta = df_meta[df_meta["prompt_key"] != "results"].copy()
    df_meta['prompt_id'] = df_meta['prompt_key'].str.split('_').str[-1].astype(int)

    # Load activations
    activation_records = []
    pattern = re.compile(r'prompt_(\d+)_layer_(\d+)\.pt')
    
    activation_files = [f for f in os.listdir(activations_dir) if f.endswith('.pt')]

    for filename in tqdm(activation_files, desc="Processing Activation Files"):
        match = pattern.search(filename)
        if match:
            prompt_id, layer_id = map(int, match.groups())
            file_path = os.path.join(activations_dir, filename)
            activation_tensor = torch.load(file_path, map_location='cpu')
            activation_records.append({
                'prompt_id': prompt_id,
                'layer_number': layer_id,
                'mean_activation': activation_tensor.to(torch.float32).mean(dim=0).numpy(),
                'last_token_activation': activation_tensor[-1].to(torch.float32).numpy()
            })
            
    df_activations_long = pd.DataFrame(activation_records)

    # Create long dataframe
    df_long = pd.merge(df_meta, df_activations_long, on='prompt_id')
    print("\nIntermediate long-format DataFrame created. Now nesting...")

    # Create nested dataframe
    grouping_cols = [c for c in df_long.columns if c not in ['layer_number', 'mean_activation', 'last_token_activation']]
    activation_cols = ['layer_number', 'mean_activation', 'last_token_activation']
    
    grouped = df_long.groupby(grouping_cols, sort=False)
    
    final_records = []
    for name, group in tqdm(grouped, desc="Nesting DataFrames"):
        record = dict(zip(grouping_cols, name))
        record['activations'] = create_nested_df(group, activation_cols)
        
        final_records.append(record)

    df_nested = pd.DataFrame(final_records)
    print(f"\nFinal nested DataFrame created with shape: {df_nested.shape}")
    
    if save:
        os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
        df_nested.to_pickle(output_pickle_path)
        print(f"\nSuccessfully saved final nested DataFrame to: {output_pickle_path}")
        
    return df_nested

def load_andyzou_situations_dataset(data_dir, user_tag='[INST]', assistant_tag='[\INST]'):
    """
    Loads the Andy Zou Situations dataset and formats it for training and testing.
    Extracted from https://github.com/andyzoujm/representation-engineering/blob/main/examples/primary_emotions/utils.py
    """
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200] #!! This is done in the original paper too, but its not clear why. Seems arbitrary

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


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

    elif dataset_name == "emotion_query" or dataset_name == "xuanfengzu_emotion_query":
        data_path = os.path.join("data", "01_stimuli", "xuanfengzu_emotion_query", "emotion_query.csv")
        df_dataset = pd.read_csv(data_path)

        if testing:
            df_dataset = df_dataset.head(10)

    elif dataset_name == "andyzou_rep_eng":        
        data_dir = os.path.join("data", "01_stimuli", "andyzou_situations_dataset")
        formatted_data = load_andyzou_situations_dataset(data_dir)

        prompt_data = []
        # Iterate through each emotion (e.g., "happiness")
        for emotion, splits in formatted_data.items():
            # Now, iterate through the splits themselves ('train' and 'test')
            for split_name, split_data in splits.items():
                
                # --- LOGIC FOR THE TRAIN SPLIT ---
                # The train split has shuffled pairs, so its labeling logic is more complex.
                if split_name == 'train':
                    prompts = split_data['data']
                    label_pairs = split_data['labels'] # List of [bool, bool] pairs

                    for i, label_pair in enumerate(label_pairs):
                        prompt1_idx = 2 * i
                        prompt2_idx = 2 * i + 1

                        if prompt1_idx < len(prompts) and prompt2_idx < len(prompts):
                            prompt_data.append({
                                "prompt_text": prompts[prompt1_idx],
                                "emotion": emotion,
                                "label": int(label_pair[0]), # Convert True/False to 1/0
                                "split": split_name # Add split info for easy filtering later
                            })
                            prompt_data.append({
                                "prompt_text": prompts[prompt2_idx],
                                "emotion": emotion,
                                "label": int(label_pair[1]),
                                "split": split_name
                            })
                
                # --- LOGIC FOR THE TEST SPLIT ---
                # The test split is not shuffled, so its labeling is a direct 1-to-1 mapping.
                elif split_name == 'test':
                    prompts = split_data['data']
                    # The original labels are in a nested list like [[1, 0, 1, 0...]], so we take the first element.
                    flat_labels = split_data['labels'][0]

                    for i, prompt_text in enumerate(prompts):
                        # Simple direct mapping
                        label = flat_labels[i]
                        prompt_data.append({
                            "prompt_text": prompt_text,
                            "emotion": emotion,
                            "label": label,
                            "split": split_name
                        })

        if testing:
            return prompt_data[:10]
        

        return prompt_data
    elif dataset_name == "out_of_domain":
        data_path = os.path.join("data", "01_stimuli", "testing_out_of_domain", "out_of_domain.csv")
        df_dataset = pd.read_csv(data_path)

        if testing:
            df_dataset = df_dataset.head(10)

    elif dataset_name == "llm_focused":
        data_path = os.path.join("data", "01_stimuli", "llm_focused_situations", "llm_focused_situations.csv")
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


def save_results_to_json(model_name, dataset_used, results_data, project_root_path, timestamp): 
    """
    Now is deprecated, because the results are saved on real time when generated.
    Just here for reference.
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
        "dataset_used": dataset_used,
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