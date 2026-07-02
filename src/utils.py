# utils.py

import os
import json
import glob
import pickle
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
    return group[activation_cols].set_index("layer_number").sort_index()


def pre_processing_results(run_to_load, dataset_used, save=True):
    """
    Pre-processes LLM experiment results into a final, nested DataFrame.
    """
    base_data_path = "data"
    activations_dir = os.path.join(base_data_path, "03_activations", f'activations_{run_to_load}')
    jsonl_file_path = os.path.join(base_data_path, "02_generated", f'outputs_{run_to_load}.jsonl')
    output_pickle_path = os.path.join(base_data_path, "03_activations", f'{dataset_used}_{run_to_load}.pkl')

    print(f"Loading data for run: {run_to_load}")

    # Load generated text and prompts, and data related
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    df_meta = pd.DataFrame(records)

    if "results" in df_meta["prompt_key"].values:
        df_meta = df_meta[df_meta["prompt_key"] != "results"].copy()
    df_meta["prompt_id"] = df_meta["prompt_key"].str.split("_").str[-1].astype(int)

    # Load activations
    activation_records = []
    pattern = re.compile(r"prompt_(\d+)_layer_(\d+)\.pt")
    
    activation_files = [f for f in os.listdir(activations_dir) if f.endswith(".pt")]

    for filename in tqdm(activation_files, desc="Processing Activation Files"):
        match = pattern.search(filename)
        if match:
            prompt_id, layer_id = map(int, match.groups())
            file_path = os.path.join(activations_dir, filename)
            activation_tensor = torch.load(file_path, map_location="cpu")
            activation_records.append({
                "prompt_id": prompt_id,
                "layer_number": layer_id,
                "mean_activation": activation_tensor.to(torch.float32).mean(dim=0).numpy(),
                "last_token_activation": activation_tensor[-1].to(torch.float32).numpy()
            })
            
    df_activations_long = pd.DataFrame(activation_records)

    # Create long dataframe
    df_long = pd.merge(df_meta, df_activations_long, on="prompt_id")
    print("\nIntermediate long-format DataFrame created. Now nesting...")

    # Create nested dataframe
    grouping_cols = [c for c in df_long.columns if c not in ["layer_number", "mean_activation", "last_token_activation"]]
    activation_cols = ["layer_number", "mean_activation", "last_token_activation"]
    
    grouped = df_long.groupby(grouping_cols, sort=False)
    
    final_records = []
    for name, group in tqdm(grouped, desc="Nesting DataFrames"):
        record = dict(zip(grouping_cols, name))
        record["activations"] = create_nested_df(group, activation_cols)
        
        final_records.append(record)

    df_nested = pd.DataFrame(final_records)
    print(f"\nFinal nested DataFrame created with shape: {df_nested.shape}")
    
    if save:
        os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
        df_nested.to_pickle(output_pickle_path)
        print(f"\nSuccessfully saved final nested DataFrame to: {output_pickle_path}")
        
    return df_nested


def consolidate_activations(run, dataset_used, base_dir="data", save=True,
                            activations_base=None, out_dir=None):
    """
    Build the nested-DataFrame activation pkl from the per-chunk pickles written
    by generate_with_hooks.py (the new storage format: a few dozen chunk_*.pkl
    files per run instead of millions of per-(prompt,layer) .pt files).

    Reads:
      base_dir/02_generated/outputs_<run>.jsonl                       (metadata)
      <act_base>/activations_<run>/chunk_*.pkl                         (pooled activations)
    Writes:
      <out_dir>/<dataset_used>_<run>.pkl

    `activations_base` is where the chunk pkls live (default
    base_dir/03_activations; pass the same dir given to generate_with_hooks,
    e.g. a network drive). `out_dir` is where the consolidated pkl is written
    (default = activations_base, since it can be large). Output schema matches
    the old pairing output, so the annotation-merge and probe/RSA scripts are
    unchanged: columns [prompt_id, prompt, generated_text, emotion_considered,
    label, split, activations], where `activations` is a per-prompt DataFrame
    indexed by layer_number with one column per pooling (e.g. last_token_activation).
    """
    act_base = activations_base if activations_base else os.path.join(base_dir, "03_activations")
    out_base = out_dir if out_dir else act_base
    act_dir = os.path.join(act_base, f"activations_{run}")
    jsonl_path = os.path.join(base_dir, "02_generated", f"outputs_{run}.jsonl")
    out_pkl = os.path.join(out_base, f"{dataset_used}_{run}.pkl")

    # ---- metadata ----
    meta = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pk = r.get("prompt_key")
            if not pk or pk == "results":
                continue
            r["prompt_id"] = int(str(pk).split("_")[-1])
            meta[r["prompt_id"]] = r

    # ---- pooled activations from chunk pkls (dedup by prompt_id) ----
    chunk_files = sorted(glob.glob(os.path.join(act_dir, "chunk_*.pkl")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pkl found in {act_dir}")
    acts = {}
    for cf in tqdm(chunk_files, desc="Reading activation chunks"):
        with open(cf, "rb") as f:
            acts.update(pickle.load(f))  # {prompt_id: {layer_idx: {pool: np.ndarray}}}

    # ---- build nested DataFrame ----
    rows = []
    for pid, layer_map in acts.items():
        m = meta.get(pid)
        if m is None:
            continue
        recs = []
        for layer in sorted(layer_map.keys()):
            rec = {"layer_number": layer}
            for pool, arr in layer_map[layer].items():
                rec[f"{pool}_activation"] = arr
            recs.append(rec)
        activations_df = pd.DataFrame(recs).set_index("layer_number").sort_index()
        rows.append({
            "prompt_id": pid,
            "prompt": m.get("prompt"),
            "generated_text": m.get("generated_text"),
            "emotion_considered": m.get("emotion_considered"),
            "label": m.get("label", -1),
            "split": m.get("split", "unknown"),
            "activations": activations_df,
        })

    nested_df = pd.DataFrame(rows).sort_values("prompt_id").reset_index(drop=True)
    print(f"Consolidated {len(nested_df)} prompts x "
          f"{nested_df['activations'].iloc[0].shape[0] if len(nested_df) else 0} layers.")
    if save:
        os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
        nested_df.to_pickle(out_pkl)
        print(f"Saved -> {out_pkl}")
    return nested_df


def load_dataset(dataset_name, testing=False):
    """
    Load datasets based on the provided dataset name.
    Read directly from specific CSV paths for andy_zou and emotion_query.
    If testing is True, it limits the dataset to the first 10 entries for quick runs.
    """
    df_dataset = None
    
    if dataset_name == "andyzou_situations" or dataset_name == "andy_zou":
        data_path = "data/01_stimuli/andyzou_situations_dataset/situations_emotions.csv"
        df_dataset = pd.read_csv(data_path)

    elif dataset_name == "emotion_query" or dataset_name == "xuanfengzu_emotion_query":
        data_path = "data/01_stimuli/xuanfengzu_emotion_query/emotion_query.csv"
        df_dataset = pd.read_csv(data_path)

    elif dataset_name == "generated_prompts":
        data_path = os.path.join("data", "01_stimuli", "generated_prompts", "generated_emotional_prompts_batched.csv")
        df_dataset = pd.read_csv(data_path)
        df_dataset.rename(columns={"generated_prompt":"situation","emotion_target":"emotion"}, inplace=True)
        mask_to_drop = df_dataset["situation"].str.startswith("JSON Decode Error", na=False)
        df_dataset = df_dataset[~mask_to_drop].reset_index(drop=True)

    elif dataset_name == "generated_human_prompts":
        # Style-matched 3rd-person human-centric control set, produced by
        # generate_prompts.py with PROMPT_STYLE="human_centric".
        data_path = os.path.join("data", "01_stimuli", "generated_human_prompts", "generated_human_emotional_prompts_batched.csv")
        df_dataset = pd.read_csv(data_path)
        df_dataset.rename(columns={"generated_prompt":"situation","emotion_target":"emotion"}, inplace=True)
        # Drop generation/parse failures (the generator writes these prefixes on error).
        failure_prefixes = ("JSON Decode Error", "RAW_OUTPUT:", "API_ERROR:")
        mask_to_drop = df_dataset["situation"].str.startswith(failure_prefixes, na=False)
        df_dataset = df_dataset[~mask_to_drop].reset_index(drop=True)

    elif dataset_name == "generated_human_conversation_prompts":
        # Human-content "sharing" set: a person confides an emotional human
        # situation to the assistant (addressed register, no call to action),
        # produced by generate_prompts.py with PROMPT_STYLE="human_conversation".
        data_path = os.path.join("data", "01_stimuli", "generated_human_conversation_prompts", "generated_human_conversation_prompts_batched.csv")
        df_dataset = pd.read_csv(data_path)
        df_dataset.rename(columns={"generated_prompt":"situation","emotion_target":"emotion"}, inplace=True)
        failure_prefixes = ("JSON Decode Error", "RAW_OUTPUT:", "API_ERROR:")
        mask_to_drop = df_dataset["situation"].str.startswith(failure_prefixes, na=False)
        df_dataset = df_dataset[~mask_to_drop].reset_index(drop=True)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
        
    if df_dataset is None:
        raise ValueError(f"Failed to load dataframe for dataset: {dataset_name}")

    if testing:
        df_dataset = df_dataset.head(10)
    

    situations = df_dataset["situation"].tolist()
    emotions = df_dataset["emotion"].tolist()
    
    prompt_data = [{"prompt_text": situation,
                    "emotion": emotion}
                    for emotion, situation in zip(emotions, situations)]
    
    return prompt_data



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
    
    with open(filepath, "w", encoding="utf-8") as f:
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