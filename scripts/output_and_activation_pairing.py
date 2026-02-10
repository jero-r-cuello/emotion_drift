import os
import sys
import json
import re
import gc
import shutil
import time
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= CONFIGURACIÓN =================
RUN_TO_LOAD = "Llama-2-7b-chat-hf_20260127_151151" #"Llama-2-7b-chat-hf_20260127_151453" #"Qwen2.5-14B-Instruct_20251220_225401"#"Llama-2-7b-chat-hf_20251014_203636"#
DATASET_USED = "andyzou_situations" # "emotion_query" # "generated_prompts" # 

BASE_DIR = "/home/jcuello/emotion_drift/data"
METADATA_PATH = os.path.join(BASE_DIR, "02_generated", f"outputs_{RUN_TO_LOAD}.jsonl")
ACTIVATIONS_DIR = os.path.join(BASE_DIR, "03_activations", f"activations_{RUN_TO_LOAD}")
OUTPUT_DIR = os.path.join(BASE_DIR, "03_activations")
TEMP_CHUNK_DIR = os.path.join(BASE_DIR, "03_activations", "temp_chunks_v2")

# OPTIMIZACIÓN 4: Chunks más pequeños (1000 prompts)
# Esto genera archivos temporales de aprox 7GB en lugar de 35GB, evitando picos de uso de disco.
CHUNK_SIZE = 1000

def load_metadata(jsonl_path):
    print(f"--> Cargando metadatos desde: {jsonl_path}")
    outputs = defaultdict(dict)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"No se encontró el archivo: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                prompt_key = record.pop('prompt_key', None)
                if prompt_key:
                    outputs[prompt_key] = record
            except json.JSONDecodeError:
                continue
    print(f"    Total de registros: {len(outputs)}")
    return outputs

def index_activation_files(act_dir):
    print(f"--> Indexando archivos en: {act_dir}")
    activation_paths = defaultdict(list)
    pattern = re.compile(r'prompt_(\d+)_layer_(\d+)\.pt')
    
    with os.scandir(act_dir) as entries:
        for entry in tqdm(entries, desc="Indexando rutas"):
            if entry.is_file() and entry.name.endswith('.pt'):
                match = pattern.match(entry.name)
                if match:
                    prompt_id = int(match.group(1))
                    layer_id = int(match.group(2))
                    activation_paths[prompt_id].append((layer_id, entry.path))

    for pid in activation_paths:
        activation_paths[pid].sort(key=lambda x: x[0])
        
    print(f"    Indexado completado. {len(activation_paths)} prompts únicos.")
    return activation_paths

def calculate_metrics(tensor, layer_idx, prompt_data):
    # Mantenemos float32 como pediste
    t_f32 = tensor.to(torch.float32)
    
    def to_np(t): return t.detach().cpu().numpy()

    # 1. Mean Pooling
    mean_act = t_f32.mean(dim=0)
    prompt_data[f'layer_{layer_idx}_mean'] = to_np(mean_act)
    
    # 2. Last-Token Pooling
    last_act = t_f32[-1]
    prompt_data[f'layer_{layer_idx}_last_token'] = to_np(last_act)
    
    # 3. Max Pooling
    max_act = t_f32.max(dim=0).values
    prompt_data[f'layer_{layer_idx}_max'] = to_np(max_act)
    
    # 4. Min Pooling
    min_act = t_f32.min(dim=0).values
    prompt_data[f'layer_{layer_idx}_min'] = to_np(min_act)
        
    # 5. Attention Mean Pooling (AMP)
    token_magnitudes = t_f32.mean(dim=1) 
    importance_scores = torch.softmax(token_magnitudes, dim=0)
    amp_act = (importance_scores[:, None] * t_f32).sum(dim=0)
    prompt_data[f'layer_{layer_idx}_amp'] = to_np(amp_act)

def process_and_chunk(activation_paths):
    # Limpieza proactiva de carpeta temporal
    if os.path.exists(TEMP_CHUNK_DIR):
        shutil.rmtree(TEMP_CHUNK_DIR)
    os.makedirs(TEMP_CHUNK_DIR)
    
    sorted_prompt_ids = sorted(activation_paths.keys())
    chunked_data = []
    chunk_counter = 0
    
    print(f"--> Iniciando procesamiento de {len(sorted_prompt_ids)} prompts...")
    
    for prompt_id in tqdm(sorted_prompt_ids, desc="Procesando tensores"):
        prompt_data = {'prompt_id': prompt_id}
        layers_info = activation_paths[prompt_id]
        
        valid_prompt = True
        
        for layer_idx, file_path in layers_info:
            try:
                # Carga en CPU
                tensor = torch.load(file_path, map_location='cpu')
                calculate_metrics(tensor, layer_idx, prompt_data)
                del tensor # Liberar RAM
            except Exception as e:
                print(f"\n[ERROR] Fallo en {file_path}: {e}", file=sys.stderr)
                valid_prompt = False
                break
        
        if valid_prompt:
            chunked_data.append(prompt_data)
        
        # Guardar si alcanzamos el tamaño del chunk (1000)
        if len(chunked_data) >= CHUNK_SIZE:
            save_chunk(chunked_data, chunk_counter)
            chunked_data = []
            chunk_counter += 1
            gc.collect()
            
    # Guardar último pedazo
    if chunked_data:
        save_chunk(chunked_data, chunk_counter)
        chunk_counter += 1
        
    return chunk_counter

def save_chunk(data, counter):
    df = pd.DataFrame(data)
    file_path = os.path.join(TEMP_CHUNK_DIR, f"chunk_{counter:04d}.pkl")
    # Pickle estándar (sin compresión extra)
    df.to_pickle(file_path)

def merge_chunks_and_metadata(num_chunks, metadata_dict):
    print(f"--> Fusionando {num_chunks} chunks temporales...")
    
    files = [os.path.join(TEMP_CHUNK_DIR, f) for f in os.listdir(TEMP_CHUNK_DIR) if f.endswith('.pkl')]
    files.sort()
    
    if not files:
        raise RuntimeError("No hay chunks para fusionar.")
        
    dfs = []
    for f in tqdm(files, desc="Leyendo chunks"):
        dfs.append(pd.read_pickle(f))
        
    df_activations = pd.concat(dfs, ignore_index=True)
    
    print("    Limpiando archivos temporales...")
    shutil.rmtree(TEMP_CHUNK_DIR)
    
    print("--> Preparando Metadatos...")
    df_meta = pd.DataFrame(metadata_dict).T
    if "results" in df_meta.index:
        df_meta = df_meta[df_meta.index != "results"]
        
    df_meta["prompt_id"] = [int(i.split('_')[-1]) for i in df_meta.index]
    
    print("--> Realizando Merge...")
    df_final = pd.merge(df_meta, df_activations, on='prompt_id')
    return df_final

def reshape_and_nest(df_wide):
    print("--> Reshaping (Wide to Long)...")
    
    df_renamed = df_wide.copy()
    
    regex_pattern = r'layer_(\d+)_(mean|last_token|max|min|amp)'
    df_renamed.columns = df_renamed.columns.str.replace(regex_pattern, r'\2_\1', regex=True)
    
    stubnames = ['mean', 'last_token', 'max', 'min', 'amp']
    id_vars = ["prompt_id", 'prompt', 'generated_text', 'emotion_considered', 'label', 'split']
    
    existing_id_vars = [c for c in id_vars if c in df_renamed.columns]
    
    long_df = pd.wide_to_long(
        df_renamed,
        stubnames=stubnames,
        i=existing_id_vars,
        j='layer_number',
        sep='_',
        suffix=r'\d+'
    ).reset_index()
    
    long_df = long_df.rename(columns={
        'mean': 'mean_activation',
        'last_token': 'last_token_activation',
        'max': 'max_activation',
        'min': 'min_activation',
        'amp': 'amp_activation'
    })
    
    print("--> Nesting...")
    
    activation_cols = ['mean_activation', 'last_token_activation', 'max_activation', 'min_activation', 'amp_activation']
    nested_cols = ['layer_number'] + activation_cols
    
    grouped = long_df.groupby(existing_id_vars)
    final_data_list = []
    
    for group_name, group_df in tqdm(grouped, desc="Anidando"):
        activations_df = group_df[nested_cols].copy().set_index('layer_number')
        activations_df.sort_index(inplace=True)
        
        if not isinstance(group_name, tuple):
            group_name = (group_name,)
        new_row = list(group_name) + [activations_df]
        final_data_list.append(new_row)
        
    nested_df = pd.DataFrame(final_data_list, columns=existing_id_vars + ['activations'])
    return nested_df

def main():
    start_time = time.time()
    print("=== PROCESAMIENTO OPTIMIZADO (Sin Concat MMM / Chunks Pequeños) ===")
    
    try:
        metadata = load_metadata(METADATA_PATH)
        act_paths = index_activation_files(ACTIVATIONS_DIR)
        
        # Procesar
        num_chunks = process_and_chunk(act_paths)
        
        if num_chunks > 0:
            df_wide = merge_chunks_and_metadata(num_chunks, metadata)
            df_final = reshape_and_nest(df_wide)
            
            output_file = os.path.join(OUTPUT_DIR, f"{DATASET_USED}_{RUN_TO_LOAD}.pkl")
            print(f"--> Guardando archivo final en: {output_file}")
            df_final.to_pickle(output_file)
            
            elapsed = time.time() - start_time
            print(f"Hecho en {elapsed/60:.2f} min.")
            
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
        
        # Limpieza de emergencia
        if os.path.exists(TEMP_CHUNK_DIR):
            print("Limpiando carpetas temporales...")
            shutil.rmtree(TEMP_CHUNK_DIR)

if __name__ == "__main__":
    main()