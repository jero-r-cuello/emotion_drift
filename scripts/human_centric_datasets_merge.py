import pandas as pd
import os
from datetime import datetime

# Rutas originales
path_az = "/home/jcuello/emotion_drift/data/03_activations/andyzou_situations_Llama-2-7b-chat-hf_20260127_151151_FINAL.pkl"
path_eq = "/home/jcuello/emotion_drift/data/03_activations/emotion_query_Llama-2-7b-chat-hf_20260127_151453_FINAL.pkl"

print("Loading datasets")
az_df = pd.read_pickle(path_az)
eq_df = pd.read_pickle(path_eq)

print(f"Rows Andy Zou: {len(az_df)}")
print(f"Rows Emotion Query: {len(eq_df)}")

az_df['dataset_source'] = 'andy_zou'
eq_df['dataset_source'] = 'emotion_query'

merged_df = pd.concat([az_df, eq_df], ignore_index=True)

output_dir = "/home/jcuello/emotion_drift/data/03_activations/"
filename = f"MERGED_andyzou_emotion_query_Llama-2-7b-chat-hf_FINAL.pkl"
output_path = os.path.join(output_dir, filename)

merged_df.to_pickle(output_path)

print(f"\nFile saved succesfully in:\n{output_path}")

# Quick check
print(merged_df[['dataset_source', 'prompt']].head())
print(merged_df[['dataset_source', 'prompt']].tail())