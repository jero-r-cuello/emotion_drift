"""
Consolidate a generation run's activations + metadata into the nested-DataFrame
pkl that the annotation-merge and probe/RSA scripts consume.

Activations are now produced by generate_with_hooks.py as a few per-chunk
pickles (chunk_*.pkl) of pooled vectors, not millions of per-(prompt,layer) .pt
files, so this step just calls utils.consolidate_activations. Output:
    data/03_activations/<DATASET_USED>_<RUN_TO_LOAD>.pkl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import consolidate_activations

# ================= CONFIG =================
RUN_TO_LOAD = "Qwen2.5-14B-Instruct_YYYYMMDD_HHMMSS"   # run id printed by generate_with_hooks / run_dp
DATASET_USED = "generated_human_prompts"               # for the output filename / provenance

if __name__ == "__main__":
    consolidate_activations(RUN_TO_LOAD, DATASET_USED, base_dir="data", save=True)
