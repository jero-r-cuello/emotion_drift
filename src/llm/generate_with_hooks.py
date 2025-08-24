# generate_with_hooks.py

import multiprocessing
import os
import pandas as pd
import re
import datetime
from vllm import LLM, SamplingParams
import vllm.hook_store as hook_store
from transformers import AutoConfig
import json
import glob
from src.utils import load_dataset


def generate(model_name, target_layers, dataset_name="andyzou_situations", dataset_testing=False, resume_run=False, user_tag='[INST]', assistant_tag='[\INST]', activation_save_batch_size=64):
    """
    Generate outputs for a list of prompts, saving results incrementally and supporting resumption.
    Captures activations from specified layers and saves results to a .jsonl file.
    """
    with multiprocessing.Manager() as manager:
        hook_instructions = manager.dict()
        hook_store.ACTIVATION_HOOKS = hook_instructions
        print("\n[REPO INFO] Shared hook store initialized.\n")

        project_root = os.getcwd()
        prompt_data = load_dataset(dataset_name, testing=dataset_testing)
        
        print("\n[REPO INFO] Initializing the LLM...\n")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=4092, #!! Max of Llama-2-7b-chat-hf, but can be changed
            enforce_eager=True
        )
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256) #!! Check, maybe the default is better
        print("\n[REPO INFO] LLM initialized successfully.\n")

        home_model_dir = "/home/models/"
        if model_name.startswith(home_model_dir):
            model_name = model_name[len(home_model_dir):]

        safe_model_name = model_name.replace("/", "_")
        
        # Setup output paths
        output_dir = os.path.join(project_root, "data", "02_generated")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_prompt_keys = set()
        results_filepath = None
        
        # Check for existing output files to resume from
        if resume_run:
            search_pattern = os.path.join(output_dir, f"outputs_{safe_model_name}_*.jsonl")
            existing_files = sorted(glob.glob(search_pattern), reverse=True)
            if existing_files:
                results_filepath = existing_files[0]  # Get the latest one
                print(f"\n[REPO INFO] Found existing results file. Resuming run in: {results_filepath}\n")
                with open(results_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Load each line as a JSON object and get its key
                            record = json.loads(line)
                            processed_prompt_keys.add(record['prompt_key'])
                        except json.JSONDecodeError:
                            print(f"[WARNING] Skipping corrupted line in {results_filepath}")
                print(f"[REPO INFO] Loaded {len(processed_prompt_keys)} previously completed prompts.\n")

        # If not resuming or no file found, create a new one
        if results_filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filepath = os.path.join(output_dir, f"outputs_{safe_model_name}_{timestamp}.jsonl")
            print(f"\n[REPO INFO] Starting new run. Results will be saved to: {results_filepath}\n")
            # The corresponding activation directory should also match this timestamp
            activations_run_dir = os.path.join(project_root, "data", "03_activations", f"activations_{safe_model_name}_{timestamp}")
        else:
            # Reconstruct the activation dir path from the resumed results file
            timestamp = os.path.basename(results_filepath).replace(f"outputs_{safe_model_name}_", "").replace(".jsonl", "")
            activations_run_dir = os.path.join(project_root, "data", "03_activations", f"activations_{safe_model_name}_{timestamp}")

        os.makedirs(activations_run_dir, exist_ok=True)
        print(f"\n[REPO INFO] Activations will be saved in: {activations_run_dir}\n")

        print(f"\n[REPO INFO] Initializing generation...\n")

        # Open the results file in append mode ('a') to add new results
        with open(results_filepath, 'a', encoding='utf-8') as f_results:
            for i, data_item in enumerate(prompt_data):
                print(f"\n" + "="*80)
                print(f"Processing data item {i+1}/{len(prompt_data)}")

                prompt_to_generate = data_item["prompt_text"]
                emotion_for_this_prompt = data_item["emotion"]
                label_for_this_prompt = data_item.get("label", -1)
                split_for_this_prompt = data_item.get("split", "unknown")

                # Define the list of prompts to be processed for this data item
                prompts_to_process = []
                if dataset_name == "emotion_query" or dataset_name == "xuanfengzu_emotion_query":
                    prompts_to_process.append(
                        (f'Please answer the following query with neutrality: {prompt_to_generate}', f"prompt_{i}_neutral")
                    )
                    prompts_to_process.append(
                        (f'Please answer the following query with {emotion_for_this_prompt}: {prompt_to_generate}', f"prompt_{i}_emotional")
                    )

                else: # For now, else is andyzou, out_of_domain, and llm_focused
                    prompts_to_process.append(
                        (prompt_to_generate, f"prompt_{i}")
                    )

                for prompt, prompt_key in prompts_to_process:
                    # Check if this specific prompt has already been processed
                    if prompt_key in processed_prompt_keys:
                        print(f"--- Skipping already processed prompt: {prompt_key} ---")
                        continue

                    print(f"--- Generating for prompt_key: {prompt_key} ---")
                    
                    hook_instructions['save_path'] = activations_run_dir
                    hook_instructions['prompt_key'] = prompt_key
                    hook_instructions['target_layers'] = target_layers
                    hook_instructions['activation_save_batch_size'] = activation_save_batch_size

                    template_str = '{user_tag} {scenario} {assistant_tag}'
                    prompt = template_str.format(scenario=prompt, user_tag=user_tag, assistant_tag=assistant_tag)
                    output = llm.generate(prompt, sampling_params)[0]
            
                    hook_instructions.clear()

                    prompt_text = output.prompt
                    generated_text = output.outputs[0].text
            
                    print(f"\nPROMPT:\n{prompt_text}")
                    print(f"\nGENERATED TEXT:\n{generated_text}")
                    print(f"\n[REPO INFO] Activations of {prompt_key} were saved by vLLM workers.")

                    # Create the result item, ensuring a consistent 'prompt_key'
                    if dataset_name == "emotion_query" or dataset_name == "xuanfengzu_emotion_query":
                        result_item = {"prompt_key": prompt_key,
                                       "prompt_id": prompt_key, # Maintain old field for compatibility if needed
                                       "prompt": prompt_text,
                                       "generated_text": generated_text,
                                       "text_emotion": emotion_for_this_prompt}
                        
                    if dataset_name == "out_of_domain":
                        result_item = {"prompt_key": prompt_key,
                                       "prompt_id": prompt_key,
                                       "prompt": prompt_text,
                                       "generated_text": generated_text,
                                       "emotion_scenario": emotion_for_this_prompt}
                        
                    else: # For now, else is andyzou
                        result_item = {"prompt_key": prompt_key,
                                       "prompt": prompt_text,
                                       "generated_text": generated_text,
                                       "emotion_considered": emotion_for_this_prompt,
                                       "label": label_for_this_prompt,
                                       "split": split_for_this_prompt}

                    # Write the result as a new line in the JSONL file and flush
                    f_results.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f_results.flush()

        print("\n[REPO INFO] Generation complete. All results have been saved.\n")

        return f'{safe_model_name}_{timestamp}' # Safe name of the run
        
if __name__ == "__main__":
    MODEL_NAME = "/home/models/Meta-Llama-3-8B" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    
    TARGET_LAYERS = [l for l in range(num_layers)] # List of layers to observe.

    generate(model_name=MODEL_NAME,
             target_layers=TARGET_LAYERS,
             dataset_name="andyzou_rep_eng", #!! Change this to the dataset you want to use.
             dataset_testing=True,
             resume_run=False)