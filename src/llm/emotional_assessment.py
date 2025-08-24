import os
import pandas as pd
import re
import datetime
from vllm import LLM, SamplingParams
from src.utils import load_dataset
import glob
import json


def assess(model_name, dataset_name, dataset_testing=False, assessment_to_use="panas", resume_run=False, user_tag='[INST]', assistant_tag='[\INST]'):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    
    project_root = os.getcwd()
    prompt_data = load_dataset(dataset_name, testing=dataset_testing)

    print(f"\n[REPO INFO] Initializing the LLM for {assessment_to_use} assessment...\n")

    assessments_path = os.path.join("data", "01_stimuli", "assessments", "emotion_assessments.json")
    with open(assessments_path, 'r', encoding='utf-8') as f:
        assessments_dict = json.load(f)

    for assessment in assessments_dict:
        if assessment.get('name') == assessment_to_use:
            assessment_prompt = assessment.get('prompt')
            break 

    llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=4092, #!! Max of Llama-2-7b-chat-hf, but can be changed
        )
    
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
        search_pattern = os.path.join(output_dir, f"outputs_{assessment_to_use}_assessment_{safe_model_name}_*.jsonl")
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
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filepath = os.path.join(output_dir, f"outputs_{assessment_to_use}_assessment_{safe_model_name}.jsonl") #_{timestamp}.jsonl
        print(f"\n[REPO INFO] Starting new run. Results will be saved to: {results_filepath}\n")
        # The corresponding activation directory should also match this timestamp
        
    #else:
        # Reconstruct the activation dir path from the resumed results file
        #timestamp = os.path.basename(results_filepath).replace(f"outputs_{assessment_to_use}_assessment_{safe_model_name}_", "").replace(".jsonl", "")


    print(f"\n[REPO INFO] Initializing generation...\n")

    # Open the results file in append mode ('a') to add new results
    with open(results_filepath, 'a', encoding='utf-8') as f_results:
        for i, data_item in enumerate(prompt_data):
            print(f"\n" + "="*80)
            print(f"Processing data item {i+1}/{len(prompt_data)}")

            prompt_to_generate = data_item["prompt_text"]

            # Define the list of prompts to be processed for this data item
            prompts_to_process = []
            prompts_to_process.append((prompt_to_generate, f"prompt_{i}"))

            for prompt, prompt_key in prompts_to_process:
                    # Check if this specific prompt has already been processed
                if prompt_key in processed_prompt_keys:
                    print(f"--- Skipping already processed prompt: {prompt_key} ---")
                    continue

                print(f"--- Generating for prompt_key: {prompt_key} ---")
                    

                template_str = '{user_tag} {scenario} \n {assessment} \n{assistant_tag}'
                prompt = template_str.format(scenario=prompt, assessment=assessment_prompt,user_tag=user_tag, assistant_tag=assistant_tag)
                output = llm.generate(prompt, sampling_params)[0]

                prompt_text = output.prompt
                generated_text = output.outputs[0].text
            
                print(f"\nPROMPT:\n{prompt_text}")
                print(f"\nGENERATED TEXT:\n{generated_text}")

                result_item = {"prompt_key": prompt_key,
                                       "prompt_id": prompt_key,
                                       "prompt": prompt_text,
                                       "generated_text": generated_text,
                                       }
                
                # Write the result as a new line in the JSONL file and flush
                f_results.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                f_results.flush()

    print("\n[REPO INFO] Generation complete. All results have been saved.\n")

if __name__ == "__main__":
    MODEL_NAME = "/home/models/Llama-2-7b-chat-hf" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    assess(model_name=MODEL_NAME, dataset_name="llm_focused", assessment_to_use="SAM_arousal")
