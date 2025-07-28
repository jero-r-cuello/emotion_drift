# generate_with_hooks.py

import multiprocessing
import os
import pandas as pd
import datetime
from vllm import LLM, SamplingParams
import vllm.hook_store as hook_store
from transformers import AutoConfig
from src.utils import save_results_to_json, load_dataset


def generate(model_name, target_layers, dataset_name="andyzou_situations", dataset_testing=False):
    """
    Generate outputs for a list of prompts using the specified model and target layers.
    Captures activations from specified layers and saves results to JSON files.
    """
    with multiprocessing.Manager() as manager:
        # Its important to charge the hook_store before initializing the LLM.
        hook_instructions = manager.dict()
        hook_store.ACTIVATION_HOOKS = hook_instructions
        print("\n[REPO INFO] Shared hook store initialized.\n")

        project_root = os.getcwd() #!! There is something about importing the data here that seems weird to me. Probably should be passed as an argument.
        
        # Load the dataset and generate the prompts.
        prompt_data = load_dataset(dataset_name, 
                                   testing=dataset_testing)  #!! Set testing=True for quick runs
        
        print("\n[REPO INFO] Initializing the LLM...\n")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  #!! Probably adding the calculate_optimal_tensor_parallel_size function in the future. Will have to debug everything again, because it will change the way the model is loaded.
            trust_remote_code=True,
            max_model_len=16384, #!! Now set to 16384, but should it be increased for longer contexts?
            enforce_eager=True # This is important to ensure that the hooks are called correctly
        )
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
        print("\n[REPO INFO] LLM initialized successfully.\n")

        all_results_to_save = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")
        activations_run_dir = os.path.join(project_root, "data", "03_activations", f"activations_{safe_model_name}_{timestamp}")
        os.makedirs(activations_run_dir, exist_ok=True)
        print(f"\n[REPO INFO] Activations will be saved in: {activations_run_dir}\n")

        print(f"\n[REPO INFO] Initializing generation...\n")

        for i, data_item in enumerate(prompt_data):
            print(f"\n" + "="*80)
            print(f"Processing prompt {i+1}/{len(prompt_data)}")

            prompt_to_generate = data_item["prompt_text"]
            emotion_for_this_prompt = data_item["emotion"]

            label_for_this_prompt = data_item.get("label", -1)
            split_for_this_prompt = data_item.get("split", "unknown")

            if dataset_name == "emotion_query" or dataset_name == "xuanfengzu_emotion_query": #!! Added this to handle the specific case of emotion_query dataset. Maybe should be handled in the load_dataset function.
                neutral_prompt = f'Please answer the following query with neutrality: {prompt_to_generate}'
                emotional_prompt = f'Please answer the following query with {emotion_for_this_prompt}: {prompt_to_generate}'
                for prompt in [neutral_prompt, emotional_prompt]:
                    prompt_key = f"prompt_{i}_neutral" if prompt == neutral_prompt else f"prompt_{i}_emotional"
                    hook_instructions['save_path'] = activations_run_dir
                    hook_instructions['prompt_key'] = prompt_key
                    hook_instructions['target_layers'] = target_layers
            
                    output = llm.generate(prompt, sampling_params)[0]
            
                    hook_instructions.clear()

                    prompt_text = output.prompt
                    generated_text = output.outputs[0].text
            
                    print(f"\nPROMPT:\n{prompt_text}")
                    print(f"\nGENERATED TEXT:\n{generated_text}")
            
                    print(f"\n[REPO INFO] Activations of {prompt_key} were saved by vLLM workers.")

                    result_item = {"prompt_id": prompt_key,
                                   "prompt": prompt_text,
                                   "generated_text": generated_text,
                                   "text_emotion": emotion_for_this_prompt}
                    all_results_to_save.append(result_item)

            else: # For now, else is andyzou
                prompt_key = f"prompt_{i}"
                hook_instructions['save_path'] = activations_run_dir
                hook_instructions['prompt_key'] = prompt_key
                hook_instructions['target_layers'] = target_layers
                
                output = llm.generate(prompt_to_generate, sampling_params)[0]
                
                hook_instructions.clear()

                prompt_text = output.prompt
                generated_text = output.outputs[0].text
                
                print(f"\nPROMPT:\n{prompt_text}")
                print(f"\nGENERATED TEXT:\n{generated_text}")
                
                print(f"\n[REPO INFO] Activations of {prompt_key} were saved by vLLM workers.")

                result_item = {"prompt": prompt_text,
                                "generated_text": generated_text,
                                "emotion_considered": emotion_for_this_prompt,
                                "label": label_for_this_prompt,
                                "split": split_for_this_prompt}
                all_results_to_save.append(result_item)
        
        if all_results_to_save:
            save_results_to_json(
                model_name=model_name,
                dataset_used=dataset_name,
                results_data=all_results_to_save,
                project_root_path=project_root,
                timestamp=timestamp
            )
        
if __name__ == "__main__":
    MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    
    TARGET_LAYERS = [l for l in range(num_layers)] # List of layers to observe.

    generate(model_name=MODEL_NAME,
             target_layers=TARGET_LAYERS,
             dataset_name="andyzou_rep_eng", #!! Change this to the dataset you want to use.
             dataset_testing=False)