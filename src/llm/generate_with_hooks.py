# generate_with_hooks.py

import multiprocessing
import os
import itertools
import pandas as pd
import datetime
from vllm import LLM, SamplingParams
import vllm.hook_store as hook_store
from src.utils import save_results_to_json, save_activations


def generate(model_name, target_layers):
    """
    Generate outputs for a list of prompts using the specified model and target layers.
    Captures activations from specified layers and saves results to JSON files.
    """
    with multiprocessing.Manager() as manager:
        # Its important to charge the hook_store before initializing the LLM.
        shared_hooks_dict = manager.dict()
        hook_store.ACTIVATION_HOOKS = shared_hooks_dict
        print("\n[REPO INFO] Shared hook store initialized.\n")

        project_root = os.getcwd() #!! There is something about importing the data here that seems weird to me.
        
        data_path = os.path.join(project_root, "data", "01_stimuli", "empatheticdialogues", "test_unique_prompts.csv")
        df_scenarios = pd.read_csv(data_path)

        scenarios = df_scenarios["prompt"].tolist()[:2] #!! Limiting to the first 2 prompts for testing
        unique_emotions = df_scenarios["context"].unique().tolist()[:2] #!! Limiting to the first 2 emotions for testing
        prompts = [f'USER: In a range from 1 to 9, consider the amount of {emotion} in the following scenario: "{scenario}". The amount of {emotion} is' 
                   for emotion, scenario in itertools.product(unique_emotions, scenarios)]

        print("\n[REPO INFO] Initializing the LLM...\n")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  #!! Probably adding the calculate_optimal_tensor_parallel_size function in the future. Will have to debug everything again, because it will change the way the model is loaded.
            trust_remote_code=True,
            max_model_len=16384, #!! Now set to 16384, but for longer contexts, it should be increased.
            enforce_eager=True # This is important to ensure that the hooks are called correctly
        )
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
        print("\n[REPO INFO] LLM initialized successfully.\n")

        all_results_to_save = []
        all_activations_to_save = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[REPO INFO] Initializing generation...\n")

        for i, prompt in enumerate(prompts):
            print(f"\n" + "="*80)
            print(f"Processing prompt {i+1}/{len(prompts)}")

            # Clean the shared hooks dictionary for each prompt
            for layer_idx in target_layers:
                shared_hooks_dict[layer_idx] = manager.list()
            
            output = llm.generate(prompt, sampling_params)[0]
            
            prompt_text = output.prompt
            generated_text = output.outputs[0].text
            
            print(f"\nPROMPT:\n{prompt_text}")
            print(f"\nGENERATED TEXT:\n{generated_text}")
            
            prompt_activations = {}
            for layer_idx in target_layers:
                captured_list = shared_hooks_dict.get(layer_idx)
                if captured_list:
                    last_token_activation = captured_list[0][-1, :] #!! Taking the last token's activation. Could be a different token if needed or the activations of all of them.
                    print(f"\nActivation (Layer {layer_idx}, Last token):")
                    print(f"  - Shape: {last_token_activation.shape}")
                    print(f"  - Firs 5 values: {last_token_activation[:5]}")
                    prompt_activations[f"layer_{layer_idx}"] = last_token_activation
                else:
                    print(f"\n[ERROR] Couldn't extract activations of layer {layer_idx}.")

            if prompt_activations:
                result_item = {"prompt": prompt_text, "generated_text": generated_text}
                all_results_to_save.append(result_item)
                prompt_key = f"prompt_{i}"
                all_activations_to_save[prompt_key] = {
                    "prompt": prompt_text,
                    "activations": prompt_activations
                }


        if all_results_to_save:
            save_results_to_json(
                model_name=model_name,
                results_data=all_results_to_save,
                project_root_path=project_root,
                timestamp=timestamp
            )
        
        if all_activations_to_save:
            save_activations(
                activations_data=all_activations_to_save,
                model_name=model_name,
                target_layers=target_layers,
                project_root_path=project_root,
                timestamp=timestamp
            )

if __name__ == "__main__":
    MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    TARGET_LAYERS = [10, 15, 20] # List of layers to observe

    generate(model_name=MODEL_NAME,
             target_layers=TARGET_LAYERS)