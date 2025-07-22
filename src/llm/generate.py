# generate.py
from vllm import LLM, SamplingParams
import os
import pandas as pd
import itertools
from src.utils import save_results_to_json

def generate_output(llm_model, prompts, sampling_params, project_root):
    """
    Generate outputs for a list of prompts using the specified sampling parameters.
    Prints the generated text for each prompt.

    Args:
        llm_model (str): The model to use for generation.
        prompts (list of str): List of prompts to generate outputs for.
        sampling_params (SamplingParams): Sampling parameters for generation.
    """
    llm = LLM(model=llm_model,
              tensor_parallel_size=1, #!! Probably adding the calculate_optimal_tensor_parallel_size function in the future
              max_model_len=16384,
              trust_remote_code=True)
    
    outputs = llm.generate(prompts, sampling_params)

    all_results_to_save = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"-"*50 + "\n")
        print(f"Prompt: {prompt}. \nGenerated text: {generated_text}")
                    
        result_item = {
            "prompt": prompt,
            "generated_text": generated_text,
        }
        
        all_results_to_save.append(result_item)

    # Save the results to a JSON file.
    save_results_to_json(model_name=llm_model,
                         results_data=all_results_to_save,
                         project_root_path=project_root)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    
    # Subo TRES niveles para llegar a la raíz del proyecto (emotion_drift)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    
    data_path = os.path.join(project_root, "data", "01_stimuli", "empatheticdialogues", "test_unique_prompts.csv")
    df_scenarios = pd.read_csv(data_path)

    scenarios = df_scenarios["prompt"].tolist()[:10]  #!! Limiting to the first 10 prompts for testing
    unique_emotions = df_scenarios["context"].unique().tolist()

    prompts = [f'USER: In a range from 1 to 9, consider the amount of {emotion} in the following scenario: "{scenario}". The amount of {emotion} is' 
               for emotion, scenario in itertools.product(unique_emotions, scenarios)]

    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=256)

    llm_model = "" # "microsoft/Phi-3-medium-128k-instruct" # "Qwen/Qwen3-14B"

    generate_output(llm_model, prompts, sampling_params, project_root)