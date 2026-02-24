import multiprocessing
import os
import pandas as pd
import re
import datetime
from vllm import LLM
import vllm.hook_store as hook_store
from transformers import AutoConfig
import json
import glob
from src.utils import load_dataset


def generate(model_name, target_layers, dataset_name="andyzou_situations", 
             dataset_testing=False, resume_run=False, assessments_path=None, 
             activation_save_batch_size=64, home_model_dir="/home/models/"):
    """
    Generate outputs for a list of prompts, saving results incrementally and supporting resumption.
    Captures activations from specified layers and saves results to a .jsonl file.
    """
        
    second_prompts_assessments = []
    assessment_condition = None
    if assessments_path:
        condition_raw = assessments_path.split("_")[-1]
        assessment_condition = condition_raw.replace(".json", "")

        try:
            with open(assessments_path, "r", encoding="utf-8") as f_assess:
                second_prompts_assessments = json.load(f_assess)
            print(f"\n[REPO INFO] {len(second_prompts_assessments)} questionnaires loaded for assessment stage.")
        except FileNotFoundError:
            print(f"\n[REPO WARNING] Assessments file not found in path: {assessments_path}. Assessment stage will be ommited.")
            second_prompts_assessments = []
        except json.JSONDecodeError:
            print(f"\n[REPO WARNING] Error in assessments file processing. Assessment stage will be ommited.")
            second_prompts_assessments = []

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
        
        print("\n[REPO INFO] LLM initialized successfully.\n")

        if model_name.startswith(home_model_dir):
            model_name = model_name[len(home_model_dir):]

        safe_model_name = model_name.replace("/", "_")
        
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
                with open(results_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            # Load each line as a JSON object and get its key
                            record = json.loads(line)
                            processed_prompt_keys.add(record["prompt_key"])
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

        assessments_results_filepath = None
        if assessments_path:
            assessments_results_folder = os.path.join(output_dir, "assessments", f"assessments_{safe_model_name}_{timestamp}")
            os.makedirs(assessments_results_folder, exist_ok=True)
            assessments_results_filepath = os.path.join(assessments_results_folder, f"{assessment_condition}_assessments_outputs_{safe_model_name}_{timestamp}.jsonl")
            print(f"\n[REPO INFO] Assessments results will be saved to: {assessments_results_filepath}\n")

        print(f"\n[REPO INFO] Initializing generation...\n")
        
        # Open the results file in append mode
        with open(results_filepath, "a", encoding="utf-8") as f_results:
            for i, data_item in enumerate(prompt_data):
                print(f"\n" + "="*80)
                print(f"Processing data item {i+1}/{len(prompt_data)}")

                prompt_to_generate = data_item["prompt_text"]
                emotion_for_this_prompt = data_item["emotion"]
                label_for_this_prompt = data_item.get("label", -1)
                split_for_this_prompt = data_item.get("split", "unknown")

                # Define the list of prompts to be processed for this data item
                prompts_to_process = []
                prompts_to_process.append((prompt_to_generate, f"prompt_{i}"))

                for prompt, prompt_key in prompts_to_process:
                    # Check if this specific prompt has already been processed
                    if prompt_key in processed_prompt_keys:
                        print(f"[REPO INFO]: Skipping already processed prompt: {prompt_key}")
                        continue

                    print(f"Generating for prompt_key: {prompt_key}")
                    
                    hook_instructions["save_path"] = activations_run_dir
                    hook_instructions["prompt_key"] = prompt_key
                    hook_instructions["target_layers"] = target_layers
                    hook_instructions["activation_save_batch_size"] = activation_save_batch_size

                    conversation = [
                        {"role": "user", "content": prompt}
                    ]

                    try:
                        outputs = llm.chat(conversation,
                                    use_tqdm=False)
                        print("\n[REPO INFO] llm.chat() worked correctly.")

                    except ValueError as e:
                        if "must provide a chat template" in str(e):
                            print("\n>>> [REPO INFO] llm.chat() failed due to lack of template. Using manual formatting with llm.generate()...")
                            
                            tokenizer = llm.get_tokenizer()
                            # We manually apply the model's chat template
                            prompt_formatted = tokenizer.apply_chat_template(
                                conversation,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            
                            outputs = llm.generate([prompt_formatted])

                        else:
                            print(f"\n[REPO INFO] An unexpected ValueError was encountered: {e}")
                            raise

                    output = outputs[0]            
                    hook_instructions.clear()

                    generated_text = output.outputs[0].text
            
                    print(f"\nPROMPT:\n{prompt}")
                    print(f"\nGENERATED TEXT:\n{generated_text}")
                    print(f"\n[REPO INFO] Activations of {prompt_key} were saved by vLLM workers.")

                    result_item = {"prompt_key": prompt_key,
                                    "prompt": prompt,
                                   "generated_text": generated_text,
                                   "emotion_considered": emotion_for_this_prompt,
                                   "label": label_for_this_prompt,
                                   "split": split_for_this_prompt}

                    f_results.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                    f_results.flush()
                    
                    if second_prompts_assessments:
                        print(f"\nStarting generation of assessment responses for prompt {prompt_key}")
                        
                        for assessment in second_prompts_assessments:
                            assessment_name = assessment.get("name", "unknown_assessment")
                            assessment_prompt = assessment.get("prompt")

                            if not assessment_prompt:
                                continue 

                            conversation_step2 = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": generated_text},
                                {"role": "user", "content": assessment_prompt}
                            ]

                            try:
                                outputs_step2 = llm.chat(conversation_step2,
                                                    use_tqdm=False)
                                
                            except ValueError as e:
                                if "must provide a chat template" in str(e):
                                    tokenizer = llm.get_tokenizer()
                                    prompt_formatted_step2 = tokenizer.apply_chat_template(
                                        conversation_step2,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )
                                    outputs_step2 = llm.generate([prompt_formatted_step2])
                                else:
                                    raise
                            
                            generated_text_step2 = outputs_step2[0].outputs[0].text

                            print(f"\nQUESTIONNAIRE ({assessment_name}):\n{assessment_prompt}")
                            print(f"\nRESPONSE:\n{generated_text_step2}")

                            # Create the result item for the second stage
                            result_item_step2 = {
                                "original_prompt_key": prompt_key,
                                "emotion_considered": emotion_for_this_prompt,
                                "original_prompt": prompt,
                                "assessment_name": assessment_name,
                                "assessment_prompt": assessment_prompt,
                                "generated_text_step2": generated_text_step2,
                                "full_conversation_history": conversation_step2
                            }
    
                            if assessments_results_filepath:
                                with open(assessments_results_filepath, "a", encoding="utf-8") as f_results_step2:
                                    f_results_step2.write(json.dumps(result_item_step2, ensure_ascii=False) + "\n")
                                    f_results_step2.flush()
                        
                        print(f"\nFinalized generation of responses to assessments for prompt {prompt_key}")

        print("\n[REPO INFO] Generation complete. All results have been saved.\n")

        return f'{safe_model_name}_{timestamp}' # Safe name of the run, for other scripts to use
        
if __name__ == "__main__":
    MODEL_NAME = "/home/models/Llama-2-7b-chat-hf" # "/home/models/Qwen2.5-14B-Instruct" #
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    
    TARGET_LAYERS = [l for l in range(num_layers)] # List of layers to observe.
    print(f"\n[REPO INFO] Model has {num_layers} layers. Targeting {len(TARGET_LAYERS)} layers\n")

    generate(model_name=MODEL_NAME,
             target_layers=TARGET_LAYERS,
             dataset_name= "emotion_query", # "andyzou_situations", # "generated_prompts", #
             dataset_testing=False,
             resume_run=False,
             assessments_path=None # "data/01_stimuli/assessments/emotion_assessments_control.json"
             )