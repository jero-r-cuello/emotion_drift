#%%
import os
import json
import glob
from vllm import LLM, SamplingParams
from transformers import AutoConfig

def run_assessment_on_file(model_name, input_run_path, output_dir_path, assessments_path, resume_run=True):
    """
    Loads a previously generated JSONL file (User prompt + Model response),
    applies a list of psychological assessments (questionnaires) as a second turn,
    and saves the results.
    """
    
    # 1. Load Assessments
    try:
        with open(assessments_path, 'r', encoding='utf-8') as f_assess:
            assessments_list = json.load(f_assess)
        print(f"\n[REPO INFO] {len(assessments_list)} questionnaires loaded.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\n[REPO ERROR] Could not load assessments file: {e}")
        return

    # 2. Load Previous Generations (Input Data)
    if not os.path.exists(input_run_path):
        print(f"\n[REPO ERROR] Input file not found: {input_run_path}")
        return

    print(f"\n[REPO INFO] Loading previous generations from: {input_run_path}")
    previous_generations = []
    with open(input_run_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                previous_generations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[REPO INFO] Loaded {len(previous_generations)} conversation records.")

    # 3. Initialize LLM
    print(f"\n[REPO INFO] Initializing the LLM...\n")
    # Handle model name paths
    home_model_dir = "/home/models/"
    full_model_path = model_name
    if not model_name.startswith("/") and os.path.exists(home_model_dir + model_name):
        full_model_path = home_model_dir + model_name
    
    llm = LLM(
        model=full_model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=4092, # Adjust based on model
        enforce_eager=True
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    print("\n[REPO INFO] LLM initialized successfully.\n")

    # 4. Setup Output Paths
    output_dir = output_dir_path
    os.makedirs(output_dir, exist_ok=True)

    # Construct new filename based on input filename to maintain traceability
    input_filename = os.path.basename(input_run_path)
    output_filename = f"assessments_from_{input_filename}"
    results_filepath = os.path.join(output_dir, output_filename)

    # 5. Resume Logic
    processed_combinations = set() # Stores "prompt_key|assessment_name"
    
    if resume_run and os.path.exists(results_filepath):
        print(f"\n[REPO INFO] Found existing results file. Scanning for completed assessments...")
        with open(results_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Create a unique signature for what has been done
                    p_key = record.get('original_prompt_key')
                    a_name = record.get('assessment_name')
                    if p_key and a_name:
                        processed_combinations.add(f"{p_key}|{a_name}")
                except json.JSONDecodeError:
                    pass
        print(f"[REPO INFO] Found {len(processed_combinations)} completed assessment items. Resuming...\n")
    else:
        print(f"\n[REPO INFO] Starting new assessment run. Saving to: {results_filepath}\n")

    # 6. Generation Loop
    with open(results_filepath, 'a', encoding='utf-8') as f_results:
        
        # Iterate over each original conversation
        for i, prev_gen in enumerate(previous_generations):
            prompt_key = prev_gen.get('prompt_key')
            original_prompt = prev_gen.get('prompt')
            original_response = prev_gen.get('generated_text')
            emotion_considered = prev_gen.get('text_emotion') or prev_gen.get('emotion_considered') or prev_gen.get('emotion_scenario')

            # Skip if data is malformed
            if not prompt_key or not original_prompt or not original_response:
                print(f"[WARNING] Skipping incomplete record at index {i}")
                continue

            print(f"\n" + "="*80)
            print(f"Processing Item {i+1}/{len(previous_generations)} (Key: {prompt_key})")

            # Iterate over all questionnaires
            for assessment in assessments_list:
                assessment_name = assessment.get("name", "unknown")
                assessment_prompt = assessment.get("prompt")

                if not assessment_prompt:
                    continue

                # Check if done
                combo_key = f"{prompt_key}|{assessment_name}"
                if combo_key in processed_combinations:
                    # Silent skip to reduce log spam, or print simple dot
                    continue

                print(f"--- Running assessment: {assessment_name} ---")

                # Construct Conversation History (Turn 1 + Turn 2 Input)
                conversation_step2 = [
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": original_response},
                    {"role": "user", "content": assessment_prompt}
                ]

                # Generate
                try:
                    outputs = llm.chat(conversation_step2, sampling_params=sampling_params, use_tqdm=False)
                except ValueError as e:
                    if "must provide a chat template" in str(e):
                        tokenizer = llm.get_tokenizer()
                        prompt_formatted = tokenizer.apply_chat_template(
                            conversation_step2, tokenize=False, add_generation_prompt=True
                        )
                        outputs = llm.generate([prompt_formatted], sampling_params)
                    else:
                        print(f"[ERROR] Failed generation for {combo_key}: {e}")
                        continue

                generated_text_step2 = outputs[0].outputs[0].text

                # Construct Result Object
                result_item = {
                    "original_prompt_key": prompt_key,
                    "emotion_considered": emotion_considered,
                    "original_prompt": original_prompt,
                    "original_response": original_response, # Optional: helpful for debugging without joining files
                    "assessment_name": assessment_name,
                    "assessment_prompt": assessment_prompt,
                    "generated_text_step2": generated_text_step2,
                    "full_conversation_history": conversation_step2
                }

                # Save immediately
                f_results.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                f_results.flush()
                
                print(f"Response: {generated_text_step2[:100]}...")

    print("\n[REPO INFO] Assessment run complete. All results saved.\n")


if __name__ == "__main__":
    # CONFIGURATION
    MODEL_NAME = "/home/models/Llama-2-7b-chat-hf"
    
    # Ruta exacta del archivo que generaste en el script 1 y quieres procesar
    INPUT_RUN_FILE = "/home/jcuello/emotion_drift/data/02_generated/outputs_Llama-2-7b-chat-hf_20251014_203636.jsonl"
    
    OUTPUT_DIR = "/home/jcuello/emotion_drift/data/02_generated/assessments/"

    # Ruta de los cuestionarios
    ASSESSMENTS_FILE = "/home/jcuello/emotion_drift/data/01_stimuli/assessments/emotion_assessments.json"

    run_assessment_on_file(
        model_name=MODEL_NAME,
        input_run_path=INPUT_RUN_FILE,
        output_dir_path=OUTPUT_DIR,
        assessments_path=ASSESSMENTS_FILE,
        resume_run=True
    )
# %%
