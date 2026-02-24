import os
import json
import time
import re
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Which run to process (must match the filename in data/02_generated/outputs_*.jsonl)
RUN_TO_PROCESS = "Llama-2-7b-chat-hf_20260127_151151"
# Number of responses per API call (each response will be annotated with 3 taxonomies, so 1 text = 3 requests)
BATCH_SIZE = 15 
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
# Dataset to be processed (for metadata purposes, it must match the source of generated responses)
DATASET = "andyzou_situations" # "generated_prompts" # "emotion_query" #

INPUT_FILE_PATH = f"data/02_generated/outputs_{RUN_TO_PROCESS}.jsonl"
OUTPUT_FILE_PATH = f"data/04_annotated/batch_results_{RUN_TO_PROCESS}.jsonl"

# Annotator specs
MODEL_NAME = "gpt-5-mini-2025-08-07"
EFFORT = "high"

client = OpenAI(api_key=API_KEY)

TAXONOMIES = {
    "ekman_basic_emotions": """You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
        *    Anger: The response to an interference with our pursuit of a goal we care about. Anger can also be triggered by someone attempting to harm us (physically or psychologically) or someone we care about. In addition to removing the obstacle or stopping the harm, anger often involves the wish to hurt the target.
        *    Disgust: Arises as a feeling of repulsion or aversion towards something offensive. We can feel disgusted by something we perceive with our physical senses (sight, smell, touch, sound, taste), by the actions or appearances of people, and even by offensive ideas. Disgust contains a range of states with varying intensities from mild dislike to intense loathing.
        *    Enjoyment: Typically arising from connection or sensory pleasure. We use the word enjoyment to describe a family of related pleasurable states, everything from peace to ecstasy.
        *    Fear: Arises in response to the threat of harm, either physical, emotional, or psychological, real or imagined. Fear activates impulses to freeze or flee, serving an important role in keeping us safe as it mobilizes us to cope with potential danger.
        *    Sadness: The response to the loss of an important object or a person to which you are very attached. Serves an important role in signaling a need to receive help or comfort. Sadness describes the range of emotional states from mild disappointment to extreme despair and anguish.
        *    Surprise: Arises when we encounter sudden and unexpected events. As the briefest of the emotions, its function is to focus our attention on determining what is happening and whether or not it is dangerous. In the moment before we figure out what is occurring, before we switch to another emotion or no emotion, surprise itself can feel good or bad.
        *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",

    "go_emotions": """You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
        *    Admiration: Finding something impressive or worthy of respect.
        *    Amusement: Finding something funny or being entertained.
        *    Anger: A strong feeling of displeasure or antagonism.
        *    Annoyance: A feeling of mild anger and/or irritation.
        *    Approval: Having or expressing a favorable opinion.
        *    Caring: Displaying kindness and concern for others.
        *    Confusion: Lack of understanding, uncertainty.
        *    Curiosity: A strong desire to know or learn something.
        *    Desire: A strong feeling of wanting something or wishing for something to happen.
        *    Disappointment: Sadness or displeasure caused by the nonfulfillment of one's hopes or expectations.
        *    Disapproval: Having or expressing an unfavorable opinion.
        *    Disgust: Revulsion or strong disapproval aroused by something unpleasant or offensive.
        *    Embarrassment: A feeling of self-consciousness, shame, or awkwardness.
        *    Excitement: A feeling of great enthusiasm and eagerness.
        *    Fear: Being afraid or worried about someone or something.
        *    Gratitude: A feeling of thankfulness and appreciation.
        *    Grief: Intense sorrow, especially caused by someone's death.
        *    Joy: A feeling of pleasure and happiness.
        *    Love: A strong positive emotion of regard and affection.
        *    Nervousness: A state of apprehension, worry and/or anxiety.
        *    Optimism: Hopefulness and confidence about the future or the success of something.
        *    Pride: Pleasure or satisfaction due to ones own achievements or the achievements of those with whom one is closely associated.
        *    Realization: Feeling of becoming aware of something.
        *    Relief: Reassurance and relaxation following release from anxiety or distress.
        *    Remorse: A regret or guilty feeling.
        *    Sadness: Emotional pain and/or sorrow.
        *    Surprise: Feeling astonished and/or startled by something unexpected.
        *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",

    "plutchik_wheel": """You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
        *    Fear: Its evolutionary function is protection. It is triggered by the perception of a threat or imminent danger, which the mind interprets as a "dangerous" situation. This emotion triggers a response of flight or avoidance, with the ultimate goal of preserving physical integrity. Its intensity ranges from mild apprehension to paralyzing terror.
        *    Anger: Has the adaptive function of destroying an obstacle that prevents the achievement of a goal. It arises when an individual is confronted with what they perceive as an "Enemy" or a barrier, prompting them to respond with aggressive behaviors to remove that barrier. This emotional state can manifest itself in a range from annoyance to rage.
        *    Joy: Its evolutionary function focuses on reproduction and affiliation. It motivates to seek and retain valuable resources, such as a potential partner, food, or a significant achievement, functioning as a signal of success or gain. It is associated with a cognition of "Possession" and promotes behaviors such as courtship or celebration. It ranges from calm to ecstasy.
        *    Sadness: Has evolved to promote social reintegration. It is triggered by the loss of a person or a valuable resource, generated by a cognition of "isolation." It functions as a distress, manifesting itself through behaviors such as crying, which seek to attract support and comfort to facilitate recovery and return to the community. Its intensity ranges from a pensive state to deep grief.
        *    Trust: Its function is affiliation and the creation of strong social bonds. It emerges when interacting with someone who is perceived as a "friend" or ally. This emotion is the basis for cooperation and mutual support within a group, fostering caring and collaborative behaviors. Its spectrum ranges from acceptance to admiration.
        *    Disgust: Has the adaptive function of provoking rejection. It acts as a defense mechanism against poisoning or disease, activating in response to a repulsive object that is interpreted as "poison". The behavioral response is visceral and immediate, including removal from the object to avoid contact. Its range extends from boredom to loathing or total aversion.
        *    Anticipation: Is oriented toward the future. Its evolutionary function is exploration and preparation. It drives to investigate new possibilities or territories, starting from the cognitive question "What's out there?" It encourages behaviors such as planning, examining, and mapping, allowing the individual to prepare to find resources or opportunities. It ranges from an interest to a state vigilance.
        *    Surprise: Its main adaptive function is orientation. It is activated by a novel and unexpected stimulus, prompting the immediate question "What is this?". It causes an instant pause in the current action, forcing the individual to refocus their attention to quickly reassess the environment and decide how to respond to the new information. Its intensity can range from distraction to amazement.
        *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above."""
}


def create_prompt(text: str, taxonomy_name: str, definition: str) -> str:
    return f"""**Role:** You are an expert data analyst specializing in Natural Language Processing (NLP), with a focus on emotion annotation. Your task is to analyze responses generated by an artificial intelligence to identify the emotions they express.
    
    **Objective:** Evaluate the following text objectively to identify the predominant emotion or emotions it contains. Your analysis must be rigorous, unbiased, and based solely on the evidence within the text.
    
    1. **Definitions of Emotion Categories:**
    {definition}
        
    2. **Annotation Rules:**
    You must follow these rules strictly:
        *   Evidence: Base your annotation exclusively on the content of the response. Do not infer emotions that are not supported by it.
        
        *   Handling Mixed Emotions:
        *   *   If a single emotion adequately describes the text's tone, use only that label.
        *   *   If two or more emotions are present, create a ranked list ordered by predominance. The first emotion in the list should be the strongest or most evident, followed by secondary emotions in descending order of importance.
        *   *   Your focus in these cases is to determine the overall 'weight' or 'proportion' of each emotion in the text as a whole. **Do not** list emotions in the order they appear. A brief emotion at the beginning is less important than a sustained emotion that permeates the rest of the text.
        *   *   If a text begins with an expression of "emotion 'A'", but the remaining 90% of the content elaborates on a topic with a clear tone of "emotion 'B'", then "emotion 'B'" is the most predominant emotion, and "emotion 'A'" is secondary. Then, **do not** list emotions in the order they appear. The output in this situation should be: ["emotion 'b'", "emotion 'a'"]
        
        *   Justifiable Selection: To guide your answer, mentally formulate the justification that explains your choice, quoting or paraphrasing key phrases from the text that support your analysis.
        
    3. **Required Output Format:**
    Your response must be a single JSON code block with the following structure:
    
    ```json
    {{
        "emotions": ["Emotion_1", "Emotion_2", ...],
        }}```
    Your entire response must be ONLY in single JSON code block. Do not include any introductory or concluding sentences outside the code block.
    
    4. **Text to Annotate:**
    Here is the text to analyze:
    {text}
"""

def extract_id_number(prompt_key: str) -> str:
    """Extract the number from ‘prompt_0’, ‘prompt_10’, etc."""
    match = re.search(r'\d+', prompt_key)
    return match.group() if match else "unknown"

def save_lines_to_file(file_path: str, lines: List[str]):
    """Adds lines to the existing output file."""
    with open(file_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def create_batch_file_content(chunk_data: List[Dict]) -> List[Dict]:
    """Generate the list of JSON objects for the batch input file."""
    batch_rows = []
    
    for item in chunk_data:
        text = item["generated_text"]
        prompt_key = item.get("prompt_key", "prompt_0")

        id_identifier = extract_id_number(prompt_key)

        for tax_name, tax_def in TAXONOMIES.items():
            custom_id = f"response-{id_identifier}-{tax_name}"
            prompt_content = create_prompt(text, tax_name, tax_def)

            request_obj = {"custom_id": custom_id,
                           "method": "POST",
                           "url": "/v1/responses",
                           "body": {
                               "model": MODEL_NAME,
                               "input": prompt_content,
                               "text": {"verbosity": "medium",
                                        "format": {"type": "json_object"}},
                                "reasoning": {"effort": EFFORT},
                                "prompt_cache_key": "gen-1757602875-878DOlG67xz757xY9tkV"
                                }
            }
            batch_rows.append(request_obj)
            
    return batch_rows

def monitor_batch(batch_id: str):
    """Check the status with the defined backoff strategy."""
    wait_times = [60, 180, 180, 300, 300, 600, 900, 1800]
    wait_idx = 0
    
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        
        progress_suffix = ""
        if batch_status.request_counts:
            completed = batch_status.request_counts.completed
            total = batch_status.request_counts.total
            progress_suffix = f" ({completed}/{total})"
        
        print(f"   Status: {status}{progress_suffix}")
        # --------------------------------

        if status in ["completed", "failed", "expired", "cancelled"]:
            return batch_status
        
        if wait_idx < len(wait_times):
            sleep_time = wait_times[wait_idx]
            wait_idx += 1
        else:
            sleep_time = wait_times[-1] # Repetir el último valor (1800)
            
        print(f"   Waiting {sleep_time} seconds for the next check...")
        time.sleep(sleep_time)


def main():
    if not os.path.exists(OUTPUT_FILE_PATH):
        print(f"Creating results file: {OUTPUT_FILE_PATH}")
        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            pass 

    print(f"Reading input file: {INPUT_FILE_PATH}")
    input_data = []
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))

    total_texts = len(input_data)
    
    for i in tqdm(range(0, total_texts, BATCH_SIZE), desc="Processing batches", unit="batch"):
        chunk = input_data[i : i + BATCH_SIZE]
        print(f"\n--- Processing batchs from {i} to {min(i + BATCH_SIZE, total_texts)} ---")
        
        # Generar las requests para este chunk
        batch_requests = create_batch_file_content(chunk)
        temp_input_filename = f"temp_batch_input_{i}.jsonl"
        
        # Guardar archivo temporal jsonl para subir
        with open(temp_input_filename, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        
        try:
            # Subir archivo
            print(f"Uploading file {temp_input_filename}...")
            batch_input_file = client.files.create(
                file=open(temp_input_filename, "rb"),
                purpose="batch"
            )
            
            # Crear Batch Job
            print("Creating batch job...")
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"description": f"Annotation run {RUN_TO_PROCESS} chunk {i}"}
            )
            print(f"Batch Job created with ID: {batch_job.id}")
            
            # Monitorear
            finished_batch = monitor_batch(batch_job.id)
            
            # Procesar Resultados
            if finished_batch.status == "completed" and finished_batch.output_file_id:
                print("Batch completed. Downloading results...")
                file_response = client.files.content(finished_batch.output_file_id)
                content = file_response.text
                
                result_lines = content.strip().split("\n")
                save_lines_to_file(OUTPUT_FILE_PATH, result_lines)
                print(f"Results saved ({len(result_lines)} lines) to {OUTPUT_FILE_PATH}")
                
            # Chequeo específico de status 'failed' y descarga del error
            elif finished_batch.status == "failed":
                print(f"The batch failed (Status: failed)!")
                print(finished_batch)
                
                # Caso 1: Fallo de validación (El error está en el objeto, no en un archivo)
                if finished_batch.errors:
                    print("\n========== Validation error ==========")
                    print(json.dumps(finished_batch.errors, indent=2))
                    print("===========================================\n")
                
                # Caso 2: Fallo con archivo generado (menos común en validación, pero posible)
                elif finished_batch.error_file_id:
                    print(f"Downloading error file (ID: {finished_batch.error_file_id})...")
                    try:
                        error_response = client.files.content(finished_batch.error_file_id)
                        print(error_response.text)
                    except Exception as e:
                        print(f"Unable to download file: {e}")
                else:
                    print("The batch failed, but there are no details about ‘errors’ or ‘error_file_id’.")

            # Otros estados (expired, cancelled, etc.)
            else:
                print(f"The batch finished with status: {finished_batch.status}")
                if finished_batch.error_file_id:
                    print(f"There is an error file (ID: {finished_batch.error_file_id}), but the status is not 'failed' or 'completed'.")

        # Completely uknown error (0 information about what happened)
        except Exception as e:
            print(f"An error occurred while processing chunk {i}: {e}")
        
        finally:
            if os.path.exists(temp_input_filename):
                os.remove(temp_input_filename)

    print("\nProcess completed.")

if __name__ == "__main__":
    main()