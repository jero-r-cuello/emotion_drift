import os
import json
import time
import re
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# --- Annotation config (full 2 models x 3 stimulus domains) ---
# Outputs + temp files go to the network drive (the root disk is full).
ANNOT_DIR = "/is/cluster/fast/jgeiping/annotated"
TMP_DIR = os.path.join(ANNOT_DIR, "_tmp")
REQ_PER_BATCH = 9000        # requests per Batch job (~3000 responses; under the 50k-req / 200MB limits)
MAX_BATCH_RETRIES = 3       # retry a failed/expired batch instead of silently skipping it

# (run_id, dataset). Each run -> its own batch_results_<run>.jsonl on the drive.
RUNS = [
    ("Llama-2-7b-chat-hf_20260625_humanprompts", "generated_human_prompts"),
    ("Llama-2-7b-chat-hf_20260625_humanconv", "generated_human_conversation_prompts"),
    ("Llama-2-7b-chat-hf_20260625_aicentric", "generated_prompts"),
    ("Qwen2.5-14B-Instruct_20260625_humanprompts", "generated_human_prompts"),
    ("Qwen2.5-14B-Instruct_20260625_humanconv", "generated_human_conversation_prompts"),
    ("Qwen2.5-14B-Instruct_20260625_aicentric", "generated_prompts"),
]

# Annotator specs
MODEL_NAME = "gpt-5-mini-2025-08-07"
EFFORT = "low"   # calibration: low ~= medium ~= high on primary labels (10/12, Jaccard 0.87) at ~1/5 the output cost

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
                                "prompt_cache_key": f"annot-{tax_name}"
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


MAX_INFLIGHT = 16   # concurrent Batch jobs (self-throttles if the org enqueue limit is hit)
POLL = 45           # seconds between status polls


def _pending_for_run(run):
    """Resume-aware: (output_path, fail_path, [pending request dicts]) for a run."""
    input_path = f"data/02_generated/outputs_{run}.jsonl"
    output_path = os.path.join(ANNOT_DIR, f"batch_results_{run}.jsonl")
    fail_path = os.path.join(ANNOT_DIR, f"failed_{run}.txt")
    os.makedirs(ANNOT_DIR, exist_ok=True)
    if not os.path.exists(input_path):
        print(f"[{run}] input missing: {input_path}; skipping", flush=True)
        return None
    done = set()
    if os.path.exists(output_path):
        for line in open(output_path, "r", encoding="utf-8"):
            try:
                done.add(json.loads(line).get("custom_id"))
            except json.JSONDecodeError:
                pass
    responses = [json.loads(l) for l in open(input_path, "r", encoding="utf-8") if l.strip()]
    all_reqs = create_batch_file_content(responses)
    todo = [r for r in all_reqs if r["custom_id"] not in done]
    print(f"[{run}] {len(responses)} resp x {len(TAXONOMIES)} = {len(all_reqs)} reqs | "
          f"{len(done)} done | {len(todo)} to do", flush=True)
    return output_path, fail_path, todo


def _submit(task):
    """Upload + create one Batch job; return job_id. Does NOT wait."""
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp = os.path.join(TMP_DIR, f"in_{task['run']}_{task['chunk_idx']}.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in task["requests"]:
            f.write(json.dumps(r) + "\n")
    try:
        bf = client.files.create(file=open(tmp, "rb"), purpose="batch")
        job = client.batches.create(
            input_file_id=bf.id, endpoint="/v1/responses", completion_window="24h",
            metadata={"description": f"annot {task['run']} chunk {task['chunk_idx']}"})
        return job.id
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_all():
    """Annotate all RUNS with up to MAX_INFLIGHT Batch jobs running CONCURRENTLY
    (instead of one-at-a-time), polling them and saving as each completes.
    Resumable (skips done custom_ids) and retries failed/expired batches."""
    out_paths, fail_paths, pending = {}, {}, []
    for run, _ds in RUNS:
        info = _pending_for_run(run)
        if info is None:
            continue
        output_path, fail_path, todo = info
        out_paths[run], fail_paths[run] = output_path, fail_path
        for ci, i in enumerate(range(0, len(todo), REQ_PER_BATCH)):
            pending.append(dict(run=run, chunk_idx=ci, requests=todo[i:i + REQ_PER_BATCH], retries=0))
    print(f"\n[orchestrator] {len(pending)} batches pending; up to {MAX_INFLIGHT} concurrent\n", flush=True)

    inflight, saved = {}, 0
    while pending or inflight:
        while pending and len(inflight) < MAX_INFLIGHT:
            task = pending.pop(0)
            try:
                jid = _submit(task)
                inflight[jid] = task
                print(f"  submitted {task['run']} chunk {task['chunk_idx']} -> {jid[:20]} "
                      f"(inflight {len(inflight)}, pending {len(pending)})", flush=True)
            except Exception as e:
                task["retries"] += 1
                print(f"  submit failed {task['run']} chunk {task['chunk_idx']}: {e}; requeue", flush=True)
                if task["retries"] <= MAX_BATCH_RETRIES:
                    pending.append(task)
                time.sleep(10)
                break   # likely the org enqueue limit; let some finish before retrying
        for jid in list(inflight):
            try:
                b = client.batches.retrieve(jid)
            except Exception:
                continue
            if b.status == "completed" and b.output_file_id:
                t = inflight.pop(jid)
                lines = [l for l in client.files.content(b.output_file_id).text.split("\n") if l.strip()]
                save_lines_to_file(out_paths[t["run"]], lines)
                saved += len(lines)
                print(f"  DONE {t['run']} chunk {t['chunk_idx']} (+{len(lines)}; total saved {saved})", flush=True)
            elif b.status in ("failed", "expired", "cancelled"):
                t = inflight.pop(jid)
                t["retries"] += 1
                print(f"  {b.status} {t['run']} chunk {t['chunk_idx']} (retry {t['retries']})", flush=True)
                if t["retries"] <= MAX_BATCH_RETRIES:
                    pending.append(t)
                else:
                    with open(fail_paths[t["run"]], "a", encoding="utf-8") as f:
                        f.write(f"chunk {t['chunk_idx']}: FAILED after {MAX_BATCH_RETRIES} retries\n")
        if pending or inflight:
            time.sleep(POLL)
    print("\nALL RUNS DONE", flush=True)


if __name__ == "__main__":
    run_all()