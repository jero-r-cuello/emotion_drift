#%%
"""
Helper script to convert the JSONL files with annotations
into a consolidated CSV file ready for the performance analysis
"""

import json
import csv
from collections import defaultdict
import sys
import pandas as pd

INPUT_JSONL_FILE = "src/nlp/gpt-5-nano-all-annotations.jsonl"
OUTPUT_CSV_FILE = "src/nlp/gpt-5-nano-consolidated_annotations.csv"

PROMPTS_CSV_FILE = "data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv"

PROMPT_ID_COLUMN = "id"
PROMPT_TEXT_COLUMN = "response_text"

TAXONOMY_1 = "ekman_basic_emotions"
TAXONOMY_2 = "go_emotions"


def load_prompts_from_csv(csv_path, id_col, text_col):
    """
    Load the prompt texts from a CSV file into a dictionary.
    Transform the numerical ID from the CSV (e.g., 0) to the format expected
    by JSONL (e.g., ‘request-0’) in order to map them.
    """
    print(f"Loading prompts from file: {csv_path}")
    prompts_dict = {}
    try:
        df = pd.read_csv(csv_path)

        if id_col not in df.columns or text_col not in df.columns:
            print(f"ERROR: The CSV file ‘{csv_path}’ must contain the columns ‘{id_col}’ and ‘{text_col}’.", file=sys.stderr)
            sys.exit(1)
        
        for index, row in df.iterrows():
            key = f"request-{row[id_col]}"
            value = row[text_col]

            if pd.notna(value):
                prompts_dict[key] = value

        print(f"Successfully loaded {len(prompts_dict)} prompts.")
        return prompts_dict

    except FileNotFoundError:
        print(f"ERROR: The prompts CSV file '{csv_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading the prompts CSV file: {e}", file=sys.stderr)
        sys.exit(1)


def parse_jsonl_to_csv(input_path, output_path, prompts_dict):
    """
    Reads a JSONL file, groups the data by prompt and model configuration,
    and writes the result to a CSV file.
    """
    grouped_data = defaultdict(dict)

    print(f"Reading and processing JSON file: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    custom_id = data.get("custom_id")
                    response_body = data.get("response", {}).get("body", {})
                    if not custom_id or not response_body:
                        print(f"WARNING: Line {i+1} omitted because of missing 'custom_id' or 'response.body'.")
                        continue

                    model_name = response_body.get("model")
                    effort = response_body.get("reasoning", {}).get("effort")
                    verbosity = response_body.get("text", {}).get("verbosity")
                    output_list = response_body.get("output", [])
                    
                    if len(output_list) < 2 or "content" not in output_list[1] or not output_list[1]["content"]:
                         print(f"WARNING: 'output' structure unexpected in line {i+1}. Omitted.")
                         continue
                    output_text_str = output_list[1]["content"][0].get("text")

                    if not all([model_name, effort, verbosity, output_text_str]):
                        print(f"WARNING: Line {i+1} omitted because of missing essential data (model, effort, verbosity, output).")
                        continue

                    prompt_id = "-".join(custom_id.split("-")[:2])
                    
                    suffix_to_remove = f"-{verbosity}-{effort}"
                    if custom_id.endswith(suffix_to_remove):
                        base_id = custom_id[:-len(suffix_to_remove)]
                        taxonomy = base_id.rpartition("-")[-1]
                    else:
                        print(f"WARNING: The format of 'custom_id' in line {i+1} ('{custom_id}') is unexpected. Omitted.")
                        continue

                    annotation_data = json.loads(output_text_str)
                    labels = annotation_data.get("emotions", [])
                    justification = annotation_data.get("justification", "")

                    grouping_key = (prompt_id, model_name, verbosity, effort)
                    grouped_data[grouping_key][taxonomy] = {"labels": labels, "justification": justification}

                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                    print(f"WARNING: Error processing line {i+1}: {e}. Omitted.")
    
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing completed. {len(grouped_data)} unique rows were grouped for the CSV.")
    print(f"Writing data to file: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "response_text", "model",
            f'{TAXONOMY_1.replace("_basic_emotions", "")}_labels', f'{TAXONOMY_1.replace("_basic_emotions", "")}_justification',
            f'{TAXONOMY_2}_labels', f'{TAXONOMY_2}_justification'
        ])

        for key, taxonomies in sorted(grouped_data.items()): # sorted() para un orden predecible
            prompt_id, model_name, verbosity, effort = key
            tax1_data = taxonomies.get(TAXONOMY_1, {})
            tax2_data = taxonomies.get(TAXONOMY_2, {})

            row = [
                prompts_dict.get(prompt_id, f"Text not found for id: {prompt_id}"),
                f"{model_name}-verbosity-{verbosity}-effort-{effort}",
                str(tax1_data.get("labels", "[]")),
                tax1_data.get("justification", ""),
                str(tax2_data.get("labels", "[]")),
                tax2_data.get("justification", "")
            ]
            writer.writerow(row)

    print(f"The file ‘{output_path}’ has been successfully generated!")


if __name__ == "__main__":
    prompts_dictionary = load_prompts_from_csv(PROMPTS_CSV_FILE, PROMPT_ID_COLUMN, PROMPT_TEXT_COLUMN)
    
    parse_jsonl_to_csv(INPUT_JSONL_FILE, OUTPUT_CSV_FILE, prompts_dictionary)
# %%
