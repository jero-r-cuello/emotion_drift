# utils.py
import os
import json
import datetime

def save_results_to_json(model_name, results_data, project_root_path):
    """
    Saves the results of the model generation to a JSON file.
    The filename is generated based on the model name and the current timestamp.
    """
    # Ensure the output directory exists
    output_dir = os.path.join(project_root_path, "data", "02_generated")

    os.makedirs(output_dir, exist_ok=True)

    # Create a filename
    safe_model_name = model_name.replace("/", "_")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs_{safe_model_name}_{timestamp}.json"

    filepath  = os.path.join(output_dir, filename)
    
    # Create the output data structure
    output_data = {
        "model_name": model_name,
        "generation_date": datetime.datetime.now().isoformat(),
        "results": results_data
    }
    
    # Save the data to a JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f'[INFO] JSON file saved successfully at {filepath}')
