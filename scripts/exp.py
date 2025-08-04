from src.utils import pre_processing_results
from src.llm.generate_with_hooks import generate
from transformers import AutoConfig

MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
TARGET_LAYERS = [l for l in range(num_layers)]

DATASET = "out_of_domain"

run_to_load = generate(model_name=MODEL_NAME,
                       target_layers=TARGET_LAYERS,
                       dataset_name=DATASET,
                       dataset_testing=False,
                       resume_run=True)

pre_processing_results(run_to_load=run_to_load,
                       dataset_used=DATASET,
                       save=True)

# Next steps: Processing of the data to obtain the predictions.