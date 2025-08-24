from src.utils import pre_processing_results
from src.llm.emotional_assessment import assess
from src.llm.generate_with_hooks import generate
from transformers import AutoConfig

MODEL_NAME = "/home/models/Llama-2-7b-chat-hf" #!! For now, this script only works with models that use vllm/model_executor/models/llama.py
    
config = AutoConfig.from_pretrained(MODEL_NAME)
num_layers = config.num_hidden_layers
TARGET_LAYERS = [l for l in range(num_layers)]

DATASET = "llm_focused"

run_to_load = generate(model_name=MODEL_NAME,
                       target_layers=TARGET_LAYERS,
                       dataset_name=DATASET,
                       dataset_testing=False,
                       resume_run=False)

assess(MODEL_NAME,
       DATASET,
       dataset_testing=False,
       assessment_to_use="joy_intensity",
       resume_run=False)

pre_processing_results(run_to_load=run_to_load,
                       dataset_used=DATASET,
                       save=True)


# Next steps: Processing of the data to obtain the predictions.