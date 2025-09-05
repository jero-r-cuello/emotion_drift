#%%
from vllm import LLM, SamplingParams

# For generative models (runner=generate) only
llm = LLM(model="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",#Qwen/Qwen2.5-32B",#/home/models/Meta-Llama-3-8B",#"/home/models/gpt-oss-20b",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )
sampling_params= SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
output = llm.generate("Hello, my name is",sampling_params=sampling_params)
print(output)
# %%