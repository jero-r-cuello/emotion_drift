#%%
from vllm import LLM

# For generative models (runner=generate) only
llm = LLM(model="/home/models/Llama-3.1-8B-Instruct",
          enforce_eager=True)
output = llm.generate("Hello, my name is")
print(output)
# %%
