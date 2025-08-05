#%%
from vllm import LLM

# For generative models (runner=generate) only
llm = LLM(model="/home/models/Meta-Llama-3-8B",
          enforce_eager=True)
output = llm.generate("Hello, my name is")
print(output)
# %%
