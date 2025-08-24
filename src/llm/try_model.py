#%%
from vllm import LLM, SamplingParams

# For generative models (runner=generate) only
llm = LLM(model="Qwen/Qwen2.5-32B",#/home/models/Meta-Llama-3-8B",#"/home/models/gpt-oss-20b",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )
sampling_params= SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
output = llm.generate("Hello, my name is",sampling_params=sampling_params)
print(output)
# %%
import pandas as pd
from vllm import LLM, SamplingParams

df = pd.read_csv("/home/jcuello/emotion_drift/data/01_stimuli/llm_focused_situations/generated_prompts.csv")
df_long = pd.melt(df, var_name='emotion', value_name='situation')

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

prompts = [f'[INST] {scenario} [\INST]' for scenario in df_long.situation]

# For generative models (runner=generate) only
llm = LLM(model="/home/models/Llama-2-7b-chat-hf",
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=False,
          tensor_parallel_size=1)

outputs = llm.generate(prompts,sampling_params)

generated_responses = []
for output in outputs:
    generated_responses.append(output.outputs[0].text)

df_long["generated_responses"] = generated_responses