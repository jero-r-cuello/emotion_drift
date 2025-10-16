#%%
## Try llm.generate
from vllm import LLM, SamplingParams

# For generative models (runner=generate) only
llm = LLM(model="mistralai/Magistral-Small-2506",#nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",#Qwen/Qwen2.5-32B",#/home/models/Meta-Llama-3-8B",#"/home/models/gpt-oss-20b",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )

sampling_params= SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
output = llm.generate("Hello, my name is",sampling_params=sampling_params)
print(output)
# %%
## Try llm.chat
from vllm import LLM, SamplingParams
import os

llm = LLM(model="/home/models/Llama-2-7b-chat-hf",#"/home/models/Meta-Llama-3-8B",#
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )

sampling_params= SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

conversation = [
    {"role": "user", "content": "Hello, my name is"}
]

try:
    outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
    print("\n>>> llm.chat() funcionó correctamente.")
    print("------------------------------------")

except ValueError as e:
    if "must provide a chat template" in str(e):
        print("\n>>> llm.chat() falló por falta de plantilla. Usando el método manual con llm.generate()...")
        
        # Obtenemos el tokenizador del motor de LLM
        tokenizer = llm.get_tokenizer()
        
        # Aplicamos manualmente la plantilla de chat de Llama 3
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True  # Crucial para que el modelo sepa que debe responder
        )
        
        print("\n--- Prompt Formateado Manualmente ---")
        print(repr(prompt)) # Usamos repr() para ver claramente los caracteres especiales
        print("------------------------------------")
        
        # Usamos llm.generate() con el prompt ya formateado
        outputs = llm.generate([prompt], sampling_params)

    else:
        # Si es un ValueError diferente, no lo manejamos y lo relanzamos
        print(f"\nSe encontró un ValueError inesperado: {e}")
        raise


generated_text = outputs[0].outputs[0].text
print(generated_text)
