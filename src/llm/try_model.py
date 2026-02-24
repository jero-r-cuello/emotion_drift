#%%
## Try llm.generate
from vllm import LLM

# For generative models (runner=generate) only
llm = LLM(model="/home/models/Qwen2.5-14B-Instruct",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )

output = llm.generate("Hello, my name is")
print(output)
# %%
## Try llm.chat
from vllm import LLM

llm = LLM(model="/home/models/Qwen2.5-14B-Instruct",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=4092,
          enforce_eager=True
          )

conversation = [
    {"role": "user", "content": "Hello, my name is"}
]

try:
    outputs = llm.chat(conversation,
                   use_tqdm=False)
    print("\n[REPO INFO] llm.chat() worked correctly.")
    print("------------------------------------")

except ValueError as e:
    if "must provide a chat template" in str(e):
        print("\nllm.chat() failed due to lack of template. Using the manual method with llm.generate()...")
        
        tokenizer = llm.get_tokenizer()
        
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("\nManually formatted prompt")
        print(repr(prompt))
        print("------------------------------------")
        
        outputs = llm.generate([prompt])

    else:
        print(f"\nAn unexpected ValueError was encountered: {e}")
        raise

generated_text = outputs[0].outputs[0].text
print(generated_text)
# %%
