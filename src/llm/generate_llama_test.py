# --- generate_llama_test.py (versión multi-capa) ---

import multiprocessing
import torch
from vllm import LLM, SamplingParams
import vllm.hook_store as hook_store

def main():
    with multiprocessing.Manager() as manager:
        print("--> Manager de procesos creado.")
        shared_hooks_dict = manager.dict()
        hook_store.ACTIVATION_HOOKS = shared_hooks_dict
        print("--> Diccionario compartido inyectado en vllm.hook_store.")

        # --- 1. CONFIGURACIÓN (EL CAMBIO ESTÁ AQUÍ) ---
        MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct"
        PROMPT = "La capital de Francia es"
        # Define una lista con todas las capas que te interesan.
        TARGET_LAYERS = [5, 15, 30]

        # --- 2. Inicializar vLLM ---
        print("--> Inicializando el motor de vLLM...")
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=16384,
            enforce_eager=True
        )
        print("--> Motor inicializado.")

        # --- 3. PREPARAR HOOKS (EL CAMBIO ESTÁ AQUÍ) ---
        print(f"--> Preparando hooks para las capas: {TARGET_LAYERS}...")
        shared_hooks_dict.clear()
        # Itera sobre tu lista de capas objetivo y prepara una entrada para cada una.
        for layer_idx in TARGET_LAYERS:
            shared_hooks_dict[layer_idx] = manager.list()
        print("--> Hooks preparados en el diccionario compartido.")

        # --- 4. Generar Texto ---
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
        print(f"--> Generando texto...")
        outputs = llm.generate(PROMPT, sampling_params)

        # --- 5. EXTRAER Y ANALIZAR (EL CAMBIO ESTÁ AQUÍ) ---
        print(f"\n--> Extrayendo activaciones desde el diccionario compartido...")

        # Itera sobre los resultados para cada capa que solicitaste.
        for layer_idx in TARGET_LAYERS:
            captured_list = shared_hooks_dict.get(layer_idx)

            if captured_list and len(captured_list) > 0:
                # Como antes, tomamos la primera captura que corresponde al prefill.
                activation_tensor = captured_list[0]
                print(f"\n--- Resultados para la Capa {layer_idx} ---")
                print(f"  ¡ÉXITO! Se capturó un tensor de activaciones.")
                print(f"  - Forma del tensor: {activation_tensor.shape}")
                
                # Opcional: imprimir el último token de esta capa
                last_token_activation = activation_tensor[-1]
                print(f"  - Primeros 5 valores del último token: {last_token_activation[:5]}")
            else:
                print(f"\n--- Resultados para la Capa {layer_idx} ---")
                print(f"  ERROR: No se encontraron activaciones.")

if __name__ == "__main__":
    main()