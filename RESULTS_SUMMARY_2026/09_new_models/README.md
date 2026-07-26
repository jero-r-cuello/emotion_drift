# New-model extension — Ekman, prompt vs gen last-token

Extends the affective-probe analysis to **5 additional models** to test whether the core
"two representations" result (emotion is *generated in the response* for AI-centric prompts,
but *given in the prompt* for human stimuli) generalizes across architectures.

## Result — plateau normalized-F1 (prompt-last vs gen-last)

| model | domain | prompt-last | gen-last | pattern |
|---|---|---|---|---|
| Llama-2-7b (dense, *original*) | ai_centric | 0.277 | **0.392** | **gen > prompt** |
|  | human_3rd | **0.381** | 0.363 | prompt > gen |
|  | human_conv | **0.393** | 0.337 | prompt > gen |
| Qwen2.5-14B (dense, *original*) | ai_centric | 0.188 | **0.217** | **gen > prompt** |
|  | human_3rd | **0.422** | 0.393 | prompt > gen |
|  | human_conv | **0.401** | 0.361 | prompt > gen |
| Llama-3.1-8B (dense) | ai_centric | 0.253 | **0.285** | **gen > prompt** |
|  | human_3rd | **0.408** | 0.326 | prompt > gen |
|  | human_conv | **0.366** | 0.298 | prompt > gen |
| Qwen3-14B (dense, reasoning) | ai_centric | 0.227 | **0.333** | **gen > prompt** |
|  | human_3rd | **0.339** | 0.311 | prompt > gen |
|  | human_conv | **0.322** | 0.298 | prompt > gen |
| GLM-4-32B (dense) | ai_centric | 0.253 | **0.358** | **gen > prompt** |
|  | human_3rd | **0.386** | 0.362 | prompt > gen |
|  | human_conv | **0.440** | 0.405 | prompt > gen |
| gemma-4-12b (MoE) | ai_centric | **0.288** | 0.249 | prompt > gen |
|  | human_3rd | **0.366** | 0.250 | prompt > gen |
|  | human_conv | **0.445** | 0.314 | prompt > gen |
| Qwen3.6-27B (hybrid, reasoning) | ai_centric | **0.222** | 0.170 | prompt > gen |
|  | human_3rd | **0.280** | 0.227 | prompt > gen |
|  | human_conv | **0.404** | 0.276 | prompt > gen |