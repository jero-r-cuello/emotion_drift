# New-model extension — Ekman, prompt vs gen last-token (2026-07 run `20260702`)

Extends the affective-probe analysis to **5 additional models** to test whether the core
"two representations" result (emotion is *generated in the response* for AI-centric prompts,
but *given in the prompt* for human stimuli) generalizes across architectures.

## Setup
- **Models:** Llama-3.1-8B-Instruct, Qwen3-14B, GLM-4-32B-0414 (dense text) · gemma-4-12b-it (MoE),
  Qwen3.6-27B (hybrid gated-deltanet) — the last two are multimodal-capable but run **text-only**.
- **Taxonomy:** Ekman only (7 classes). **Stimuli:** the same 3 domains × ~21–22k prompts.
- **Capture:** residual stream at the **prompt's last token** and the **generated response's last token**.
  Text models via forward hooks; VL/hybrid models via vLLM's `aux_hidden_state` mechanism
  (residual only — see `activation_hooks.py`).
- **Probe:** per-layer logistic regression (StandardScaler, C=0.1, balanced), 8k stratified subsample,
  chance-corrected normalized macro-F1; plateau = mid–upper layers.
- **Metric table:** `newmodel_ekman_selectivity.csv` (plateau normalized-F1 per model × domain × slot).

## Figures (`figures/`, from `scripts/plot_newmodel_gen_selectivity.py`)
- **`gen_vs_prompt_plateau.png`** — plateau normalized-F1 bars, prompt-last vs gen-last, per model × domain.
  The AI-centric panel is the headline: gen>prompt (red taller) for the three dense models, prompt>gen for
  gemma-4/Qwen3.6; every human-domain bar is prompt>gen.
- **`gen_selectivity_by_layer.png`** — full per-layer curves (5 models × 3 domains), gen-last (red dashed)
  vs prompt-last (blue), x = relative depth. Same story resolved across the depth of the network.
  (Point estimates from the probe CSVs — no bootstrap bands; the 4 core models' probe weights weren't
  retained, so the weight-geometry figures 05b/06b are only reproducible for GLM without re-probing.)

## Result — plateau normalized-F1 (prompt-last vs gen-last)

| model | domain | prompt-last | gen-last | pattern |
|---|---|---|---|---|
| Llama-3.1-8B (dense) | ai_centric | 0.253 | **0.285** | **gen > prompt** |
|  | human_3rd | **0.408** | 0.326 | prompt > gen |
|  | human_conv | **0.366** | 0.298 | prompt > gen |
| Qwen3-14B (dense) | ai_centric | 0.188 | **0.225** | **gen > prompt** |
|  | human_3rd | **0.325** | 0.258 | prompt > gen |
|  | human_conv | **0.437** | 0.340 | prompt > gen |
| GLM-4-32B (dense) | ai_centric | 0.253 | **0.358** | **gen > prompt** |
|  | human_3rd | **0.386** | 0.362 | prompt > gen |
|  | human_conv | **0.440** | 0.405 | prompt > gen |
| gemma-4-12b (MoE) | ai_centric | **0.288** | 0.249 | prompt > gen |
|  | human_3rd | **0.366** | 0.250 | prompt > gen |
|  | human_conv | **0.445** | 0.314 | prompt > gen |
| Qwen3.6-27B (hybrid) | ai_centric | **0.272** | 0.199 | prompt > gen |
|  | human_3rd | **0.385** | 0.237 | prompt > gen |
|  | human_conv | **0.527** | 0.352 | prompt > gen |

**All 5 new models done (7 total with the original Llama-2 + Qwen2.5).** Summary: **5 of 7 models
reproduce** the AI-centric gen > prompt dissociation — all the *dense* text models (Llama-2, Qwen2.5,
Llama-3.1-8B, Qwen3-14B, GLM-4-32B); the **2 exceptions (gemma-4 MoE, Qwen3.6 hybrid) are prompt-dominant
in every domain**. All human-domain cells are prompt > gen for *every* model — only the *AI-centric* cell
differs, and only for those two.

## Findings
1. **5 of 7 models reproduce the double dissociation** (original Llama-2, Qwen2.5; new Llama-3.1-8B,
   Qwen3-14B, GLM-4-32B): AI-centric emotion is stronger at the **gen** token (generated in the response),
   human-domain emotion is stronger at the **prompt** token (given in the stimulus). This holds from
   **7B to 32B** and across old→newest models — notably **GLM-4-32B (32B, newest) reproduces**, so the
   effect is not about scale or recency.
2. **2 models (gemma-4-12b, Qwen3.6-27B) break it — prompt > gen in *all* domains including AI-centric.**
   AI-centric plateau (prompt / gen): gemma 0.288/0.249, Qwen3.6 0.272/0.199.

### What we can and cannot claim about *why* those two differ
The two exceptions happen to be the ones with a (multimodal) vision path AND non-standard text backbones
(gemma-4 is MoE; Qwen3.6 is a Qwen3-Next-style **hybrid** with gated-deltanet + attention layers), and
they are newer / differently trained. With only 6 models and all of these factors confounded, **we cannot
attribute the difference to any single cause** — and in particular **NOT to "being vision-language":**
inference is text-only (`--limit-mm-per-prompt`, vision skipped), so the vision path never runs. Earlier
phrasing that called these "VL models" and the split "text-vs-VL / architectural" was an over-claim. The
honest statement is: **two specific models (gemma-4, Qwen3.6) do not show the AI-centric gen>prompt
effect, for reasons this 6-model dataset cannot isolate.**

### What the control DID establish (capture method is not the confound)
Those two are read out via vLLM's `aux_hidden_state` mechanism rather than forward hooks (their decoders
run in a compiled path hooks can't reach). To rule out that the *readout method* explains their behavior,
we captured a text model (Llama-3.1-8B) via **both** methods on the same prompts with **greedy decoding**
(so the generated sequences are identical and the *gen*-last token is comparable, not just the
deterministic prompt-last) — `scripts/compare_capture.py`:

**cosine(hooks, aux): prompt-last = 1.00000, gen-last = 1.00000** (min 1.00000, all 320+320 layer-vectors
> 0.999). Bit-for-bit identical for BOTH slots — critically including gen-last, which is what the
prompt-dominance result hinges on. (An earlier control only checked prompt-last under stochastic
decoding; this closes the gen-last gap.)

Two further sanity checks specific to gemma-4/Qwen3.6 (where hooks can't fire, so no ground truth):
their gen-last F1 **differs** from prompt-last F1 (so gen-last is not aliased to the prompt token — the
classic wrong-position bug would make them equal), and gen-last F1 is 0.25–0.35 (well above chance, so
not a degenerate/constant vector). So gemma-4/Qwen3.6's prompt-dominance is a **real property of those
models' representations, not a capture bug.** *What* about those models causes it (MoE? hybrid attention?
training recipe?) is open. Residual caveat: gemma (MoE) / Qwen3.6 (hybrid) use slightly different
`_maybe_add_hidden_state` code paths than Llama, validated only indirectly.

*Table auto-updates as GLM-4-32B finishes (a 5th new model, `/fast`-read-bound).*
