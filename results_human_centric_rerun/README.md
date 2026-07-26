# Affective-representation analysis — 2026 regenerated run

Figures from a **full re-run of the pipeline on freshly generated 2026 data** (run id `20260625`).
All headline figures from the draft reproduce, plus several new analyses: the four activation
**capture slots**, the **prompt-vs-generated-token** representation split (the `*b` / `06c` figures),
cross-domain probe geometry, and a set of **exploratory behavioral analyses** (emotion drift, etc.).

## Provenance (novel data, not the original repo's results)

- **Models:** `Llama-2-7b-chat-hf` (32 layers) and `Qwen2.5-14B-Instruct` (48 layers).
- **Stimulus domains (3)** — standardized labels used throughout:
  - **`ai_centric`** = `generated_prompts` — prompts addressed *to* the model.
  - **`human_3rd`** = `generated_human_prompts` — 3rd-person human vignettes (person/addressee unrelated to the model).
    *(Earlier drafts called this `human_centric`; all names/labels are now `human_3rd`.)*
  - **`human_conv`** = `generated_human_conversation_prompts` — human content *shared with* the model (no call to action).
- **Responses:** ~130k model generations (2 models × 3 domains × ~21–22k each).
- **Annotation:** GPT-5-mini judge (Batch API, `effort=low`, the **unchanged psychologist prompt**), 3 taxonomies
  (Ekman / GoEmotions / Plutchik) → **390,288 annotations**, 0 parse failures.
- **Activations:** every prompt × layer stored **four capture slots**:
  `last_token` (prompt's last token, residual stream), `last_token_component` (MLP sub-block),
  `gen_last_token` (the *generated response's* last token, residual), `gen_last_token_component`.
- **Probes:** per-layer logistic regression (StandardScaler + L2, `C=0.1`, `class_weight=balanced`),
  **80/20 stratified split**. Metric = **chance-corrected normalized macro-F1** ("selectivity"):
  `(F1_real − F1_shuffled) / (1 − F1_shuffled)`.

**Novelty check:** vs the original reference CSVs, AI-centric Llama numbers are identical on **0/32 layers**
(mean |Δ macro-F1| ≈ 0.02) — genuinely regenerated. Human domains are wholly new stimuli.

---

## The two headline stories

**1. Taxonomy equivalence (reproduced, both archs).** Ekman/GoEmotions/Plutchik are decoded
equivalently by linear probes; RSA corroborates. Robust across Llama *and* Qwen. *Caveat (new):*
label-level equivalence is conditional — for AI-centric/conversational responses Plutchik routes to
Anticipation/Trust (no Ekman analog), so Ekman↔Plutchik agreement is high only for clearly-emotional
human-3rd content (0.63–0.76) and lower elsewhere.

**2. The model has two emotion representations (new).** Response-emotion (`gen_last_token`) is ~40%
more decodable than input-emotion (`last_token`), the two are **near-orthogonal**, input-emotion is
**domain-specific** and response-emotion is **domain-general**. For AI-centric prompts the affect is
*generated in the response*; for human stimuli it is *given in the prompt* (see 04 / 06c and the
behavioral corroboration below).

---

## Folder guide

### 01_selectivity_taxonomy_equivalence/
Chance-corrected macro-F1 per layer, one curve per taxonomy, all 6 model×domain sets.

### 01b_selectivity_genlast/
Same, but probes trained on the **generated-response** token. gen-last ≫ prompt-last for ai_centric.

### 02_rsa_llama/  &  02b_rsa_genlast/
RSA (50 stratified-bootstrap subsamples), prompt-token (02) and gen-token (02b). 

### 03_cross_test_asymmetry/  &  03b_cross_test_genlast/
Train on one domain, test on the other.

### 04_capture_slots/
Per-layer selectivity, one line per capture slot (prompt/gen × residual/MLP-component),
3 taxonomy panels per plot. **21 plots = 7 models × 3 domains**: `slots_<model>_<domain>.png`
(models: `llama` = Llama-2-7b, `qwen` = Qwen2.5-14B, `llama31`, `qwen314`, `glm4`, `gemma4`, `qwen36`).

| model | ai_centric | human_3rd | human_conv |
|---|---|---|---|
| **Llama-2-7b** (dense) | 0.277 → **0.392** ↑ | 0.381 → 0.363 ↓ | 0.393 → 0.337 ↓ |
| **Qwen2.5-14B** (dense) | 0.188 → **0.217** ↑ | 0.422 → 0.393 ↓ | 0.401 → 0.361 ↓ |
| **Llama-3.1-8B** (dense) | 0.253 → **0.285** ↑ | 0.408 → 0.326 ↓ | 0.366 → 0.298 ↓ |
| **Qwen3-14B** (dense, reasoning)* | 0.227 → **0.333** ↑ | 0.339 → 0.311 ↓ | 0.322 → 0.298 ↓ |
| **GLM-4-32B** (dense) | 0.253 → **0.358** ↑ | 0.386 → 0.362 ↓ | 0.440 → 0.405 ↓ |
| **gemma-4-12b** (MoE) | 0.288 → 0.249 ↓ | 0.366 → 0.250 ↓ | 0.445 → 0.314 ↓ |
| **Qwen3.6-27B** (hybrid, reasoning)* | 0.222 → 0.170 ↓ | 0.280 → 0.227 ↓ | 0.404 → 0.276 ↓ |


### 05_feature_heatmaps_within_dataset/  &  05b (gen)
Cosine between GoEmotions and Plutchik probe **weight vectors** at the plateau layer (Llama L20, Qwen L28).
Conceptually-corresponding categories share a direction (warm block-diagonal); unrelated pairs ≈ 0.

### 06_cross_domain_class_angles/ (symlog)  ·  06b_cross_domain_angles_genlast/  ·  06c_prompt_vs_gen_position/
Cosine between same-taxonomy probe weights across domains/positions (symlog color scale).

### 07_label_countplots/
Primary + multilabel emotion-frequency distributions per taxonomy, all 6 sets.

### 08_extras_exploratory/
Behavioral analyses of the responses themselves:
