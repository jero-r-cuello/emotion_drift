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
Chance-corrected macro-F1 per layer, one curve per taxonomy, all 6 model×domain sets. Curves overlap,
plateau at **L15–25** (Llama) / **L20–35** (Qwen). Llama ai_centric ≈ 0.28 (Ekman); Qwen ≈ 0.17–0.22.
Human domains are *more* decodable at the prompt token (0.35–0.39) — see 04.

### 01b_selectivity_genlast/
Same, but probes trained on the **generated-response** token. gen-last ≫ prompt-last for ai_centric.

### 02_rsa_llama/  &  02b_rsa_genlast/
RSA (50 stratified-bootstrap subsamples), prompt-token (02) and gen-token (02b). Label-RDM Spearman
plateau ≈ 0.07–0.11 (prompt). **Dissociation:** gen-token RSA correlation is *lower* than prompt for
Ekman (0.047 vs 0.111) even though the probe decodes gen-token *better* — the response representation
is linearly separable but concentrated on the dominant response emotion, so less RDM-structured.

### 03_cross_test_asymmetry/  &  03b_cross_test_genlast/
Train on one domain, test on the other (bootstrap mean, 95% CI). **Transfer asymmetry (prompt token, Ekman, plateau):**
- ai_centric→ai_centric 0.374 → **ai→human 0.269** (−28%).
- human_3rd→human_3rd **0.464** (best in-domain) → **human→ai 0.255** (−45%).
- The strongest in-distribution probe (human_3rd) **collapses most** out-of-distribution.
- 3-pair version: the two human domains cross-transfer best to each other.

**03b — same on the generated-response token:** cross-domain transfer is **stronger and more symmetric** than
on the prompt token (Ekman plateau): ai↔human_3rd 0.269/**0.316**, ai↔human_conv 0.323/0.352,
human_3rd↔human_conv 0.384/0.332. The human→ai direction rises 0.255→0.316 vs the prompt token — direct
evidence that the **response-emotion representation is domain-general** (matches 06b's higher gen cross-domain
cosine). So the asymmetry that dominates the input representation largely dissolves in the response representation.

### 04_capture_slots/
Per-layer selectivity, one line per capture slot (prompt/gen × residual/MLP-component),
3 taxonomy panels per plot. **21 plots = 7 models × 3 domains**: `slots_<model>_<domain>.png`
(models: `llama` = Llama-2-7b, `qwen` = Qwen2.5-14B, `llama31`, `qwen314`, `glm4`, `gemma4`, `qwen36`).

**Double dissociation (where affect lives)** — Ekman plateau (50-85% depth), prompt-last vs gen-last;
full table in `capture_slots_ekman_plateau.csv`:

| model | ai_centric | human_3rd | human_conv |
|---|---|---|---|
| **Llama-2-7b** (dense) | 0.277 → **0.392** ↑ | 0.381 → 0.363 ↓ | 0.393 → 0.337 ↓ |
| **Qwen2.5-14B** (dense) | 0.188 → **0.217** ↑ | 0.422 → 0.393 ↓ | 0.401 → 0.361 ↓ |
| **Llama-3.1-8B** (dense) | 0.253 → **0.285** ↑ | 0.408 → 0.326 ↓ | 0.366 → 0.298 ↓ |
| **Qwen3-14B** (dense) | 0.188 → **0.225** ↑ | 0.325 → 0.258 ↓ | 0.437 → 0.340 ↓ |
| **GLM-4-32B** (dense) | 0.253 → **0.358** ↑ | 0.386 → 0.362 ↓ | 0.440 → 0.405 ↓ |
| **gemma-4-12b** (MoE) | 0.288 → 0.249 ↓ | 0.366 → 0.250 ↓ | 0.445 → 0.314 ↓ |
| **Qwen3.6-27B** (hybrid) | 0.272 → 0.199 ↓ | 0.385 → 0.237 ↓ | 0.527 → 0.352 ↓ |

(cells read `prompt-last → gen-last`; ↑ = gen > prompt.)

**gen-last > prompt-last only for ai_centric, and only in the 5 dense models** — every human-domain cell is
prompt > gen for every model. Residual ≥ MLP-component throughout. The two exceptions (gemma-4 MoE,
Qwen3.6 hybrid) are prompt-dominant everywhere; that split is **not** attributable to their vision towers
(inference is text-only) and is confounded across MoE/hybrid-attention/training — see
`RESULTS_SUMMARY_2026/09_new_models/README.md`. The draft's core figures all used the *prompt* token →
**conservative**.

The dissociation also holds in **go_emotions and plutchik**, not just Ekman (see the 3 panels per plot),
so it is not a taxonomy artifact.

### 05_feature_heatmaps_within_dataset/  &  05b (gen)
Cosine between GoEmotions and Plutchik probe **weight vectors** at the plateau layer (Llama L20, Qwen L28).
Conceptually-corresponding categories share a direction (warm block-diagonal); unrelated pairs ≈ 0.

### 06_cross_domain_class_angles/ (symlog)  ·  06b_cross_domain_angles_genlast/  ·  06c_prompt_vs_gen_position/
Cosine between same-taxonomy probe weights across domains/positions (symlog color scale).
- **Cross-domain (prompt):** same-class cosine near-orthogonal (≈0.01–0.03) — domain-specific.
- **Cross-domain (gen, 06b):** more aligned (≈0.08–0.19) — response-emotion is domain-general.
- **Prompt vs gen (06c):** near-orthogonal (≈0.02–0.09, both archs) — **two distinct representations**.
- *(`06_cross_domain_probe_angles/` is the older per-layer version, kept for reference.)*
- **Caveat:** raw high-dim logreg-weight cosines can read low even when the predictive direction partly
  transfers (cross-test shows ai→human ≈ 0.27). A within-domain 50/50-split control would calibrate this — not run.

### 07_label_countplots/
Primary + multilabel emotion-frequency distributions per taxonomy, all 6 sets.

### 08_extras_exploratory/
Behavioral analyses of the responses themselves:
- **Emotion drift** (`emotion_drift_stayrate_*`, `drift_confusion_*`): when a prompt is seeded for a basic
  emotion, only **Enjoyment & Sadness survive**; Anger/Disgust/Surprise drift away. Target is domain-gated:
  ai_centric → **Neutral** (deflect/helpful), human → **Sadness** (empathize). Qwen is far more suppressed
  as an AI (68% Neutral vs Llama 23%). *An RLHF/alignment signature.*
- **Behavioral independence** (corroborates the "generated" claim): NMI between seeded and response emotion —
  ai_centric **0.05–0.12** (response ~independent of prompt = *generated*) vs human 0.24–0.29 (*given*).
- **Per-emotion prompt→gen** (`perclass_prompt_vs_gen_*`): the ai_centric gen advantage is **broad across
  emotions** (Anger +0.19, Neutral +0.17, Fear +0.13); human domains show no such gain. Neutral gains at
  the gen token in *every* domain (engage-vs-deflect is a response-level decision).
- **Cross-taxonomy agreement** (`cross_taxonomy_agreement`): Ekman↔Plutchik high only for human_3rd.
- **Drift ↔ decodability** (`drift_vs_decodability_llama`): **Pearson r = 0.93** — emotions the model won't
  express are the ones the probe can't decode. Confound noted (probe F1 is stratified + class-balanced, so
  this is genuine data-scarcity for rare emotions, not a split artifact).
- **Response-emotion by domain** (`response_emotion_by_domain_*`).

---

## Status / pending
- **03b cross-test-gen:** computing (nested-pkl loads are CPU-slow to unpickle over the shared mount).
- **Within-domain split control** to calibrate the 06 cross-domain cosines — not run.
- **Model extension** (Qwen3-14B / Qwen3.6-27B / gemma-4-12b): Qwen3-14B validated on the pipeline;
  Qwen3.6-27B and gemma-4 are vision-language archs needing a hook layer-path fix (+ vLLM bump for gemma) —
  in progress on a separate track.
- Full per-layer heatmap stacks live in `figures/` (96 Llama / 144 Qwen PNGs each); only plateau-layer
  representatives are copied here.
