# Reproducibility

This document maps the paper onto the code and the result files in this archive.
Run every command from the repository root. Environment setup is in the
[README](README.md).

## What is in the archive

| Path | Content |
|---|---|
| `src/`, `scripts/` | The full pipeline: stimulus generation, inference with activation capture, annotation, probing, and analyses |
| `data/01_stimuli/` | The three stimulus corpora used in the paper |
| `data/04_annotated/` | The expert-annotated calibration subset and the LLM annotations used to select the judge |
| `results/`, `results_human_centric_rerun/` | The CSVs behind the figures and tables of the paper |

Not included, because of their size: the model responses (about 3 GB), the judge
annotations of the full corpora (about 6 GB), and the hidden-state activations.
Stages 2 to 4 below regenerate them.

### Naming

| Paper | Dataset name in the code | Short name in run ids and result files |
|---|---|---|
| AI-centric | `generated_prompts` | `aicentric` |
| Human 3rd-person | `generated_human_prompts` (probe dirs: `human_centric`) | `humanprompts`, `human3rd` |
| Human conversational | `generated_human_conversation_prompts` | `humanconv` |

Read-out positions are columns of the activation DataFrame:
`last_token_activation` is the prompt's last token and
`gen_last_token_activation` is the last token of the generated response.

## Pipeline

### 1. Stimulus generation (Section 3.1, Appendix B)

Requires `OPENROUTER_API_KEY`. Set `PROMPT_STYLE` at the top of the script to
`"ai_centric"`, `"human_centric"` (human 3rd-person) or `"human_conversation"`,
then run:

```bash
python src/llm/generate_prompts.py
```

The three generation templates, the emotion concepts and the Wikipedia
noise-injection seed are all defined in this script. The corpora used in the paper
are:

- `data/01_stimuli/generated_prompts/generated_emotional_prompts_batched.csv` (AI-centric)
- `data/01_stimuli/generated_human_prompts/generated_human_emotional_prompts_batched.csv` (human 3rd-person)
- `data/01_stimuli/generated_human_conversation_prompts/generated_human_conversation_prompts_batched.csv` (human conversational)

### 2. Inference and activation capture (Section 3.3)

Requires a GPU. One run per model and dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.llm.generate_with_hooks \
    --model <path-or-hf-id> --dataset generated_prompts
```

Use `--tp N` to split a large model across N GPUs. Responses are written to
`data/02_generated/outputs_<run>.jsonl`. Residual-stream activations of every
layer are captured at the prompt's last token and, in a second pass, at the last
generated token (`src/llm/activation_hooks.py`). They are written as chunks under
`data/03_activations/activations_<run>/`. The script prints the run id
(`<model>_<timestamp>`).

To consolidate the chunks into one DataFrame, set `RUN_TO_LOAD` and
`DATASET_USED` at the top of the script and run:

```bash
python scripts/output_and_activation_pairing.py   # -> data/03_activations/<dataset>_<run>.pkl
```

### 3. Annotation (Section 3.2)

Requires `OPENAI_API_KEY`. List the `(run_id, dataset)` pairs to annotate in
`RUNS` at the top of the script. Each response is annotated under Ekman,
Plutchik and GoEmotions by GPT-5-mini through the OpenAI Batch API:

```bash
python src/nlp/responses_annotation.py   # -> data/04_annotated/batch_results_<run>.jsonl
```

The annotation prompt and the three sets of category definitions (Appendix A)
are defined in this script. To join the labels to the activations, set
`RUN_TO_LOAD`, `DATASET_USED` and `ANNOTATIONS_FILE` at the top of the script and
run:

```bash
python scripts/annotations_and_activations_merge.py   # -> data/03_activations/<dataset>_<run>_FINAL.pkl
```

### 4. Linear probes (Section 3.3, Appendix C)

Configured through environment variables:

```bash
PROBE_LLM=Llama-2-7b-chat-hf PROBE_MODEL_DIM=4096 \
PROBE_DATASET=generated_prompts \
PROBE_ACT_COL=last_token_activation \
PROBE_DATA_PATH=data/03_activations/<dataset>_<run>_FINAL.pkl \
python src/probes/train_linear_probes_on_annotations.py
```

For each taxonomy and layer, this trains the probe and three shuffled-label
controls. It writes the metrics to
`results/probes_<dataset><position>_<model>/full_probing_metrics_<model>_final_F1.csv`
and the fitted probes to `models/`. Set `PROBE_ACT_COL=gen_last_token_activation`
for the generated-token position.

With the same variables, the test-set bootstrap confidence intervals (10,000
iterations) are computed by:

```bash
python src/probes/visualize_normalized_f1.py
```

### 5. Analyses (Section 4)

**RSA** (Section 4.1), with the same `PROBE_*` variables:

```bash
python src/probes/RSA_multilabel.py   # -> results/<model>_<dataset>/rsa_analysis/rsa_robust_metrics.csv
```

**Probe-weight cosines across taxonomies** (Section 4.1), read from the fitted
probes in `models/`:

```bash
PROBE_LLM=Llama-2-7b-chat-hf PROBE_DATASET=generated_prompts python src/probes/features_heatmap.py
```

**Same-class probe-weight cosines across domains or positions** (Section 4.2).
Set the two probe sets to compare. Two domains:

```bash
HEATMAP_LLM=Llama-2-7b-chat-hf HEATMAP_DS_A=generated_prompts HEATMAP_DS_B=human_centric \
python src/probes/interdataset_features_heatmap.py
```

For two positions, compare for example `generated_prompts` with
`generated_prompts_gen_last_token`.

### 6. Judge calibration (Appendix A)

These scripts run on the 132-response expert subset
(`data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv`):

```bash
python src/nlp/llm_annotators_test.py                    # the seven-annotator panel (OpenRouter)
python src/nlp/gpt_5_annotators_test.py                  # GPT-5 family, effort and verbosity settings (OpenAI Batch API)
python src/nlp/jsonl_to_csv.py                           # batch output -> gpt-5*-consolidated_annotations.csv
python src/nlp/annotators_performance_analysis.py        # per-annotator errors against the expert labels
python src/nlp/robust_annotators_performance_analysis.py # Krippendorff's alpha (MASI), Jaccard, bootstrap CIs
```

The last script reads `data/04_annotated/models_annotations_final.csv` and
produces the agreement results of Appendix A.

### 7. Stimulus checks (Appendix B)

```bash
python scripts/appendix_b_figures.py                 # intent-polarity flow and n-gram concentration
python src/llm/generated_prompt_variance_checks.py   # additional lexical-diversity checks
```

The intent-polarity flow reads the responses and annotations of stages 2 and 3.
The polarity of every concept and label is defined in `src/nlp/emotion_polarity.py`.

## Result files and where they appear in the paper

| File | Paper |
|---|---|
| `results/probes_generated_prompts_<model>/full_probing_metrics_<model>_final_F1_taxonomy.csv` | Section 4.1 selectivity; raw macro-F1 and shuffled-label controls (Appendix C) |
| `results/probes_generated_prompts_<model>/normalized_f1_bootstrap_ci_<model>.csv` | Confidence intervals of the Section 4.1 selectivity |
| `results/Llama-2-7b-chat-hf_generated_prompts/rsa_analysis/rsa_robust_metrics_taxonomy.csv` | Section 4.1 RSA |
| `results/Qwen2.5-14B-Instruct_generated_prompts/rsa_analysis/rsa_robust_metrics.csv` | RSA on Qwen2.5-14B-Instruct (Appendix D) |
| `results/slot_sweep_<model>_<domain>.csv` | Section 4.2 per-layer selectivity, per model, domain and read-out position (Appendix E) |
| `results_human_centric_rerun/04_capture_slots/capture_slots_ekman_plateau.csv` | Section 4.2 plateau selectivity, prompt vs. generated token, seven models |
| `results_human_centric_rerun/06_cross_domain_class_angles/allmodels_sameclass_cosine_summary.csv` | Section 4.2 same-class cosines across domains and positions (Appendix E tables) |

Files with the `_taxonomy` suffix hold the probing run of the taxonomy analysis
(Section 4.1).

Model abbreviations in `slot_sweep_*`: `Llama` = Llama-2-7b-chat-hf,
`Qwen` = Qwen2.5-14B-Instruct, `Llama31` = Llama-3.1-8B-Instruct,
`Qwen314` = Qwen3-14B, `GLM4` = GLM-4-32B, `Gemma4` = gemma-4-12b-it,
`Qwen36` = Qwen3.6-27B.

To regenerate the Section 4.1 selectivity, RSA, raw macro-F1 and taxonomy
figures from the included CSVs, no GPU needed:

```bash
python src/probes/paper_figures.py   # -> figures/manuscript_revised/
```
