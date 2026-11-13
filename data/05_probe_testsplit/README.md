# Prompt-position Ekman probes + exact test splits

For 7 models x 3 domains: the linear Ekman probes trained on the **prompt read-out
position** (residual stream at the last prompt token), and the activations of the
**exact 20% test split** those probes were evaluated on, for layers in the 50–85% depth
plateau. Produced by `scripts/export_probe_testsplit.py` (driver: `scripts/export_probe_testsplit_all.sh`).

Models: `llama2` (Llama-2-7b-chat), `qwen25` (Qwen2.5-14B-Instruct), `llama31` (Llama-3.1-8B-Instruct),
`gemma4` (gemma-4-12b-it), `glm4` (GLM-4-32B-0414), `qwen314` (Qwen3-14B), `qwen36` (Qwen3.6-27B).
Domains: `ai_centric` (generated_prompts), `human_3rd` (generated_human_prompts),
`human_conv` (generated_human_conversation_prompts).

## Layout

```
<model>/<domain>/
  probes/layer_<L>.joblib     sklearn Pipeline(StandardScaler, LogisticRegression) -> 7 classes
  test_activations_fp16.npz   prompt_key, prompt_id, ekman_label, layer_<L> (n_test x d, float16)
  test_split.csv              prompt_key, prompt_id, ekman_label (same row order as the npz)
  probe_eval.csv              per-layer test macro-F1 (fp32, fp16), logged F1, split_verified
  meta.json                   run id, n_layers, plateau layers, exported layers, n_train/n_test, d
```

Here: 5 layers evenly spread over the plateau (`meta.json: repo_layers`).
The **full plateau** (every layer with 0.5 <= L/n_layers <= 0.85, float32, one npz per layer)
is on the cluster at `/is/cluster/fast/jgeiping/probe_testsplit_export/<model>/<domain>/`.

Labels are the *primary* (first-listed) Ekman label of the judge annotation; the 7 classes are
Anger, Disgust, Enjoyment, Fear, Neutral, Sadness, Surprise.

## Split / provenance

Identical to `scripts/merge_and_probe.py`: labelled prompts sorted by prompt_id ->
`sample(n=8000, random_state=42)` -> drop classes with <5 samples ->
`train_test_split(test_size=0.2, random_state=42, stratify=y)`.

* `llama31, gemma4, glm4, qwen314, qwen36`: probes are the ones saved by `merge_and_probe.py`
  (Qwen3 models: the 20260724 reasoning rerun). Re-scoring them on the reconstructed test split
  reproduces the logged macro-F1 exactly (`split_verified=True` in `probe_eval.csv`).
* `llama2, qwen25`: no probes from this recipe were saved, so they were retrained with the
  identical recipe on the identical split (`probe_source=retrained_same_recipe`).

float16 storage does not change any probe prediction on these sets (`test_macro_f1_fp16`).

## Loading

```python
import numpy as np, joblib
d = np.load("llama31/ai_centric/test_activations_fp16.npz")
clf = joblib.load("llama31/ai_centric/probes/layer_22.joblib")
X = d["layer_22"].astype(np.float32)
pred = clf.predict(X)            # compare with d["ekman_label"]; d["prompt_key"] identifies rows
```
Probes were saved with scikit-learn 1.9.0 / numpy 2.3.5.
