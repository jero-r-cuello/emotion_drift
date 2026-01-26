# Latent Affective Structure in LLMs: AI-Centric Emotion Analysis

This repository contains the code and data for the research project investigating the latent affective structure of Large Language Models (LLMs), specifically Llama-2-7b and Qwen2.5-14B. 

## 📖 Project Overview

The core research question driving this project is: **Does the latent space of LLMs reflect an affective structure consistent with human psychological theories when subjected to ontologically relevant (AI-centric) stimuli?**

Unlike traditional studies utilizing human-centric datasets (e.g., Reddit posts), this project generates and evaluates **AI-centric stimuli** (prompts relevant to a digital entity's existence (e.g., system updates, deletion threats)) to analyze emotional representations within the model's residual stream.

### Key Methodologies
1.  **Stimuli Generation:** Using "Generator Models" (Claude, Gemini, Grok) with noise injection to create diverse AI-centric prompts.
2.  **Activation Extraction:** Custom hooks in `vLLM` to capture internal hidden states during inference.
3.  **Multi-Theory Annotation:** Using a Model-as-a-Judge approach to annotate responses based on three psychological frameworks:
    *   **Ekman:** 6 Basic Emotions.
    *   **Plutchik:** Wheel of Emotions (8 basic emotions).
    *   **GoEmotions:** Fine-grained taxonomy (27 categories).
4.  **Analysis:** Linear Probing (separability) and Representational Similarity Analysis (RSA).

---

## 🛠️ Installation & Requirements

This project relies on `vLLM` for inference and `PyTorch` for analysis.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/jero-r-cuello-emotion_drift.git
    cd jero-r-cuello-emotion_drift
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (Conda/venv).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Set up your API keys for the generator/judge models (OpenAI/OpenRouter):
    ```bash
    export OPENAI_API_KEY="your_key_here"
    ```

---

## 🚀 Workflow & Usage

The analysis pipeline consists of four distinct stages.

### 1. Prompt Generation
Generate synthetic, AI-centric emotional prompts using external APIs.

*   **Generate Prompts:**
    ```bash
    python src/llm/generate_prompts.py
    ```
    *Output:* `data/01_stimuli/generated_prompts/`

*   **Validate Variance:**
    Check the lexical diversity of the generated prompts.
    ```bash
    python src/llm/generated_prompt_variance_checks.py
    ```

### 2. Experiment (Inference & Extraction)
Run the target LLM (Llama-2 or Qwen) to generate responses and extract hidden states. 

*   **Run Inference:**
    This script uses modified `vLLM` model definitions (located in `vllm_changes/`) to hook into the forward pass and save activations to disk.
    ```bash
    # Adjust CUDA_VISIBLE_DEVICES as needed
    CUDA_VISIBLE_DEVICES=0 python -m src.llm.generate_with_hooks
    ```
    *Output:* 
    *   Text responses: `data/02_generated/outputs_*.jsonl`
    *   Activations: `data/03_activations/activations_*/`

*   *(Optional)* **Emotional Assessment (Questionnaires):**
    Apply psychological questionnaires as a second turn in the conversation.
    ```bash
    python src/llm/emotional_assessment.py
    ```

### 3. Annotation (Model-as-a-Judge)
Annotate the generated responses using GPT-5-mini, based on specific taxonomies.

*   **Run Annotation:**
    ```bash
    python src/nlp/responses_annotation.py
    ```
    *Output:* `data/04_annotated/annotated_results.jsonl`

### 4. Data Processing & Analysis
Merge the text inputs, annotations, and vector activations into a unified format for analysis.

1.  **Pair Outputs with Activations:**
    Matches the JSONL text outputs with the `.pt` tensor files.
    ```bash
    python scripts/output_and_activation_pairing.py
    ```

2.  **Merge Annotations:**
    Adds the emotion labels to the dataset.
    ```bash
    python scripts/annotations_and_activations_merge.py
    ```

3.  **Descriptive Analysis:**
    Analyze assessment responses distributions and reluctance rates.
    ```bash
    python src/nlp/rating_analysis.py
    ```

4.  **Linear Probing (Classifiers):**
    Train Logistic Regression probes on the residual stream to predict emotion labels.
    ```bash
    # Train probes on annotated responses
    python src/probes/train_linear_probes_on_annotations.py
    
    # (Optional) Train probes on prompt-implied emotions
    # python src/probes/train_linear_probe_prompt_emotion.py
    ```

5.  **Feature Visualization (Heatmaps):**
    Compare weight vectors across taxonomies.
    ```bash
    python src/probes/features_heatmap.py
    ```

6.  **Representational Similarity Analysis (RSA):**
    Compute RDMs to compare the geometry of the model's activation space against theoretical emotion models.
    ```bash
    python src/probes/RSA_multilabel.py
    ```


---

## 📂 Directory Structure

```text
├── data/
│   ├── 01_stimuli/       # Input datasets (Human-centric & AI-centric prompts)
│   ├── 02_generated/     # LLM text outputs and assessment questionnaires
│   ├── 03_activations/   # Saved PyTorch tensors (Hidden States) and merged PKL files
│   └── 04_annotated/     # JSONL files containing emotion labels from the Judge
├── results/              # CSV metrics (F1 scores, RSA correlations)
├── scripts/              # Data processing and merging utilities
├── src/
│   ├── llm/              # Generation, Hooks, and Model loading
│   ├── nlp/              # Text processing, Annotation logic, Reluctance analysis
│   └── probes/           # Linear Probing, RSA, and Heatmap generation
└── vllm_changes/         # Monkey-patched model files for vLLM activation extraction

## ⚠️ Note on vLLM Modifications
This project uses custom logic to extract activations from `vLLM`. The files in `vllm_changes/` (`llama.py`, `qwen2.py`, `hook_store.py`) inject a hook into the `forward` pass of the model to offload hidden states to the CPU/Disk during inference. `src/llm/generate_with_hooks.py` handles the orchestration of these hooks.