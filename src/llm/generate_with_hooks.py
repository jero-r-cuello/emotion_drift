"""
Batched generation + residual-stream activation extraction.

Rewritten for vLLM 0.22.x (v1 engine). Activations are captured by runtime
forward hooks (see src/llm/activation_hooks.py) registered on every worker via
collective_rpc, instead of overwriting vLLM's model source. Prompts run through
llm.chat in batches (full continuous batching), and the captured last-token (or
full) residuals are joined back to prompts by their token ids.

Multi-GPU is via vLLM's native flags, not a hand-rolled sharding layer:
  --tp  tensor_parallel_size      (split one model across GPUs; for big models)
  --pp  pipeline_parallel_size    (split layers across GPUs; for big models)
Throughput for short prompts now comes from continuous batching on a single
GPU (the old code ran one prompt at a time). To use several GPUs at once,
run one model per GPU (different CUDA_VISIBLE_DEVICES). In-process
data_parallel_size>1 is NOT supported by the offline LLM API and is rejected.

Example:
  # single GPU, batched
  CUDA_VISIBLE_DEVICES=0 python -m src.llm.generate_with_hooks \
      --model /home/models/Qwen2.5-14B-Instruct --dataset generated_human_prompts
  # big model split across 4 GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.llm.generate_with_hooks \
      --model /home/models/Qwen2.5-32B-Instruct --tp 4
"""
import os
# collective_rpc needs to cloudpickle the hook closures into worker processes.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import json
import glob
import pickle
import datetime

from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoConfig

from src.utils import load_dataset
from src.llm.activation_hooks import make_register_fn, make_fetch_fn, merge_worker_caps


def generate(model_name, target_layers, dataset_name="generated_human_prompts",
             dataset_testing=False, resume_run=False, poolings=("last_token",),
             tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
             gpu_memory_utilization=0.90, max_model_len=4096,
             chunk_size=256, max_tokens=256, temperature=1.0, top_p=1.0,
             home_model_dir="/home/models/", run_id=None,
             shard_index=0, num_shards=1, activations_base=None,
             capture_response=True, limit_mm_per_prompt=None):
    """
    Generate responses for every prompt in `dataset_name` and save, per prompt,
    the residual-stream activation (last prompt token by default) at each layer
    in `target_layers`. Results stream to data/02_generated/outputs_<run>.jsonl
    and data/03_activations/activations_<run>/prompt_<i>_layer_<l>.pt, matching
    the layout expected by src/utils.py and the pairing scripts.
    """
    project_root = os.getcwd()
    if data_parallel_size > 1:
        raise ValueError(
            "data_parallel_size>1 is not supported by the in-process offline LLM API "
            "(it requires vLLM's multi-process DP launcher). For multi-GPU here, use "
            "--tp/--pp for one model, or run separate processes per GPU subset / one "
            "model per GPU. Single-GPU batched generation is already fast for short "
            "prompts. See docs/RUN_human_centric.md.")
    prompt_data = load_dataset(dataset_name, testing=dataset_testing)
    print(f"\n[REPO INFO] Loaded {len(prompt_data)} prompts from '{dataset_name}'.")

    print("\n[REPO INFO] Initializing the LLM...\n")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        # Disable chunked prefill so vLLM never splits a single prompt's prefill
        # across forwards (it schedules whole prompts per forward instead). This
        # is what the capture rule (query_len == seq_len) relies on; with chunked
        # prefill ON, prompts straddling the per-forward token budget get split
        # and silently missed (~1%). Prefix caching off so each prefill is a
        # clean, uncached pass. (Requires max_num_batched_tokens >= longest prompt.)
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,            # forward hooks require eager (no CUDA graphs)
        enable_prefix_caching=False,
        # For vision-language checkpoints (gemma-4, Qwen3.6-VL): load text-only and
        # skip multimodal profiling/memory. Ignored for pure-text models (None).
        **({"limit_mm_per_prompt": limit_mm_per_prompt} if limit_mm_per_prompt else {}),
    )
    print("\n[REPO INFO] LLM initialized.\n")

    n_hooks = llm.collective_rpc(make_register_fn(target_layers, poolings))
    fetch = make_fetch_fn()
    print(f"[REPO INFO] Activation hooks registered per worker: {n_hooks} "
          f"(poolings={list(poolings)}).")

    if model_name.startswith(home_model_dir):
        model_name = model_name[len(home_model_dir):]
    safe_model_name = model_name.replace("/", "_")

    output_dir = os.path.join(project_root, "data", "02_generated")
    os.makedirs(output_dir, exist_ok=True)

    # Per-shard outputs file (the activations dir is shared across shards; its
    # filenames carry the global prompt index, so shards never collide). The
    # launcher (scripts/run_dp.py) merges the per-shard outputs afterwards.
    shard_suffix = f"_shard{shard_index}of{num_shards}" if num_shards > 1 else ""
    if num_shards > 1:
        print(f"[REPO INFO] Shard {shard_index}/{num_shards}: processing global indices "
              f"i % {num_shards} == {shard_index}.")

    # ---- resume / run naming ----
    processed_prompt_keys = set()
    results_filepath = None
    if resume_run:
        existing = sorted(glob.glob(os.path.join(output_dir, f"outputs_{safe_model_name}_*{shard_suffix}.jsonl")), reverse=True)
        if existing:
            results_filepath = existing[0]
            with open(results_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_prompt_keys.add(json.loads(line)["prompt_key"])
                    except json.JSONDecodeError:
                        print(f"[WARNING] Skipping corrupted line in {results_filepath}")
            print(f"[REPO INFO] Resuming {results_filepath}: {len(processed_prompt_keys)} prompts already done.")

    if results_filepath is None:
        timestamp = run_id if run_id else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filepath = os.path.join(output_dir, f"outputs_{safe_model_name}_{timestamp}{shard_suffix}.jsonl")
    else:
        timestamp = os.path.basename(results_filepath).replace(f"outputs_{safe_model_name}_", "").replace(f"{shard_suffix}.jsonl", "").replace(".jsonl", "")
    # Activations (the bulk) can be redirected to e.g. a large network drive;
    # the small outputs JSONL stays under the repo. Consolidation reads chunk
    # pkls from here (pass the same base to utils.consolidate_activations).
    act_root = activations_base if activations_base else os.path.join(project_root, "data", "03_activations")
    activations_run_dir = os.path.join(act_root, f"activations_{safe_model_name}_{timestamp}")
    os.makedirs(activations_run_dir, exist_ok=True)
    print(f"[REPO INFO] Outputs -> {results_filepath}")
    print(f"[REPO INFO] Activations -> {activations_run_dir}\n")

    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    # Work list carries the GLOBAL index, so prompt_key stays aligned with the
    # dataset rows (downstream pairing parses prompt_<i>) even when resuming.
    work = [(i, item) for i, item in enumerate(prompt_data)
            if f"prompt_{i}" not in processed_prompt_keys
            and (num_shards == 1 or i % num_shards == shard_index)]
    print(f"[REPO INFO] {len(work)} prompts to process in chunks of {chunk_size}.\n")

    with open(results_filepath, "a", encoding="utf-8") as f_out:
        for chunk_idx, c0 in enumerate(range(0, len(work), chunk_size)):
            chunk = work[c0:c0 + chunk_size]
            conversations = [[{"role": "user", "content": item["prompt_text"]}] for _, item in chunk]

            # Phase 1: generate responses; the hook captures the LAST PROMPT token.
            outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
            caps_prompt = merge_worker_caps(llm.collective_rpc(fetch))

            # Phase 2 (optional): prefill [prompt + response] token ids so the last
            # token is the model's last GENERATED token; the same hook captures it.
            # Equivalent to the residual at that token during generation (causal),
            # and reuses the validated prefill capture. max_tokens=1 (no new gen).
            caps_gen, full_keys = {}, []
            if capture_response:
                tok_prompts = []
                for out in outputs:
                    full = list(out.prompt_token_ids) + list(out.outputs[0].token_ids)
                    full_keys.append(tuple(full))
                    tok_prompts.append(TokensPrompt(prompt_token_ids=full))
                llm.generate(tok_prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
                caps_gen = merge_worker_caps(llm.collective_rpc(fetch))

            # One pickle per chunk: {prompt_id: {layer_idx: {field: np.ndarray}}}.
            # Per layer/prompt: last_token[_component] (prompt last token, residual
            # stream / partial component) + gen_last_token[_component] (response
            # last token), the latter prefixed "gen_".
            chunk_acts = {}
            n_missing = 0
            for j, ((gi, item), out) in enumerate(zip(chunk, outputs)):
                prompt_key = f"prompt_{gi}"
                generated_text = out.outputs[0].text
                layer_map = caps_prompt.get(tuple(out.prompt_token_ids))
                if not layer_map:
                    n_missing += 1
                else:
                    gen_map = caps_gen.get(full_keys[j]) if capture_response else None
                    merged = {}
                    for layer, d in layer_map.items():
                        md = dict(d)
                        if gen_map and layer in gen_map:
                            for pn, vec in gen_map[layer].items():
                                md["gen_" + pn] = vec
                        merged[layer] = md
                    chunk_acts[gi] = merged
                f_out_record = {
                    "prompt_key": prompt_key,
                    "prompt": item["prompt_text"],
                    "generated_text": generated_text,
                    "emotion_considered": item["emotion"],
                    "label": item.get("label", -1),
                    "split": item.get("split", "unknown"),
                }
                f_out.write(json.dumps(f_out_record, ensure_ascii=False) + "\n")

            # Write activations pkl BEFORE flushing the jsonl, so any prompt_key
            # present in the jsonl is guaranteed to have its activations on disk
            # (resume relies on the jsonl; consolidation dedups by prompt_id).
            chunk_name = f"chunk_{shard_index:02d}_{chunk_idx:05d}.pkl"
            with open(os.path.join(activations_run_dir, chunk_name), "wb") as f_pkl:
                pickle.dump(chunk_acts, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_out.flush()

            done = min(c0 + chunk_size, len(work))
            msg = f"[REPO INFO] {done}/{len(work)} prompts done"
            if n_missing:
                msg += f"  ([WARN] {n_missing} with no captured activations)"
            print(msg)

    print("\n[REPO INFO] Generation complete.\n")
    return f"{safe_model_name}_{timestamp}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batched generation + activation extraction.")
    ap.add_argument("--model", default="/home/models/Qwen2.5-14B-Instruct")
    ap.add_argument("--dataset", default="generated_human_prompts",
                    help="generated_human_prompts | generated_prompts | emotion_query | andyzou_situations")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (big models)")
    ap.add_argument("--pp", type=int, default=1, help="pipeline_parallel_size (big models)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total shards for multi-process data parallelism (set by scripts/run_dp.py)")
    ap.add_argument("--shard-index", type=int, default=0, help="this process's shard index in [0, num_shards)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256, help="matches the existing AI-set runs")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--poolings", default="last_token",
                    help="comma-separated subset of last_token,mean,max,min,amp (probe uses last_token)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--testing", action="store_true", help="first 10 prompts only")
    ap.add_argument("--run-id", default=None, help="reuse an explicit run id/timestamp")
    ap.add_argument("--activations-dir", default=None,
                    help="base dir for activation chunk pkls (default data/03_activations; e.g. a network drive)")
    ap.add_argument("--no-response-activation", action="store_true",
                    help="skip the 2nd pass that captures the last GENERATED-token activation")
    ap.add_argument("--limit-mm-per-prompt", default=None,
                    help='JSON, e.g. \'{"image":0,"audio":0}\' — load VL checkpoints text-only')
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    # Multimodal/unified configs (e.g. Gemma4UnifiedConfig) nest the LM config.
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None and hasattr(config, "get_text_config"):
        n_layers = config.get_text_config().num_hidden_layers
    TARGET_LAYERS = list(range(n_layers))
    print(f"\n[REPO INFO] {args.model}: {n_layers} layers; targeting all.\n")

    generate(
        model_name=args.model,
        target_layers=TARGET_LAYERS,
        dataset_name=args.dataset,
        dataset_testing=args.testing,
        resume_run=args.resume,
        poolings=tuple(p.strip() for p in args.poolings.split(",") if p.strip()),
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        activations_base=args.activations_dir,
        capture_response=not args.no_response_activation,
        limit_mm_per_prompt=(__import__("json").loads(args.limit_mm_per_prompt)
                             if args.limit_mm_per_prompt else None),
    )
