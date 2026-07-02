"""
Batch-aware residual-stream activation capture for vLLM 0.22.x (v1 engine).

Replaces the old approach of overwriting vLLM's model-definition files. Instead
we register PyTorch forward hooks on the decoder layers *at runtime*, on every
worker, via ``LLM.collective_rpc``. This:

  * survives vLLM upgrades (no per-architecture source copies),
  * composes with tensor / pipeline / data parallelism (hooks register on all
    workers), and
  * is batch-aware, so ``llm.generate`` can run with full continuous batching.

How the per-prompt last token is found inside a batched forward
----------------------------------------------------------------
``get_forward_context().attn_metadata`` (a dict -> FlashAttentionMetadata in
v1) exposes ``query_start_loc`` (cumulative query offsets) and ``seq_lens``.
For sequence slot ``i`` the query length this forward is
``qsl[i+1] - qsl[i]``; when it equals ``seq_lens[i]`` the sequence's *entire*
prompt is being prefilled in this forward, and ``qsl[i+1]-1`` indexes its last
prompt token. (Decode steps have query length 1 and are skipped.) This holds as
long as prompts are not chunked across forwards and there is no prefix-cache
hit, so callers must build the engine with ``enable_prefix_caching=False`` and a
``max_num_batched_tokens`` at least as large as the longest prompt.

Captured vectors are keyed by ``tuple(prompt_token_ids)``, read from the
runner's input-id buffer. The caller joins them back to prompts via
``RequestOutput.prompt_token_ids`` — a version-independent mapping that avoids
vLLM's internal request-id / slot bookkeeping.

Requires env ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` so collective_rpc can
cloudpickle the closures below.
"""
from typing import Dict, List, Tuple


def make_register_fn(target_layers: List[int], poolings=("last_token",)):
    """Return a self-contained ``register(worker)`` closure for collective_rpc.

    The closure registers forward hooks on the decoder layers and stashes an
    accumulator on the model object: ``{prompt_token_ids_tuple: {layer_idx:
    {pool_name: np.float32[hidden]}}}``. Poolings are computed at capture time
    from the prompt's full prefill slice, so we store a few small vectors per
    layer instead of the whole sequence — and never write millions of files.

    `poolings` is any subset of {last_token, mean, max, min, amp}. The probe
    uses last_token; the others mirror the pooling-comparison appendix. Kept
    self-contained (no module-level refs) so cloudpickle serializes it by value
    into each worker process.
    """
    target_set = set(int(x) for x in target_layers)
    pools = tuple(poolings)

    def register(worker):
        import os
        import torch  # noqa: F401  (kept for clarity; tensors are torch)
        from vllm.forward_context import get_forward_context

        runner = worker.model_runner
        model = runner.model

        # Locate the decoder-layer ModuleList generically (Llama/Qwen/...).
        layers = None
        for path in ("model.layers", "model.model.layers", "layers",
                     "model.language_model.layers", "model.language_model.model.layers",
                     "language_model.layers", "language_model.model.layers"):
            obj = model
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                layers = obj
                break
        if layers is None:
            raise RuntimeError("Could not locate decoder layers on the model.")
        if os.environ.get("ACT_HOOK_DEBUG"):
            import sys as _sys
            print(f"[HOOKDBG] model={type(model).__name__} matched_layers_path={path} "
                  f"n={len(layers)} layer0={type(layers[0]).__name__}", flush=True, file=_sys.stderr)
            _ml = [(n, len(m)) for n, m in model.named_modules()
                   if type(m).__name__ == 'ModuleList' and len(m) > 8]
            print(f"[HOOKDBG] all big ModuleLists: {_ml}", flush=True, file=_sys.stderr)

        # Locate the token embedding and capture the current forward's flattened
        # input_ids from ITS input. This is model-agnostic and avoids depending on
        # runner internals (e.g. runner.input_buffers exists for some models but
        # not others). embed_tokens runs first, so cur["ids"] is populated before
        # the decoder-layer hooks below fire in the same forward.
        embed = None
        for path in ("model.embed_tokens", "embed_tokens", "model.model.embed_tokens",
                     "model.language_model.embed_tokens", "language_model.embed_tokens",
                     "language_model.model.embed_tokens"):
            obj = model
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                embed = obj
                break
        if embed is None:
            raise RuntimeError("Could not locate token embedding on the model.")

        cur = {"ids": None}

        def embed_pre_hook(module, args, kwargs):
            ids = args[0] if args else kwargs.get("input_ids")
            cur["ids"] = ids
            model._act_dbg["embed"] += 1
        embed.register_forward_pre_hook(embed_pre_hook, with_kwargs=True)

        store = {}
        model._act_store = store
        model._act_dbg = {"embed": 0, "fires": 0, "noids": 0, "stored": 0, "continued": 0}

        def pool_slice(t):
            # t: [seq, hidden] float32; return {pool_name: np.float32[hidden]}
            out = {}
            if "last_token" in pools:
                out["last_token"] = t[-1]
            if "mean" in pools:
                out["mean"] = t.mean(dim=0)
            if "max" in pools:
                out["max"] = t.max(dim=0).values
            if "min" in pools:
                out["min"] = t.min(dim=0).values
            if "amp" in pools:
                w = torch.softmax(t.mean(dim=1), dim=0)
                out["amp"] = (w[:, None] * t).sum(dim=0)
            return {k: v.cpu().numpy() for k, v in out.items()}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                try:
                    # vLLM decoder layers return (hidden_states, residual) with a
                    # fused add+norm: the TRUE residual stream is their sum, while
                    # output[0] alone is just the block's sub-output (the original
                    # repo probed this partial component). Store both.
                    if isinstance(output, tuple) and len(output) >= 2:
                        resid = output[0] + output[1]   # residual stream (resid_post)
                        comp = output[0]                # partial hidden_states component
                    else:
                        resid = output
                        comp = output
                    am = get_forward_context().attn_metadata
                    if isinstance(am, dict):
                        am = next(iter(am.values()))
                    qsl = am.query_start_loc.tolist()
                    sl = am.seq_lens.tolist()
                    # Flattened input_ids captured from the embedding layer this
                    # forward (model-agnostic; see embed_pre_hook above).
                    model._act_dbg["fires"] += 1
                    input_ids_buf = cur["ids"]
                    if input_ids_buf is None:
                        model._act_dbg["noids"] += 1
                        return
                    for i in range(len(sl)):
                        start, end = qsl[i], qsl[i + 1]
                        if (end - start) != sl[i]:
                            model._act_dbg["continued"] += 1
                            continue  # not a full-prompt prefill (decode/chunk)
                        key = tuple(input_ids_buf[start:end].tolist())
                        # Pool in float32 (model computes bf16). Store residual
                        # stream under <pool> and the partial component under
                        # <pool>_component.
                        d = pool_slice(resid[start:end].detach().float())
                        for pname, vec in pool_slice(comp[start:end].detach().float()).items():
                            d[pname + "_component"] = vec
                        store.setdefault(key, {})[layer_idx] = d
                except Exception as e:  # never let a hook crash generation
                    store.setdefault(("__error__",), {})[layer_idx] = repr(e)
            return hook

        # Compile-safe capture via vLLM's aux_hidden_state mechanism (EagleModelMixin).
        # For models whose decoder runs inside a @support_torch_compile'd inner text
        # model (e.g. VL wrappers Qwen3_5 / Gemma4), per-layer forward hooks never fire.
        # Instead we ask the inner model to emit residuals at every layer via
        # aux_hidden_state_layers, and read them off the TOP (eager) model's output,
        # returning just hidden_states so the runner is unaffected. Only the residual
        # stream is available this way (no *_component slot).
        if os.environ.get("ACT_HOOK_AUX"):
            tm = None
            for tp in ("model", "language_model.model", "language_model", "model.model"):
                o = model
                for a in tp.split("."):
                    o = getattr(o, a, None)
                    if o is None:
                        break
                if o is not None and hasattr(o, "aux_hidden_state_layers") and hasattr(o, "layers"):
                    tm = o
                    break
            if tm is None:
                raise RuntimeError("ACT_HOOK_AUX: no aux-capable inner text model found")
            n_layers = len(tm.layers)
            tm.aux_hidden_state_layers = tuple(range(1, n_layers + 1))  # residual after each layer

            def aux_hook(module, inputs, output):
                if os.environ.get("ACT_HOOK_DEBUG"):
                    import sys as _s
                    print(f"[HOOKDBG] aux_hook FIRED on {type(module).__name__}: "
                          f"output={type(output).__name__} "
                          f"len={len(output) if isinstance(output, tuple) else '-'} "
                          f"elt1={type(output[1]).__name__ if isinstance(output, tuple) and len(output) > 1 else '-'}",
                          flush=True, file=_s.stderr)
                if not (isinstance(output, tuple) and len(output) == 2
                        and isinstance(output[1], (list, tuple))):
                    return output  # aux not present -> leave untouched
                hidden, aux = output
                try:
                    am = get_forward_context().attn_metadata
                    if isinstance(am, dict):
                        # Hybrid models (Qwen3Next/Qwen3_5) mix full-attention and
                        # gated-deltanet layers; only the full-attention metadata
                        # carries query_start_loc. They all describe the same batch.
                        am = next((v for v in am.values()
                                   if getattr(v, "query_start_loc", None) is not None), None)
                    if am is None or cur["ids"] is None:
                        return hidden
                    qsl = am.query_start_loc.tolist()
                    sl = am.seq_lens.tolist()
                    ids = cur["ids"]
                    for i in range(len(sl)):
                        start, end = qsl[i], qsl[i + 1]
                        if (end - start) != sl[i]:
                            continue
                        key = tuple(ids[start:end].tolist())
                        for lj, a in enumerate(aux):
                            store.setdefault(key, {})[lj] = pool_slice(a[start:end].detach().float())
                    model._act_dbg["stored"] += 1
                except Exception as e:
                    if os.environ.get("ACT_HOOK_DEBUG"):
                        import sys as _s, traceback
                        traceback.print_exc(file=_s.stderr)
                    store.setdefault(("__error__",), {})[0] = repr(e)
                return hidden  # strip aux so the runner receives a plain tensor

            model._act_handles = [model.register_forward_hook(aux_hook)]
            return n_layers

        handles = []
        if os.environ.get("ACT_HOOK_MONKEYPATCH"):
            # Some models (e.g. VL wrappers) run decoder layers via a path that
            # bypasses register_forward_hook. Monkeypatch .forward instead: it
            # still fires as long as the layer executes in eager python.
            def _wrap(idx, layer):
                h = make_hook(idx)
                orig = layer.forward
                def forward(*a, **k):
                    out = orig(*a, **k)
                    h(layer, None, out)
                    return out
                return forward
            for idx, layer in enumerate(layers):
                if idx in target_set:
                    layer.forward = _wrap(idx, layer)
            model._act_handles = []
            return len(target_set)
        for idx, layer in enumerate(layers):
            if idx in target_set:
                handles.append(layer.register_forward_hook(make_hook(idx)))
        model._act_handles = handles
        return len(handles)

    return register


def make_fetch_fn():
    """Return a self-contained ``fetch(worker)`` closure for collective_rpc.

    Returns this worker's captures and resets the store. Returned dict maps
    prompt_token_ids tuple -> {layer_idx: tensor}. Under tensor parallelism every
    rank holds the (replicated) residual, so ranks are duplicates; under pipeline
    parallelism they are complementary. The caller merges across workers.
    Defined via a factory (not a bare module function) so cloudpickle serializes
    it by value, with no worker-side import of this module required.
    """
    def fetch(worker):
        import os
        model = worker.model_runner.model
        store = getattr(model, "_act_store", {})
        if os.environ.get("ACT_HOOK_DEBUG"):
            print(f"[HOOKDBG] fetch: store={len(store)} keys dbg={getattr(model, '_act_dbg', None)}", flush=True)
        out = dict(store)
        store.clear()
        return out
    return fetch


def merge_worker_caps(caps_per_worker: List[dict]) -> dict:
    """Merge collective_rpc results across workers (union of layer dicts)."""
    merged: Dict[tuple, Dict[int, object]] = {}
    for caps in caps_per_worker:
        if not caps:
            continue
        for key, layer_map in caps.items():
            merged.setdefault(key, {}).update(layer_map)
    return merged
