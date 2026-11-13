# Dory baseline

This page documents the native Dory implementation added for
[NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
and the exact scope that is qualified today. It describes a **baseline
increment**, not full support: the plan for continuous serving, cache
management, execution and speculation is tracked in
[RFC #176](https://github.com/ThinkFlowLab/vllm-rlt/issues/176).

## Model layout

Dory is one physical stack of 72 sublayers. Each sublayer holds a single mixer,
either attention or MLP, plus an NGPT slerp residual:

| Stage | Sublayers | KV |
| --- | --- | --- |
| Input (`n_input_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |
| Recurrent (`n_recurrent_layers`) | 40 (20 attention, 20 MLP) | per loop depth, or one shared plane |
| Output (`n_output_layers`) | 16 (8 attention, 8 MLP) | boundary KV at depth 0 |

`layer_types` maps every physical sublayer to `sliding_attention`,
`full_attention` or `mlp`, and the released artifact lists 27 sliding and 9
full attention sublayers: every fourth attention pair is full attention. When
`layer_types` is omitted the constructor generates exactly that pattern, and
refuses anything that is not attention/MLP pairs so a custom stack cannot fall
back to an all-sliding architecture by accident. `sliding_window` (1024 in the
artifact) applies only to `sliding_attention` sublayers.

The recurrent stage runs `n_recurrent_loops` times. Before every loop the
retained input-stage output is combined with the state and re-normalized, so
the runner carries `[recurrent state, injection input]` and
`DoryConfig.hidden_size` is the doubled carry width while
`DoryConfig.d_model` is the artifact width. `config.json`'s `hidden_size` maps
to `d_model`.

NGPT arithmetic is preserved explicitly: FP32 `justnorm` accumulation, learned
per-channel scales stored without their constant factors, post-RoPE Q/K
normalization, the output gate before the projection, the MLP gate/up scaling,
and slerp residuals with `ngpt_alpha_init`. Logits are scaled by the learned
vocabulary-wise `lm_head.logit_scale`.

## KV modes

`recurrent_kv_cache_mode` selects between two different semantics, and each maps
to one cache layout:

| Mode | Runtime layout | Recurrent attention history |
| --- | --- | --- |
| `per_loop` (default) | `last_exited` | Independent KV per loop depth |
| `last_loop` | `shared` | One plane per layer, overwritten by each loop |

Boundary stages always write depth 0, so input and output KV are never
replicated per loop. In `last_loop` mode a token's current-step KV is overwritten
by the next loop, so results depend on the chunk schedule; the qualified
schedule is one token per step. `per_loop` gives the same result for
token-at-a-time and all-positions-at-once execution.

Only the 36 attention sublayers own KV. The cache pool is sized by that compact
count (`DoryConfig.num_kv_layers`), not by all 72 physical sublayers, and
`StageEngine`/`plan_cache` select it through `num_kv_layers` when a model
declares one. Attention layers therefore reach the cache through a packed
layer id rather than their physical sublayer index.

## Grouped storage and optional SWA reclamation

Dory uses one pool of attention-layer pages, grouped by window policy and
boundary/recurrent plane domain. `reclaim_swa` defaults to `False`. To return
expired sliding-window pages to the pool, enable it together with incremental
allocation on an already loaded model:

```python
from vllm_rlt import CacheConfig
from vllm_rlt.config import ExecutionConfig
from vllm_rlt.engine.llm_engine import LLMEngine

engine = LLMEngine(
    model,
    attention_backend="torch",
    execution_config=ExecutionConfig(use_uva=False),
    cache_config=CacheConfig(
        num_blocks=256,
        layout=model.kv_cache_layout,
        incremental_allocation=True,
        reclaim_swa=True,
    ),
)
```

`num_blocks` preserves the existing all-layer KV byte budget: the native pool
converts it to `num_blocks * model.config.num_kv_layers` single-layer pages.
Reclamation supports ordinary synchronous fixed-loop `ouro` execution. Each
`per_loop` plane advances independently; shared recurrent history in
`last_loop` advances only after the token's final loop. Pages retire only after
all layers in their group have consumed the range and the submission completes.
Full-attention history remains resident.

Native SWA attention stages only the needed page-aligned read span. Logical token
positions, write addresses and public layer maps remain absolute; the backend's
read-table context lengths are local to that span. Native attention calls must
match the layer's declared window, including `None` for full-attention layers.

For read-only group policy, `cache.layer_group(kv_layer_id)` returns
`(group_id, policy)`: a manager-local group ID and the immutable `KVGroup` with
compact layer IDs, window, recurrent domain and plane count. The layer ID is a
compact KV index, not a physical sublayer index.

For read-only cache inspection, `cache.retained_range(request_id, kv_layer_id,
depth)` returns `(resident_start, written_end)` in absolute token positions.
`cache.layer_block_table(request_id, kv_layer_id, depth)` returns the absolute
logical-page map, with `-1` for reclaimed pages. Here `cache` is
`engine.cache_manager`, and `kv_layer_id` is a compact KV index. Reading a
reclaimed full prefix is rejected; these queries do not retain page references
or expose mutable allocator state.

Grouped prefix caching publishes one entry per token page holding every
owned (group, plane, layer) page, ordered as the allocator splits a claim.
The entry takes its own reference, so the publishing request can finish or
release its pages without freeing the cached ones. Publication stops at the
first token page with any owned page missing. Prefix caching together with
`reclaim_swa` is rejected until hit-path reclamation is qualified. The
inspection methods are not a pin API.

## Supported today

- Native strict safetensors loading with the artifact's per-projection key
  names (`q/k/v/g/o_proj`, `gate_proj`/`up_proj`) and the learned scale tensors.
  There is no fused-key remapping: a checkpoint that packs `qkv_proj` or
  `gate_up_proj` fails loudly instead of loading approximately.
- Synchronous eager execution: ordinary bounded-chunk prefill followed by token
  decoding, with the configured fixed loop count.
- `loops=` at load time to fix 1, 2 or 4 loops.
- Greedy and sampled decoding, request seeds, EOS (`2`, `11`) and length
  stopping through the existing engine.
- Prefix caching (`enable_prefix_caching=True`) in `per_loop` mode. A
  published page carries boundary KV from plane 0 and recurrent KV from every
  plane; boundary layers never write deeper planes, so they are not required
  there. A hit restores exactly that coverage and the scheduler resumes prefill
  from the cached offset. `last_loop` keeps one shared plane and stays rejected.
  Prefix keys hash tokens only. That is sound because the loop count is fixed
  when the model loads and the layout is fixed by `CacheConfig`, and each
  engine owns its own KV manager, so pages from different loop counts or
  layouts never share a cache. A cache shared across loop counts would have to
  add both to the key. Evicted prefixes fall back to a full cold prefill.

## Rejected today

These fail fast rather than degrade silently:

- WCPB, mixed or staged prefill; asynchronous execution; PD.
- CUDA Graphs (the sliding-window path needs host-planned KV metadata).
- Prefix caching in `last_loop` mode.
- Preemption and native-cache snapshot/restore.
- Self-speculation.
- Per-request loop counts other than the configured full depth: Dory has no
  trained exit gate, so the runtime refuses to invent one, including via
  `min_loops` with `exit_threshold=0`.

Sliding-window masking is implemented in the **Torch reference attention
backend** only (`window=` on `KVCacheManager.attend`), and backend selection now
requires the `sliding_window` capability, so a Dory engine on Triton/Flash fails
at construction instead of mid-request. The reference kernel is a Python loop
over rows: correct, but slow enough that it distorts memory-profiling wall time.
Windowed Triton/Flash kernels remain follow-up work. Optional physical SWA page
reclamation is available on the grouped Torch path described above.

## Validation status

`tests/test_dory.py` covers configuration validation, strict loading with
single-file and sharded checkpoints, stage/KV contracts, sliding-window masking,
engine scheduling, and numerical agreement with an independent dense oracle
(`tests/reference/dory.py`) for both KV modes — including a greedy generation
comparison against that oracle.
Prefix reuse is checked at loops 1, 2 and 4: cold and cache-hit prefill
logits match the oracle, cached prefixes longer than the sliding window,
block-aligned prompts, diverging suffixes, concurrent sharers and a
re-request after its prefix was evicted all reproduce the cold greedy tokens,
and publication refuses pages missing any owned plane.

Prefix pages stay pinned for the life of the cache entry. Prefix tests run on
both the legacy all-layer blocks and grouped storage.

The current grouped integration is covered by tiny FP32 CPU numerical and
multi-request lifecycle tests for both KV modes, including independent dense
logits, page reuse, cancellation and cleanup. GPU/backend and real-checkpoint
qualification remain separate follow-up work.

Not yet performed, and required before claiming support:

- Real-checkpoint BF16 comparison with the pinned upstream implementation
  (logits, generated outputs, stopping) at 1, 2 and 4 loops.
- Automatic CUDA memory profiling: `plan_cache`'s CUDA profiling branch remains
  unexercised for Dory; current grouped checks use an explicit KV capacity.
- Long-context behaviour beyond the tested lengths; the artifact's 256K context
  is a configuration limit, not a qualified deployment length.
- Performance and GPU allocator-peak memory measurement. Request-held KV page
  accounting does not measure either.
