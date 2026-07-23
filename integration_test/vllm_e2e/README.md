# vLLM E2E KV Cache Verification

End-to-end integration tests that verify the KVCM vLLM connector's **translation
layer** (the mapping between KVCM manager blocks, global token positions and
vLLM physical blocks) actually moves the correct KV data — not merely that the
transport layer round-trips bytes.

## Why this test exists

The connector translates between three block spaces:

```
KVCM manager block idx  ->  global token idx  ->  vLLM physical block id
        (step 1, connector-only)      (step 2/3, shared with vLLM)
```

A bug in step 1 makes **save** gather KV from the wrong physical slots and
**load** scatter KV to the wrong physical slots. Because save and load share the
same translation function, a transport-level round trip still "matches" — the
bug is *symmetric* and invisible to storage-layer tests.

To break the symmetry, these tests capture the KV data **independently of the
connector's translation**: they read vLLM's paged KV cache using vLLM's own
block-table mapping (`slot = block_table[pos // bs] * bs + pos % bs`, where the
connector's `local_block_ids` *is* the block table). This reference is
independent of step 1, so a step-1 bug makes the captured data diverge from what
the connector saved/loaded.

> Note: GPU compute is non-deterministic, so we cannot assert "decode tokens with
> cache-hit == decode tokens without". We therefore verify at the KV-cache object
> level instead.

## How it works

A `VerifyingConnector` subclasses the production `TairKvCacheConnector` and is
injected via vLLM's `kv_connector_module_path` — **no vLLM or connector source
files are modified**. It captures KV data one record **per manager block**, keyed
by the captured token ids, so the driver can match reference vs loaded records by
content:

- **Reference capture (save path)** — in `wait_for_save()` (the forward pass has
  completed), for each save request it reads the KV of each saved manager block
  straight from the paged cache using the independent block-table mapping.
- **Loaded capture (load path)** — loads are async (`load_kv_async=True`): the
  load step has no forward pass, and the worker does not yet know the request's
  token ids. So the load's block info is recorded in `start_load_kv()` (which is
  always called), and the capture is emitted in a later `wait_for_save()` once the
  request's token ids have arrived on the worker. Before reading, it waits for the
  async scatter to finish (tracked by wrapping the load done-callback) and calls
  `torch.cuda.synchronize()`.

Captures are written to `$KVCM_E2E_CAPTURE_DIR` as `{ref|loaded}_tp{rank}_{token_hash}.pt`.

The driver:

1. Sends the base prompts (phase 1) and waits for reference captures.
2. Waits until the manager has **committed** the save — it tokenizes each prompt
   and polls the manager's `getCacheLocation` (the same query the connector makes)
   until a prefix match exists. This closes the race between the async save and
   the phase-2 query.
3. Sends prefix+suffix prompts (phase 2) and waits for loaded captures.
4. Compares: **every loaded block must match a reference block** (same tp rank +
   token content). The direction matters — saves are incremental, so some saved
   blocks may legitimately not be reloaded (e.g. the tokenization boundary
   block), but every loaded block must correspond to something that was saved.
   Comparison is **bit-exact** (`torch.equal`) preferred, with **cosine
   similarity > 99.99%** as a fallback (recorded for analysis).

Each TP rank captures independently, so TP runs verify every rank.

## Layout

```
integration_test/vllm_e2e/
├── BUILD               # py_test targets (tagged manual + gpu)
├── test_connector.py   # VerifyingConnector (injected via kv_connector_module_path)
├── e2e_lib.py          # orchestration: manager + vLLM + driver + comparison
├── test_basic.py       # single request, TP=1
├── test_concurrent.py  # 4 concurrent requests, TP=1
└── test_tp.py          # TP=2 with preferred_block_size=32 (cross-block mapping)
```

## Requirements

- 2 GPUs (A10 or equivalent) for the TP test; 1 GPU suffices for basic/concurrent.
- A Python env with `vllm==0.22.1`, `torch`, and the built KVCM wheels installed
  (default: `/root/ws/env/global_vllm/.venv/bin/python`, override with
  `KVCM_E2E_PYTHON`).
- The Qwen2.5-7B-Instruct model (default:
  `/root/ws/resources/models/Qwen2.5-7B-Instruct`, override with `KVCM_E2E_MODEL`).
- The KVCM manager binary built: `bazelisk build //kv_cache_manager:kv_cache_manager_bin`.

## Run

```bash
# Build the manager binary (and optionally the wheels)
bazelisk build //kv_cache_manager:kv_cache_manager_bin --stamp

# Run all three scenarios
bazelisk test //integration_test/vllm_e2e/... --cache_test_results=no --test_output=errors

# Or a single scenario
bazelisk test //integration_test/vllm_e2e:test_tp --cache_test_results=no --test_output=errors
```

The tests are tagged `manual` + `gpu`, so they are **not** picked up by
`//integration_test/...` in the normal CPU CI; they run in the dedicated
`test-vllm-e2e` workflow (on PRs and on demand) or when invoked explicitly.

## Configuration knobs (env)

| Variable | Default | Purpose |
|---|---|---|
| `KVCM_E2E_PYTHON` | `/root/ws/env/global_vllm/.venv/bin/python` | Python interpreter that hosts vLLM |
| `KVCM_E2E_MODEL` | `/root/ws/resources/models/Qwen2.5-7B-Instruct` | Model path |

The manager always uses the local-file storage backend (no special hardware).

## What each test covers

| Test | TP | Prompts | preferred_block_size | Focus |
|---|---|---|---|---|
| `test_basic` | 1 | 1 | 0 (= vLLM bs) | single save/load round trip |
| `test_concurrent` | 1 | 4 | 0 | ReqState tracking, per-request attribution, async races |
| `test_tp` | 2 | 2 | 32 (≠ vLLM bs 16) | TP coordination + cross-block translation |
