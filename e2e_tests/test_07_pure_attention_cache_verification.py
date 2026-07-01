"""
TEST 7: Pure-Attention Cache Hit Verification (Qwen2.5-7B-Instruct)

Same flow as test_06 but on a pure transformer (no hybrid/Mamba/GDN layers).
Purpose: isolate the core save/load data path from hybrid complexity.

Expected block_size: 16 (FlashAttention default, no hybrid alignment).

Verifies:
1. Model loads with TP=2, connector registers
2. First request (cold cache): save triggered
3. Second request (warm cache): cache hit, load triggered
4. Output determinism: identical tokens with temperature=0 + seed
"""

import logging
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from e2e_tests.utils import (
    get_kvcm_metrics,
    print_header,
    print_section,
)

MODEL_PATH = "/root/ws/resources/models/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 4096
MAX_TOKENS = 256
BLOCK_SIZE = 16  # FlashAttention default for pure attention models

# ~2000 tokens, enough for many full blocks at block_size=16
LONG_PROMPT = (
    "Write a comprehensive technical deep-dive about distributed inference systems for large language models. "
    "Your explanation should cover the following topics in significant detail:\n\n"
    "1. Tensor Parallelism: Explain how model weights are sharded across multiple GPUs during inference. "
    "Describe the column-parallel and row-parallel linear layer patterns. Discuss how all-reduce and "
    "all-gather operations synchronize activations across tensor-parallel workers. Compare Megatron-style "
    "tensor parallelism with sequence parallelism. Include analysis of communication volume as a function "
    "of hidden size, number of layers, and number of GPUs.\n\n"
    "2. Pipeline Parallelism: Describe how different layers are assigned to different pipeline stages. "
    "Explain the bubble problem in naive pipeline parallelism and how GPipe, PipeDream, and 1F1B scheduling "
    "address it. Discuss micro-batch sizing and its impact on pipeline efficiency.\n\n"
    "3. KV Cache Management: Describe how autoregressive generation maintains a growing key-value cache. "
    "Explain paged attention and how virtual memory concepts are applied to GPU memory management for "
    "inference. Discuss prefix caching, where common prompt prefixes are cached and reused across requests. "
    "Describe how external KV cache systems enable sharing cached state across different inference instances.\n\n"
    "4. Speculative Decoding: Explain how a smaller draft model generates candidate tokens that are verified "
    "by the target model in parallel. Discuss acceptance rate, speedup bounds, and verification overhead.\n\n"
    "5. Quantization: Describe INT8, INT4, and FP8 quantization schemes for weight and activation quantization. "
    "Explain the difference between post-training quantization and quantization-aware training.\n\n"
    "6. Batching Strategies: Compare static batching with continuous batching (also called iteration-level "
    "scheduling). Explain how continuous batching improves GPU utilization by allowing requests to start and "
    "finish independently. Discuss chunked prefill and its role in balancing prefill and decode throughput.\n\n"
    "7. Memory Management: Discuss GPU memory hierarchy (HBM, L2 cache, shared memory) and how it "
    "affects inference performance. Explain memory pooling strategies used by PyTorch and CUDA allocators.\n\n"
    "8. Scheduling and Request Routing: Explain how inference servers route requests to replicas. Discuss "
    "load balancing strategies including round-robin, least-connections, and cache-aware routing.\n\n"
)


def _get_connector_config():
    """Get connector config for pure-attention model."""
    return KVTransferConfig(
        kv_connector="TairKvCacheConnector",
        kv_role="kv_both",
        kv_connector_module_path="kv_cache_manager.py_connector.vllm.v1_connector",
        kv_connector_extra_config={
            "manager_uri": "http://localhost:6382",
            "coordinator_base_port": 5556,  # different port to avoid conflict
            "instance_group": "test-group",
            "instance_id": f"test-pure-{int(time.time())}",
            "preferred_block_size": BLOCK_SIZE,
            "write_timeout_seconds": 30,
            "sdk_thread_num": 4,
            "sdk_queue_size": 100,
            "sdk_get_timeout_ms": 5000,
            "sdk_put_timeout_ms": 10000,
            "read_iov_block_size": 0,
            "write_iov_block_size": 0,
            "hf3fs_concurrent_io_block_count": 1,
            "block_per_save_task": 128,
            "block_per_load_task": 128,
            "async_get_cache_location": True,
            "auto_discover_leader": False,
            "leader_retry_count": 1,
            "leader_retry_base_interval_seconds": 0.005,
            "discovery_refresh_interval_seconds": 30,
            "min_discover_interval_seconds": 1,
            "log_level": "DEBUG",  # verbose for debugging
        },
    )


def test_pure_attention_cache_verification() -> bool:
    print_header("TEST 7: Pure-Attention Cache Hit Verification")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  Max model len: {MAX_MODEL_LEN}")

    # ── 7.1 Load model ──
    print_section("[7.1] Loading model with TP=2 (pure attention)...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            enforce_eager=True,
            enable_prefix_caching=False,
            kv_transfer_config=_get_connector_config(),
        )
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback; traceback.print_exc()
        return False

    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=MAX_TOKENS)
    print(f"  Prompt chars: {len(LONG_PROMPT)}")
    print(f"  max_tokens: {MAX_TOKENS}")

    # ── 7.2 Baseline metrics ──
    print_section("[7.2] Baseline Prometheus metrics...")
    hit_before = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_before = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter:   {hit_before}")
    print(f"  query_block_counter: {query_before}")

    # ── 7.3 First request (cold cache → save) ──
    print_section("[7.3] First request — cold cache, expect save...")
    t0 = time.time()
    try:
        out1 = llm.generate([LONG_PROMPT], sampling_params)
        t1 = time.time() - t0
        text1 = out1[0].outputs[0].text
        tokens1 = out1[0].outputs[0].token_ids
        prompt_tokens1 = len(out1[0].prompt_token_ids)
        print(f"✓ Generation 1 complete ({t1:.2f}s)")
        print(f"  Prompt tokens: {prompt_tokens1}")
        print(f"  Output tokens: {len(tokens1)}")
        print(f"  Total tokens:  {prompt_tokens1 + len(tokens1)}")
        print(f"  Text (first 80 chars): {text1[:80]!r}")
    except Exception as e:
        print(f"✗ Generation 1 failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── 7.4 Wait for save ──
    print_section("[7.4] Waiting for async save...")
    time.sleep(5)
    hit_mid = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_mid = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter:   {hit_mid}")
    print(f"  query_block_counter: {query_mid}")

    # ── 7.5 Second request (warm cache → load) ──
    print_section("[7.5] Second request — warm cache, expect load...")
    t0 = time.time()
    try:
        out2 = llm.generate([LONG_PROMPT], sampling_params)
        t2 = time.time() - t0
        text2 = out2[0].outputs[0].text
        tokens2 = out2[0].outputs[0].token_ids
        print(f"✓ Generation 2 complete ({t2:.2f}s)")
        print(f"  Output tokens: {len(tokens2)}")
        print(f"  Text (first 80 chars): {text2[:80]!r}")
    except Exception as e:
        print(f"✗ Generation 2 failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── 7.6 Final metrics ──
    print_section("[7.6] Final Prometheus metrics...")
    hit_after = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_after = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter:   {hit_after}")
    print(f"  query_block_counter: {query_after}")

    # ── 7.7 Assertions ──
    print_section("[7.7] Verification...")
    all_passed = True

    if tokens1 == tokens2:
        print("✓ PASS: Token outputs identical")
    else:
        print("✗ FAIL: Token outputs differ!")
        print(f"  1st: {list(tokens1)[:10]}...")
        print(f"  2nd: {list(tokens2)[:10]}...")
        # Show first divergence point
        for i, (a, b) in enumerate(zip(tokens1, tokens2)):
            if a != b:
                print(f"  First divergence at token {i}: {a} vs {b}")
                break
        all_passed = False

    h_b = hit_before if hit_before is not None else 0.0
    h_a = hit_after if hit_after is not None else 0.0
    if h_a > h_b:
        print(f"✓ PASS: Cache hit confirmed — hit_block_counter +{h_a - h_b} ({h_b} → {h_a})")
    else:
        print(f"✗ FAIL: No cache hit — hit_block_counter unchanged ({h_b} → {h_a})")
        all_passed = False

    if prompt_tokens1 >= BLOCK_SIZE:
        print(f"✓ PASS: Prompt tokens ({prompt_tokens1}) >= block_size ({BLOCK_SIZE})")
    else:
        print(f"✗ FAIL: Prompt too short ({prompt_tokens1} < {BLOCK_SIZE})")
        all_passed = False

    print(f"\n  Latency: 1st={t1:.2f}s  2nd={t2:.2f}s  speedup={t1/t2:.2f}x")

    if all_passed:
        print("\n✓ Pure-attention cache verification PASSED")
    else:
        print("\n✗ Pure-attention cache verification FAILED")

    return all_passed


if __name__ == "__main__":
    import sys
    success = test_pure_attention_cache_verification()
    sys.exit(0 if success else 1)
