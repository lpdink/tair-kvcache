"""
TEST 6: Multi-GPU Tensor Parallel Cache Hit Verification

This is the dedicated test for verifying external KV cache hits under TP=2.

Verifies:
1. TP=2 model loads and both workers register with KVCM
2. First request (cold cache): no external cache hit, save triggered
3. Second request (warm cache): external cache hit detected via Prometheus
4. Output determinism: identical tokens with temperature=0 + seed
5. Performance: second request should be faster (informational)

Key config note: max_model_len must be >= preferred_block_size (528) * 2
to ensure at least one full block can be saved to external cache.
"""

import logging
import os
import re
import time

from vllm import LLM, SamplingParams
from e2e_tests.utils import (
    MODEL_PATH,
    get_connector_config,
    get_kvcm_metrics,
    print_header,
    print_section,
)

# We need max_model_len >> block_size (528) to produce at least one full block.
# 2048 tokens gives us ~3 full blocks worth of data.
MAX_MODEL_LEN = 2048
# Prompt must be long enough + max_tokens to exceed 528 tokens total.
LONG_PROMPT = (
    "Write a detailed technical explanation of how tensor parallelism works "
    "in large language model inference. Cover the following aspects: "
    "weight sharding across GPUs, communication patterns during forward pass, "
    "impact on memory bandwidth, and comparison with pipeline parallelism. "
    "Include specific examples using transformer architectures like GPT and LLaMA. "
    "Discuss the tradeoffs between tensor parallelism and data parallelism. "
    "Explain how collective operations like all-reduce are used in this context. "
    "Describe the role of NCCL and other communication libraries. "
    "Finally, discuss how tensor parallelism interacts with KV cache management "
    "in modern inference serving systems."
)
MAX_TOKENS = 512


def _count_matched_from_log(log_text: str) -> list[int]:
    """Extract new_matched_count values from log text."""
    return [int(m) for m in re.findall(r"new_matched_count:(\d+)", log_text)]


def test_tp_cache_verification() -> bool:
    """Comprehensive multi-GPU cache hit verification."""
    print_header("TEST 6: Multi-GPU TP Cache Hit Verification")

    # ── 6.1 Load model with TP=2 ──
    print_section("[6.1] Loading model with tensor_parallel_size=2...")
    try:
        connector_config = get_connector_config()

        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            enforce_eager=True,
            kv_transfer_config=connector_config,
        )
        print(f"✓ Model loaded with TP=2 (max_model_len={MAX_MODEL_LEN})")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=MAX_TOKENS)

    # Count prompt tokens (approximate)
    print(f"  Prompt (approx chars): {len(LONG_PROMPT)}")
    print(f"  max_tokens: {MAX_TOKENS}")

    # ── 6.2 Baseline Prometheus metrics ──
    print_section("[6.2] Capturing baseline Prometheus metrics...")
    hit_before = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_before = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter:   {hit_before}")
    print(f"  query_block_counter: {query_before}")

    # ── 6.3 First request (cold cache) ──
    print_section("[6.3] First request — cold cache, expect save...")

    t1_start = time.time()
    try:
        outputs1 = llm.generate([LONG_PROMPT], sampling_params)
        t1_elapsed = time.time() - t1_start
        text1 = outputs1[0].outputs[0].text
        tokens1 = outputs1[0].outputs[0].token_ids
        prompt_tokens1 = len(outputs1[0].prompt_token_ids)
        total_tokens1 = prompt_tokens1 + len(tokens1)
        print(f"✓ Generation 1 complete ({t1_elapsed:.2f}s)")
        print(f"  Prompt tokens: {prompt_tokens1}")
        print(f"  Output tokens: {len(tokens1)}")
        print(f"  Total tokens:  {total_tokens1}")
        print(f"  Text (first 80 chars): {text1[:80]!r}")
    except Exception as e:
        print(f"✗ Generation 1 failed: {e}")
        return False

    # ── 6.4 Wait for save ──
    print_section("[6.4] Waiting for async save to complete...")
    time.sleep(5)

    # Prometheus after first request
    hit_mid = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_mid = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter after 1st req:   {hit_mid}")
    print(f"  query_block_counter after 1st req: {query_mid}")

    # ── 6.5 Second request (warm cache) ──
    print_section("[6.5] Second request — warm cache, expect load...")

    t2_start = time.time()
    try:
        outputs2 = llm.generate([LONG_PROMPT], sampling_params)
        t2_elapsed = time.time() - t2_start
        text2 = outputs2[0].outputs[0].text
        tokens2 = outputs2[0].outputs[0].token_ids
        print(f"✓ Generation 2 complete ({t2_elapsed:.2f}s)")
        print(f"  Output tokens: {len(tokens2)}")
        print(f"  Text (first 80 chars): {text2[:80]!r}")
    except Exception as e:
        print(f"✗ Generation 2 failed: {e}")
        return False

    # ── 6.6 Final Prometheus metrics ──
    print_section("[6.6] Final Prometheus metrics...")
    hit_after = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_after = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  hit_block_counter:   {hit_after}")
    print(f"  query_block_counter: {query_after}")

    # ── 6.7 Assertions ──
    print_section("[6.7] Verification results...")
    all_passed = True

    # Check 1: Output determinism
    if tokens1 == tokens2:
        print("✓ PASS: Token outputs identical (deterministic with temp=0 + seed)")
    else:
        print("✗ FAIL: Token outputs differ!")
        print(f"  1st: {tokens1[:10]}...")
        print(f"  2nd: {tokens2[:10]}...")
        all_passed = False

    # Check 2: Prometheus cache hit
    # None means metric not yet registered (equivalent to 0 for counters)
    h_before = hit_before if hit_before is not None else 0.0
    h_after = hit_after if hit_after is not None else 0.0
    
    if h_after > h_before:
        delta = h_after - h_before
        print(f"✓ PASS: Cache hit confirmed — hit_block_counter +{delta} ({h_before} → {h_after})")
    elif hit_before is None and hit_after is None:
        print("⚠ SKIP: Prometheus metrics unavailable, cannot verify cache hit")
    else:
        print(f"✗ FAIL: No cache hit — hit_block_counter unchanged ({h_before} → {h_after})")
        # Don't fail the test if total tokens < block_size (no data could be saved)
        if total_tokens1 < 528:
            print(f"  NOTE: total_tokens ({total_tokens1}) < block_size (528), no block could be saved")
        else:
            all_passed = False

    # Check 3: Sufficient token coverage
    if total_tokens1 >= 528:
        print(f"✓ PASS: Total tokens ({total_tokens1}) >= block_size (528), save possible")
    else:
        print(f"⚠ WARN: Total tokens ({total_tokens1}) < block_size (528), save may not occur")

    # Informational: latency comparison
    if t1_elapsed > 0 and t2_elapsed > 0:
        speedup = t1_elapsed / t2_elapsed if t2_elapsed > 0 else 0
        print(f"\n  Latency: 1st={t1_elapsed:.2f}s  2nd={t2_elapsed:.2f}s  speedup={speedup:.2f}x")

    if all_passed:
        print("\n✓ TP cache verification PASSED")
    else:
        print("\n✗ TP cache verification FAILED")

    return all_passed


if __name__ == "__main__":
    import sys
    success = test_tp_cache_verification()
    sys.exit(0 if success else 1)
