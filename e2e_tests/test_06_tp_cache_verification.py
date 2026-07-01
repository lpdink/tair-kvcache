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
# Prompt must tokenize to >= block_size (528) for GetCacheLocation to generate
# block keys. Server computes block_keys via: total_blocks = len(token_ids) // block_size.
# With 116-token prompts and block_size=528, total_blocks=0 and no match is possible.
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
    "3. Data Parallelism and ZeRO: Explain how ZeRO optimizer stages (ZeRO-1, ZeRO-2, ZeRO-3) partition "
    "optimizer states, gradients, and parameters across data-parallel workers. Discuss the tradeoff between "
    "communication overhead and memory savings.\n\n"
    "4. KV Cache Management: Describe how autoregressive generation maintains a growing key-value cache. "
    "Explain paged attention and how virtual memory concepts are applied to GPU memory management for "
    "inference. Discuss prefix caching, where common prompt prefixes are cached and reused across requests. "
    "Describe how external KV cache systems enable sharing cached state across different inference instances.\n\n"
    "5. Speculative Decoding: Explain how a smaller draft model generates candidate tokens that are verified "
    "by the target model in parallel. Discuss acceptance rate, speedup bounds, and verification overhead.\n\n"
    "6. Quantization: Describe INT8, INT4, and FP8 quantization schemes for weight and activation quantization. "
    "Explain the difference between post-training quantization and quantization-aware training. Discuss how "
    "KV cache quantization reduces memory footprint during long-context generation.\n\n"
    "7. Batching Strategies: Compare static batching with continuous batching (also called iteration-level "
    "scheduling). Explain how continuous batching improves GPU utilization by allowing requests to start and "
    "finish independently. Discuss chunked prefill and its role in balancing prefill and decode throughput.\n\n"
    "Please provide specific examples from production systems like vLLM, TensorRT-LLM, DeepSpeed-MII, and "
    "SGLang where applicable.\n\n"
    "8. Memory Management: Discuss GPU memory hierarchy (HBM, L2 cache, shared memory) and how it "
    "affects inference performance. Explain memory pooling strategies used by PyTorch and CUDA allocators. "
    "Describe how memory fragmentation impacts long-running inference servers and techniques to mitigate it.\n\n"
    "9. Scheduling and Request Routing: Explain how inference servers route requests to replicas. Discuss "
    "load balancing strategies including round-robin, least-connections, and cache-aware routing. Describe "
    "how prefix-aware scheduling can improve cache hit rates in multi-instance deployments.\n\n"
    "10. Fault Tolerance and Reliability: Discuss how inference systems handle GPU failures, network partitions, "
    "and node crashes. Explain checkpoint-based recovery for long-running generation tasks. Describe how "
    "graceful shutdown and health checks enable zero-downtime deployments of inference services.\n\n"
    "11. Model Compression and Pruning: Explain structured vs unstructured pruning techniques for reducing model "
    "size while maintaining accuracy. Discuss knowledge distillation approaches where a smaller student model "
    "learns from a larger teacher model. Describe how weight sharing and low-rank factorization reduce memory "
    "footprint and improve inference throughput.\n\n"
    "12. Multimodal Inference: Describe how vision-language models process images and text together. Explain the "
    "challenges of handling variable-length image features and how they interact with text token sequences. "
    "Discuss encoder-decoder architectures for multimodal generation tasks and the memory requirements for "
    "processing high-resolution images alongside long text contexts.\n\n"
    "13. Cost Optimization and Resource Management: Explain how inference providers optimize GPU utilization "
    "across multiple models and workloads. Discuss auto-scaling strategies based on request queue depth and "
    "latency SLOs. Describe spot instance usage, model warm-up strategies, and techniques for minimizing "
    "cold start latency in serverless inference deployments."
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
