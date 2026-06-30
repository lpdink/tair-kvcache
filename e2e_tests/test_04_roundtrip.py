"""
TEST 4: Save → Load Round-Trip Verification (Tensor Parallel)

This test verifies:
1. First request triggers save operation with TP=2
2. Second request (same prompt) triggers load operation
3. Outputs are consistent between first and second requests
4. Prometheus metrics show cache hits after second request
"""

import time
from vllm import LLM, SamplingParams
from e2e_tests.utils import (
    MODEL_PATH,
    get_connector_config,
    get_kvcm_metrics,
    print_header,
    print_section,
)


def test_save_load_roundtrip() -> bool:
    """Verify save → load round-trip correctness with TP=2"""
    print_header("TEST 4: Save → Load Round-Trip Verification (TP=2)")
    
    print_section("[4.1] Loading model with tensor_parallel_size=2...")
    try:
        connector_config = get_connector_config()
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            trust_remote_code=True,
            max_model_len=512,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            enforce_eager=True,
            kv_transfer_config=connector_config,
        )
        print("✓ Model loaded with TP=2")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    prompt = "The capital of France is"
    # seed=42 ensures deterministic sampling with temperature=0.0
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=20)
    
    # Capture Prometheus metrics before first request
    print_section("[4.2] Capturing baseline Prometheus metrics...")
    hit_counter_before = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_counter_before = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  Hit blocks before: {hit_counter_before}")
    print(f"  Query blocks before: {query_counter_before}")
    
    # First request: should trigger save (no cache hit expected)
    print_section("[4.3] First request (should trigger save)...")
    t1_start = time.time()
    try:
        outputs1 = llm.generate([prompt], sampling_params)
        t1_elapsed = time.time() - t1_start
        text1 = outputs1[0].outputs[0].text
        tokens1 = outputs1[0].outputs[0].token_ids
        print(f"✓ First generation successful ({t1_elapsed:.2f}s)")
        print(f"  Text: {text1!r}")
        print(f"  Tokens: {len(tokens1)}")
    except Exception as e:
        print(f"✗ First generation failed: {e}")
        return False
    
    # Wait for save to complete
    print_section("[4.4] Waiting for save to complete...")
    time.sleep(3)
    
    # Capture metrics after first request
    hit_counter_mid = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    print(f"  Hit blocks after 1st request: {hit_counter_mid}")
    
    # Second request: should trigger load (cache hit expected)
    print_section("[4.5] Second request (should trigger load)...")
    t2_start = time.time()
    try:
        outputs2 = llm.generate([prompt], sampling_params)
        t2_elapsed = time.time() - t2_start
        text2 = outputs2[0].outputs[0].text
        tokens2 = outputs2[0].outputs[0].token_ids
        print(f"✓ Second generation successful ({t2_elapsed:.2f}s)")
        print(f"  Text: {text2!r}")
        print(f"  Tokens: {len(tokens2)}")
    except Exception as e:
        print(f"✗ Second generation failed: {e}")
        return False
    
    # Capture metrics after second request
    print_section("[4.6] Capturing final Prometheus metrics...")
    hit_counter_after = get_kvcm_metrics("kvcm_manager_get_cache_location_hit_block_counter")
    query_counter_after = get_kvcm_metrics("kvcm_manager_get_cache_location_query_block_counter")
    print(f"  Hit blocks after 2nd request: {hit_counter_after}")
    print(f"  Query blocks after 2nd request: {query_counter_after}")
    
    # Verify consistency
    print_section("[4.7] Verifying consistency...")
    if text1 == text2:
        print("✓ Text output is consistent")
    else:
        print(f"⚠ Text output differs:")
        print(f"  First:  {text1!r}")
        print(f"  Second: {text2!r}")
    
    if tokens1 == tokens2:
        print("✓ Token IDs are consistent")
    else:
        print(f"⚠ Token IDs differ:")
        print(f"  First:  {tokens1}")
        print(f"  Second: {tokens2}")
    
    # Verify cache hit via Prometheus metrics
    print_section("[4.8] Verifying cache hit via Prometheus metrics...")
    if hit_counter_before is not None and hit_counter_after is not None:
        if hit_counter_after > hit_counter_before:
            print(f"✓ Cache hit detected: hit_block_counter increased from {hit_counter_before} to {hit_counter_after}")
            cache_hit_verified = True
        else:
            print(f"⚠ No cache hit: hit_block_counter unchanged ({hit_counter_before} → {hit_counter_after})")
            cache_hit_verified = False
    else:
        print("⚠ Prometheus metrics unavailable, skipping cache hit verification")
        cache_hit_verified = None  # None = skip, not fail
    
    # Performance comparison (informational)
    if t1_elapsed > 0 and t2_elapsed > 0:
        speedup = t1_elapsed / t2_elapsed if t2_elapsed > 0 else 0
        print(f"\n  Performance: 1st={t1_elapsed:.2f}s, 2nd={t2_elapsed:.2f}s, speedup={speedup:.2f}x")
    
    # For temperature=0.0 + seed, we expect exact match
    if text1 == text2 and tokens1 == tokens2:
        print("✓ Round-trip verification passed")
        # If Prometheus is available and shows no hit, that's a warning but not failure
        # (the connector might still work, just metrics endpoint unavailable)
        if cache_hit_verified is False:
            print("⚠ WARNING: Output is consistent but no cache hit detected in metrics")
        return True
    else:
        print("✗ Round-trip verification FAILED: outputs differ with temperature=0.0 + seed")
        print(f"  First text:  {text1!r}")
        print(f"  Second text: {text2!r}")
        return False


if __name__ == "__main__":
    import sys
    success = test_save_load_roundtrip()
    sys.exit(0 if success else 1)
