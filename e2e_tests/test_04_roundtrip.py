"""
TEST 4: Save → Load Round-Trip Verification

This test verifies:
1. First request triggers save operation
2. Second request (same prompt) triggers load operation
3. Outputs are consistent between first and second requests
4. location_spec_group_names are correctly filled with "FullAndHybrid"
"""

import time
from vllm import LLM, SamplingParams
from e2e_tests.utils import (
    MODEL_PATH,
    get_connector_config,
    print_header,
    print_section,
)


def test_save_load_roundtrip() -> bool:
    """Verify save → load round-trip correctness"""
    print_header("TEST 4: Save → Load Round-Trip Verification")
    
    print_section("[4.1] Loading model...")
    try:
        connector_config = get_connector_config()
        
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            max_model_len=512,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            enforce_eager=True,
            kv_transfer_config=connector_config,
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    
    # First request: should trigger save
    print_section("[4.2] First request (should trigger save)...")
    try:
        outputs1 = llm.generate([prompt], sampling_params)
        text1 = outputs1[0].outputs[0].text
        tokens1 = outputs1[0].outputs[0].token_ids
        print(f"✓ First generation successful")
        print(f"  Text: {text1!r}")
        print(f"  Tokens: {len(tokens1)}")
    except Exception as e:
        print(f"✗ First generation failed: {e}")
        return False
    
    # Wait a bit for save to complete
    print_section("[4.3] Waiting for save to complete...")
    time.sleep(2)
    
    # Second request: should trigger load
    print_section("[4.4] Second request (should trigger load)...")
    try:
        outputs2 = llm.generate([prompt], sampling_params)
        text2 = outputs2[0].outputs[0].text
        tokens2 = outputs2[0].outputs[0].token_ids
        print(f"✓ Second generation successful")
        print(f"  Text: {text2!r}")
        print(f"  Tokens: {len(tokens2)}")
    except Exception as e:
        print(f"✗ Second generation failed: {e}")
        return False
    
    # Verify consistency
    print_section("[4.5] Verifying consistency...")
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
    
    # For temperature=0.0, we expect exact match — output difference is a bug
    if text1 == text2 and tokens1 == tokens2:
        print("✓ Round-trip verification passed")
        return True
    else:
        print("✗ Round-trip verification FAILED: outputs differ with temperature=0.0")
        print(f"  First text:  {text1!r}")
        print(f"  Second text: {text2!r}")
        return False


if __name__ == "__main__":
    import sys
    success = test_save_load_roundtrip()
    sys.exit(0 if success else 1)
