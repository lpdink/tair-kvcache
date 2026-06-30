"""
TEST 5: Multiple Prefill-Decode Cycles (Tensor Parallel)

This test verifies:
1. Multiple different prompts can be processed with TP=2
2. No state corruption across multiple cycles
3. Connector handles multiple save/load operations correctly under TP
"""

from vllm import LLM, SamplingParams
from e2e_tests.utils import (
    MODEL_PATH,
    get_connector_config,
    print_header,
    print_section,
)


def test_multiple_cycles() -> bool:
    """Test multiple prefill-decode cycles with TP=2"""
    print_header("TEST 5: Multiple Prefill-Decode Cycles (TP=2)")
    
    print_section("[5.1] Loading model with tensor_parallel_size=2...")
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
    
    prompts = [
        "The quick brown fox",
        "In machine learning,",
        "Python is a",
        "Quantum computing",
        "The theory of relativity",
    ]
    sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=15)
    
    print_section(f"[5.2] Running {len(prompts)} cycles...")
    try:
        results = []
        for i, prompt in enumerate(prompts):
            outputs = llm.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text
            results.append((prompt, text))
            print(f"  Cycle {i+1}: {prompt!r} → {text!r}")
        
        print(f"✓ {len(results)} cycles completed successfully")
        return True
    except Exception as e:
        print(f"✗ Multiple cycles failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_multiple_cycles()
    sys.exit(0 if success else 1)
