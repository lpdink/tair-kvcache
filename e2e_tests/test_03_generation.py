"""
TEST 3: Simple Text Generation (Baseline)

This test verifies:
1. Model can generate text with connector enabled
2. Generation produces reasonable output
3. No crashes during inference
"""

from vllm import LLM, SamplingParams
from e2e_tests.utils import (
    MODEL_PATH,
    get_connector_config,
    print_header,
    print_section,
)


def test_simple_generation() -> bool:
    """Simple text generation with connector enabled"""
    print_header("TEST 3: Simple Text Generation (Baseline)")
    
    print_section("[3.1] Loading model...")
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
    
    print_section("[3.2] Generating text...")
    try:
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"✓ Generation successful")
            print(f"  Prompt: {output.prompt!r}")
            print(f"  Generated: {output.outputs[0].text!r}")
            print(f"  Tokens: {len(output.outputs[0].token_ids)}")
        
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_simple_generation()
    sys.exit(0 if success else 1)
