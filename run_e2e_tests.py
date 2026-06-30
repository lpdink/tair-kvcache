#!/usr/bin/env python3
"""
TairKvCacheConnector E2E Test Suite

Main test runner that orchestrates all E2E tests and generates a report.
Tests 1-3 are single-GPU baselines; tests 4-6 run with TP=2 and verify
cache hit behavior via Prometheus metrics.

Usage:
    python run_e2e_tests.py [--test NUM] [--skip NUM]

Options:
    --test NUM    Run only test NUM (e.g., --test 6)
    --skip NUM    Skip test NUM (can be used multiple times)
"""

import sys
import argparse
from typing import Dict, Callable

# Import test modules
from e2e_tests.test_01_connectivity import test_basic_connectivity
from e2e_tests.test_02_model_loading import test_model_loading_with_connector
from e2e_tests.test_03_generation import test_simple_generation
from e2e_tests.test_04_roundtrip import test_save_load_roundtrip
from e2e_tests.test_05_multiple_cycles import test_multiple_cycles
from e2e_tests.test_06_tp_cache_verification import test_tp_cache_verification


# Test registry
TESTS: Dict[int, tuple[str, Callable[[], bool]]] = {
    1: ("Basic Connectivity", test_basic_connectivity),
    2: ("Model Loading with Connector", test_model_loading_with_connector),
    3: ("Simple Generation", test_simple_generation),
    4: ("Save → Load Round-Trip (TP=2)", test_save_load_roundtrip),
    5: ("Multiple Cycles (TP=2)", test_multiple_cycles),
    6: ("Multi-GPU TP Cache Verification", test_tp_cache_verification),
}


def main():
    parser = argparse.ArgumentParser(description="Run E2E tests for hybrid attention support")
    parser.add_argument("--test", type=int, help="Run only this test number")
    parser.add_argument("--skip", type=int, action="append", help="Skip this test number (can be used multiple times)")
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("TairKvCacheConnector E2E Test Suite")
    print("Testing: Qwen3.5-4B (GDN/Mamba + Attention) with TP=1/2")
    print("="*80)
    
    # Determine which tests to run
    if args.test:
        tests_to_run = [args.test]
    else:
        tests_to_run = list(TESTS.keys())
    
    if args.skip:
        tests_to_run = [t for t in tests_to_run if t not in args.skip]
    
    # Run tests
    results = {}
    for test_num in tests_to_run:
        if test_num not in TESTS:
            print(f"\n⚠ Test {test_num} not found, skipping")
            continue
        
        test_name, test_func = TESTS[test_num]
        print(f"\n[Running TEST {test_num}: {test_name}]")
        
        try:
            result = test_func()
            results[test_num] = (test_name, result)
        except Exception as e:
            print(f"✗ Test {test_num} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_num] = (test_name, False)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_num in sorted(results.keys()):
        test_name, result = results[test_num]
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status}: TEST {test_num} - {test_name}")
    
    passed = sum(1 for _, result in results.values() if result is True)
    failed = sum(1 for _, result in results.values() if result is False)
    skipped = sum(1 for _, result in results.values() if result is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠ Some tests failed")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
