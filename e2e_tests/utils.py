"""
Shared utilities and configuration for E2E tests
"""

import time
from typing import Dict, Any


# ============================================================================
# Configuration
# ============================================================================
KVCMSERVER_HOST = "localhost"
KVCMSERVER_PORT = 6382
KVCMSERVER_BASE_URL = f"http://{KVCMSERVER_HOST}:{KVCMSERVER_PORT}"

MODEL_PATH = "/root/ws/resources/models/Qwen3.5-4B"
MODEL_NAME = "Qwen3.5-4B"
BLOCK_SIZE = 528
DTYPE = "bfloat16"


def get_connector_config() -> Dict[str, Any]:
    """Get TairKvCacheConnector configuration"""
    return {
        "kv_connector": "TairKvCacheConnector",
        "kv_role": "kv_both",  # Required: specifies this instance can both produce and consume KV cache
        "kv_connector_module_path": "kv_cache_manager.py_connector.vllm.v1_connector",
        "kv_connector_extra_config": {
            "manager_uri": KVCMSERVER_BASE_URL,
            "coordinator_base_port": 5555,
            "instance_group": "test-group",
            "instance_id": f"test-instance-{int(time.time())}",
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
            "log_level": "INFO",
        }
    }


def get_register_payload() -> Dict[str, Any]:
    """Get register instance payload with hybrid specs"""
    return {
        "trace_id": "test_register_001",
        "instance_id": "test-instance-001",
        "instance_group": "test-group",
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": MODEL_NAME,
            "dtype": DTYPE,
            "use_mla": False,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
        },
        "location_spec_infos": [
            {"name": "tp0", "size": 1024 * 1024 * 100},
            {"name": "tp0_hybrid", "size": 1024 * 1024 * 50},
        ],
        "location_spec_groups": [
            {"name": "Full", "spec_names": ["tp0"]},
            {"name": "FullAndHybrid", "spec_names": ["tp0", "tp0_hybrid"]},
        ],
    }


def print_header(title: str):
    """Print test header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_section(title: str):
    """Print test section"""
    print(f"\n{title}")
