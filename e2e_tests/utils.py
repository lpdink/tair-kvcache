"""
Shared utilities and configuration for E2E tests
"""

import re
import time
from typing import Dict, Any, Optional

import requests


# ============================================================================
# Configuration
# ============================================================================
KVCMSERVER_HOST = "localhost"
KVCMSERVER_PORT = 6382
KVCMSERVER_ADMIN_PORT = 6492
KVCMSERVER_BASE_URL = f"http://{KVCMSERVER_HOST}:{KVCMSERVER_PORT}"
KVCMSERVER_ADMIN_URL = f"http://{KVCMSERVER_HOST}:{KVCMSERVER_ADMIN_PORT}"

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


def get_register_payload(tp_size: int = 1) -> Dict[str, Any]:
    """Get register instance payload with hybrid specs.
    
    Args:
        tp_size: Tensor parallel size (default: 1)
    """
    # Generate location specs for each TP rank
    location_spec_infos = []
    attn_specs = []
    hybrid_specs = []
    
    for rank in range(tp_size):
        attn_spec_name = f"tp{rank}"
        hybrid_spec_name = f"tp{rank}_hybrid"
        location_spec_infos.append({"name": attn_spec_name, "size": 1024 * 1024 * 100})
        location_spec_infos.append({"name": hybrid_spec_name, "size": 1024 * 1024 * 50})
        attn_specs.append(attn_spec_name)
        hybrid_specs.append(hybrid_spec_name)
    
    return {
        "trace_id": "test_register_001",
        "instance_id": "test-instance-001",
        "instance_group": "test-group",
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": MODEL_NAME,
            "dtype": DTYPE,
            "use_mla": False,
            "tp_size": tp_size,
            "dp_size": 1,
            "pp_size": 1,
        },
        "location_spec_infos": location_spec_infos,
        "location_spec_groups": [
            {"name": "Full", "spec_names": attn_specs},
            {"name": "FullAndHybrid", "spec_names": attn_specs + hybrid_specs},
        ],
    }


def get_kvcm_metrics(metric_name: str) -> Optional[float]:
    """Query KVCM Prometheus metrics endpoint.
    
    Multiple instances may each have their own counter; this function
    sums all matching values for counter-type metrics.
    
    Args:
        metric_name: Metric name to query (e.g., "kvcm_manager_get_cache_location_hit_block_counter")
        
    Returns:
        Sum of metric values as float, or None if metric not found or endpoint unavailable.
    """
    try:
        resp = requests.get(f"{KVCMSERVER_ADMIN_URL}/metrics", timeout=5)
        if resp.status_code != 200:
            print(f"⚠ Prometheus endpoint returned {resp.status_code}")
            return None
        
        # Parse Prometheus text format
        # Multiple lines may match (one per instance/label set) — sum them.
        pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([\d.eE+-]+)$'
        total = 0.0
        found = False
        for line in resp.text.splitlines():
            match = re.match(pattern, line)
            if match:
                total += float(match.group(1))
                found = True
        
        return total if found else None
    except Exception as e:
        print(f"⚠ Failed to query Prometheus metrics: {e}")
        return None


def print_header(title: str):
    """Print test header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_section(title: str):
    """Print test section"""
    print(f"\n{title}")
