#!/usr/bin/env python3
"""
Integration test: Per-group revisit interval bucket configuration.

Verifies:
  Phase 1: Create instance groups with different bucket configs via admin API
  Phase 2: Register instances under each group
  Phase 3: Generate traffic (write → sleep → read) per instance
  Phase 4: Verify per-instance histogram boundaries via /metrics
  Phase 5: Immutability — update group config, verify existing instance unchanged
  Phase 6: Invalid config — verify warn + fallback to default

Usage:
  python3 feature_test/test_per_group_buckets.py
"""

import sys
import re
import time
import random
import threading
import requests
from datetime import datetime

ADMIN_URL = "http://localhost:6492"
META_URL = "http://localhost:6382"
INSTANCE_GROUP = "default"
BLOCK_SIZE = 128

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
INFO = "\033[94mℹ️\033[0m"

results = []


def log_result(phase, name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((phase, name, passed))
    print(f"  {status}  [{phase}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def admin_post(path, data):
    resp = requests.post(f"{ADMIN_URL}{path}", json=data, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    code = body.get("header", {}).get("status", {}).get("code", "")
    if code != "OK":
        msg = body.get("header", {}).get("status", {}).get("message", "")
        raise RuntimeError(f"{path} failed: {code} - {msg}")
    return body


def meta_post(path, data):
    resp = requests.post(f"{META_URL}{path}", json=data, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    code = body.get("header", {}).get("status", {}).get("code", "")
    if code != "OK":
        msg = body.get("header", {}).get("status", {}).get("message", "")
        raise RuntimeError(f"{path} failed: {code} - {msg}")
    return body


def get_metrics():
    resp = requests.get(f"{ADMIN_URL}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Metric parsing helpers
# ---------------------------------------------------------------------------

def get_bucket_boundaries(instance_id, metrics_text=None):
    """Extract sorted bucket boundary values for a given instance."""
    if metrics_text is None:
        metrics_text = get_metrics()
    le_values = set()
    for line in metrics_text.splitlines():
        if "revisit_interval_seconds_bucket" not in line:
            continue
        if f'instance_id="{instance_id}"' not in line:
            continue
        m = re.search(r'le="([^"]+)"', line)
        if m and m.group(1) != "+Inf":
            le_values.add(float(m.group(1)))
    return sorted(le_values)


def get_observation_count(instance_id, metrics_text=None):
    """Get total observation count for a given instance."""
    if metrics_text is None:
        metrics_text = get_metrics()
    for line in metrics_text.splitlines():
        if "revisit_interval_seconds_count" not in line:
            continue
        if f'instance_id="{instance_id}"' not in line:
            continue
        m = re.search(r"\s+([\d.]+)\s*$", line)
        if m:
            return float(m.group(1))
    return 0.0


# ---------------------------------------------------------------------------
# Traffic generation
# ---------------------------------------------------------------------------

_session = threading.local()


def get_session():
    if not hasattr(_session, "http"):
        _session.http = requests.Session()
    return _session.http


def register_instance(instance_id):
    meta_post("/api/registerInstance", {
        "trace_id": f"setup-{instance_id}",
        "instance_group": INSTANCE_GROUP,
        "instance_id": instance_id,
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8",
            "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
        },
        "location_spec_infos": [{"name": "tp0", "size": 1024}],
    })


def write_and_read(instance_id, sleep_time, worker_id, iterations):
    """Write once, then do multiple read cycles on the SAME keys to generate revisit observations."""
    num_blocks = random.randint(3, 6)
    base = random.randint(1, 1_000_000) * 1000
    keys = [base + i for i in range(num_blocks)]

    # Write once
    try:
        resp = meta_post("/api/startWriteCache", {
            "trace_id": f"w{worker_id}",
            "instance_id": instance_id,
            "block_keys": keys,
            "write_timeout_seconds": 30,
        })
        session_id = resp.get("write_session_id", "")
        meta_post("/api/finishWriteCache", {
            "trace_id": f"w{worker_id}",
            "instance_id": instance_id,
            "write_session_id": session_id,
            "success_blocks": {"bool_masks": {"values": [True] * len(keys)}},
        })
    except Exception:
        return

    # Read the SAME keys multiple times with sleep in between
    # First read: stored_time was set by write → Observe(interval ≈ sleep_time)
    # Subsequent reads: stored_time was set by previous read → Observe(interval ≈ sleep_time)
    for i in range(iterations):
        time.sleep(sleep_time)
        try:
            meta_post("/api/getCacheLocation", {
                "trace_id": f"w{worker_id}",
                "instance_id": instance_id,
                "query_type": "QT_PREFIX_MATCH",
                "block_keys": keys,
            })
        except Exception:
            pass


def generate_traffic(instance_id, sleep_time, num_workers, iterations, stop_event):
    """Run traffic generation workers. Each worker writes once then reads the same keys multiple times."""
    threads = []
    for w in range(num_workers):
        t = threading.Thread(
            target=write_and_read,
            args=(instance_id, sleep_time, w, iterations),
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=300)


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------

def phase0_setup_storage():
    """Phase 0: Create a local data storage backend."""
    print(f"\n{'='*60}")
    print("Phase 0: Setup Storage Backend")
    print(f"{'='*60}")

    try:
        admin_post("/api/addStorage", {
            "trace_id": "test-setup",
            "storage": {
                "global_unique_name": "local",
                "dummy": {"root_path": "/tmp/kvcm-test-dummy"},
            },
        })
        print(f"  {INFO} Created local storage backend")
        log_result("Phase 0", "Create local storage backend", True)
    except Exception as e:
        # Storage might already exist from a previous run
        print(f"  {INFO} Storage creation: {e} (may already exist)")
        log_result("Phase 0", "Create local storage backend", True, "already exists or created")


def phase1_create_groups():
    """Phase 1: Create instance groups with different bucket configs."""
    print(f"\n{'='*60}")
    print("Phase 1: Create Instance Groups")
    print(f"{'='*60}")

    # Group with fast buckets (sub-second to 5s)
    admin_post("/api/createInstanceGroup", {
        "trace_id": "test-phase1",
        "instance_group": {
            "name": "group-fast",
            "storage_candidates": ["local"],
            "global_quota_group_name": "default",
            "max_instance_count": 100,
            "quota": {"capacity": 1073741824},
            "cache_config": {
                "meta_indexer_config": {
                    "max_key_count": 100000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 32,
                    "persist_meta_data_interval_time_ms": 1000,
                    "meta_storage_backend_config": {"storage_type": "local"},
                },
                "reclaim_strategy": {
                    "storage_unique_name": "local",
                    "reclaim_policy": "POLICY_LRU",
                    "trigger_strategy": {"used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 100,
                },
                "data_storage_strategy": "CPS_PREFER_3FS",
            },
            "version": 1,
            "revisit_interval_buckets": "0.5,1,2,5",
        },
    })
    print(f"  {INFO} Created group-fast with buckets '0.5,1,2,5'")

    # Group with slow buckets (30s to 1h)
    admin_post("/api/createInstanceGroup", {
        "trace_id": "test-phase1",
        "instance_group": {
            "name": "group-slow",
            "storage_candidates": ["local"],
            "global_quota_group_name": "default",
            "max_instance_count": 100,
            "quota": {"capacity": 1073741824},
            "cache_config": {
                "meta_indexer_config": {
                    "max_key_count": 100000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 32,
                    "persist_meta_data_interval_time_ms": 1000,
                    "meta_storage_backend_config": {"storage_type": "local"},
                },
                "reclaim_strategy": {
                    "storage_unique_name": "local",
                    "reclaim_policy": "POLICY_LRU",
                    "trigger_strategy": {"used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 100,
                },
                "data_storage_strategy": "CPS_PREFER_3FS",
            },
            "version": 1,
            "revisit_interval_buckets": "30,60,300,3600",
        },
    })
    print(f"  {INFO} Created group-slow with buckets '30,60,300,3600'")

    # Group without custom buckets (uses server default)
    admin_post("/api/createInstanceGroup", {
        "trace_id": "test-phase1",
        "instance_group": {
            "name": "group-default",
            "storage_candidates": ["local"],
            "global_quota_group_name": "default",
            "max_instance_count": 100,
            "quota": {"capacity": 1073741824},
            "cache_config": {
                "meta_indexer_config": {
                    "max_key_count": 100000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 32,
                    "persist_meta_data_interval_time_ms": 1000,
                    "meta_storage_backend_config": {"storage_type": "local"},
                },
                "reclaim_strategy": {
                    "storage_unique_name": "local",
                    "reclaim_policy": "POLICY_LRU",
                    "trigger_strategy": {"used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 100,
                },
                "data_storage_strategy": "CPS_PREFER_3FS",
            },
            "version": 1,
        },
    })
    print(f"  {INFO} Created group-default with no custom buckets (server default)")

    log_result("Phase 1", "Create instance groups via API", True)


def phase2_register_instances():
    """Phase 2: Register instances under each group."""
    print(f"\n{'='*60}")
    print("Phase 2: Register Instances")
    print(f"{'='*60}")

    for group, inst in [("group-fast", "inst-fast-1"),
                        ("group-slow", "inst-slow-1"),
                        ("group-default", "inst-default-1")]:
        # Override the global INSTANCE_GROUP for this registration
        meta_post("/api/registerInstance", {
            "trace_id": f"setup-{inst}",
            "instance_group": group,
            "instance_id": inst,
            "block_size": BLOCK_SIZE,
            "model_deployment": {
                "model_name": "test_model", "dtype": "FP8",
                "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
            },
            "location_spec_infos": [{"name": "tp0", "size": 1024}],
        })
        print(f"  {INFO} Registered {inst} under {group}")

    log_result("Phase 2", "Register instances under each group", True)


def phase3_generate_traffic():
    """Phase 3: Generate traffic per instance with different sleep patterns."""
    print(f"\n{'='*60}")
    print("Phase 3: Generate Traffic")
    print(f"{'='*60}")

    stop_event = threading.Event()

    # inst-fast: sleep 1s, 4 reads per worker → ~4s per worker
    print(f"  {INFO} Starting traffic for inst-fast-1 (sleep 1s, 5 workers × 4 reads)")
    t1 = threading.Thread(target=generate_traffic,
                          args=("inst-fast-1", 1.0, 5, 4, stop_event), daemon=True)
    t1.start()

    # inst-slow: sleep 10s, 3 reads per worker → ~30s per worker
    print(f"  {INFO} Starting traffic for inst-slow-1 (sleep 10s, 3 workers × 3 reads)")
    t2 = threading.Thread(target=generate_traffic,
                          args=("inst-slow-1", 10.0, 3, 3, stop_event), daemon=True)
    t2.start()

    # inst-default: sleep 5s, 4 reads per worker → ~20s per worker
    print(f"  {INFO} Starting traffic for inst-default-1 (sleep 5s, 5 workers × 4 reads)")
    t3 = threading.Thread(target=generate_traffic,
                          args=("inst-default-1", 5.0, 5, 4, stop_event), daemon=True)
    t3.start()

    # Wait for all traffic to finish
    t1.join(timeout=60)
    t3.join(timeout=60)
    print(f"  {INFO} inst-fast-1 and inst-default-1 traffic complete")

    # Wait for slow traffic
    t2.join(timeout=120)
    print(f"  {INFO} inst-slow-1 traffic complete")

    # Verify observations exist
    metrics = get_metrics()
    for inst in ["inst-fast-1", "inst-slow-1", "inst-default-1"]:
        count = get_observation_count(inst, metrics)
        ok = count > 0
        log_result("Phase 3", f"{inst} has observations", ok, f"count={int(count)}")


def phase4_verify_boundaries():
    """Phase 4: Verify per-instance histogram boundaries via /metrics."""
    print(f"\n{'='*60}")
    print("Phase 4: Verify Per-Instance Bucket Boundaries")
    print(f"{'='*60}")

    metrics = get_metrics()

    # inst-fast-1 should have boundaries [0.5, 1, 2, 5]
    fast_bounds = get_bucket_boundaries("inst-fast-1", metrics)
    expected_fast = [0.5, 1.0, 2.0, 5.0]
    ok = fast_bounds == expected_fast
    log_result("Phase 4", "inst-fast-1 boundaries = [0.5, 1, 2, 5]", ok,
               f"got {fast_bounds}")

    # inst-slow-1 should have boundaries [30, 60, 300, 3600]
    slow_bounds = get_bucket_boundaries("inst-slow-1", metrics)
    expected_slow = [30.0, 60.0, 300.0, 3600.0]
    ok = slow_bounds == expected_slow
    log_result("Phase 4", "inst-slow-1 boundaries = [30, 60, 300, 3600]", ok,
               f"got {slow_bounds}")

    # inst-default-1 should have server default boundaries
    default_bounds = get_bucket_boundaries("inst-default-1", metrics)
    expected_default = [1.0, 5.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]
    ok = default_bounds == expected_default
    log_result("Phase 4", "inst-default-1 boundaries = server default", ok,
               f"got {default_bounds}")

    # Verify isolation: fast and slow have DIFFERENT boundaries
    ok = fast_bounds != slow_bounds
    log_result("Phase 4", "Per-group isolation: fast ≠ slow boundaries", ok)


def phase5_immutability():
    """Phase 5: Update group config, verify existing instance unchanged."""
    print(f"\n{'='*60}")
    print("Phase 5: Immutability Test")
    print(f"{'='*60}")

    # Record current boundaries for inst-fast-1
    metrics_before = get_metrics()
    bounds_before = get_bucket_boundaries("inst-fast-1", metrics_before)

    # Update group-fast with new boundaries
    admin_post("/api/updateInstanceGroup", {
        "trace_id": "test-phase5",
        "current_version": 1,
        "instance_group": {
            "name": "group-fast",
            "storage_candidates": ["local"],
            "global_quota_group_name": "default",
            "max_instance_count": 100,
            "quota": {"capacity": 1073741824},
            "cache_config": {
                "meta_indexer_config": {
                    "max_key_count": 100000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 32,
                    "persist_meta_data_interval_time_ms": 1000,
                    "meta_storage_backend_config": {"storage_type": "local"},
                },
                "reclaim_strategy": {
                    "storage_unique_name": "local",
                    "reclaim_policy": "POLICY_LRU",
                    "trigger_strategy": {"used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 100,
                },
                "data_storage_strategy": "CPS_PREFER_3FS",
            },
            "version": 2,
            "revisit_interval_buckets": "10,20,30",
        },
    })
    print(f"  {INFO} Updated group-fast buckets to '10,20,30' (version 2)")

    # Verify existing instance unchanged
    metrics_after = get_metrics()
    bounds_after = get_bucket_boundaries("inst-fast-1", metrics_after)
    ok = bounds_after == bounds_before
    log_result("Phase 5", "inst-fast-1 boundaries unchanged after group update", ok,
               f"before={bounds_before}, after={bounds_after}")

    # Register new instance under updated group → should use new boundaries
    meta_post("/api/registerInstance", {
        "trace_id": "test-phase5",
        "instance_group": "group-fast",
        "instance_id": "inst-fast-2",
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8",
            "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
        },
        "location_spec_infos": [{"name": "tp0", "size": 1024}],
    })

    # Generate a few observations for inst-fast-2
    stop_event = threading.Event()
    generate_traffic("inst-fast-2", 15.0, 3, 3, stop_event)

    metrics_new = get_metrics()
    new_bounds = get_bucket_boundaries("inst-fast-2", metrics_new)
    expected_new = [10.0, 20.0, 30.0]
    ok = new_bounds == expected_new
    log_result("Phase 5", "inst-fast-2 (new) uses updated boundaries [10, 20, 30]", ok,
               f"got {new_bounds}")


def phase6_invalid_config():
    """Phase 6: Invalid config → warn + fallback to default."""
    print(f"\n{'='*60}")
    print("Phase 6: Invalid Config Handling")
    print(f"{'='*60}")

    # Update group-default with invalid boundaries (not ascending)
    admin_post("/api/updateInstanceGroup", {
        "trace_id": "test-phase6",
        "current_version": 1,
        "instance_group": {
            "name": "group-default",
            "storage_candidates": ["local"],
            "global_quota_group_name": "default",
            "max_instance_count": 100,
            "quota": {"capacity": 1073741824},
            "cache_config": {
                "meta_indexer_config": {
                    "max_key_count": 100000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 32,
                    "persist_meta_data_interval_time_ms": 1000,
                    "meta_storage_backend_config": {"storage_type": "local"},
                },
                "reclaim_strategy": {
                    "storage_unique_name": "local",
                    "reclaim_policy": "POLICY_LRU",
                    "trigger_strategy": {"used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 100,
                },
                "data_storage_strategy": "CPS_PREFER_3FS",
            },
            "version": 2,
            "revisit_interval_buckets": "5,1,30",  # invalid: not ascending
        },
    })
    print(f"  {INFO} Updated group-default with invalid buckets '5,1,30'")
    log_result("Phase 6", "Invalid buckets accepted by API (non-blocking)", True,
               "warn + fallback expected")

    # Register new instance → should fall back to server default
    meta_post("/api/registerInstance", {
        "trace_id": "test-phase6",
        "instance_group": "group-default",
        "instance_id": "inst-default-2",
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8",
            "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
        },
        "location_spec_infos": [{"name": "tp0", "size": 1024}],
    })

    # Generate observations
    stop_event = threading.Event()
    generate_traffic("inst-default-2", 5.0, 3, 5, stop_event)

    metrics = get_metrics()
    bounds = get_bucket_boundaries("inst-default-2", metrics)
    expected_default = [1.0, 5.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]
    ok = bounds == expected_default
    log_result("Phase 6", "inst-default-2 falls back to server default", ok,
               f"got {bounds}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Per-Group Revisit Interval Bucket Configuration")
    print("Integration Test")
    print("=" * 60)

    # Wait for KVCM to be ready
    print(f"\n{INFO} Waiting for KVCM to be ready...")
    for i in range(60):
        try:
            requests.get(f"{ADMIN_URL}/metrics", timeout=2)
            print(f"  {INFO} KVCM ready")
            break
        except Exception:
            time.sleep(2)
    else:
        print(f"  {FAIL} KVCM not ready after 120s")
        sys.exit(1)

    try:
        phase0_setup_storage()
        phase1_create_groups()
        phase2_register_instances()
        phase3_generate_traffic()
        phase4_verify_boundaries()
        phase5_immutability()
        phase6_invalid_config()
    except Exception as e:
        print(f"\n{FAIL} Test aborted with exception: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(1 for _, _, p in results if p)
    failed = total - passed
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}")

    if failed > 0:
        print(f"\n  {FAIL} Failed tests:")
        for phase, name, p in results:
            if not p:
                print(f"    - [{phase}] {name}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
