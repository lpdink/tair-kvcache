#!/usr/bin/env python3
"""
Integration test: GetCacheLocation hit-rate metrics.

Generates realistic traffic with 80-100% hit rate for Grafana dashboard demo.

Prerequisites:
  1. KVCacheManager running (docker compose -f docker/docker-compose-test.yaml up -d)

Usage:
  python3 test/test_hit_rate_metrics.py
"""

import sys
import re
import random
import requests

META_URL = "http://localhost:6382"
ADMIN_URL = "http://localhost:6492"
TRACE_ID = "hit_rate_traffic_gen"
INSTANCE_GROUP = "default"
BLOCK_SIZE = 128

# Simulate 5 models, each with its own instance_id
MODELS = ["qwen-72b-fp8", "glm-4-fp8", "kimi-k2-fp8", "deepseek-v3-fp8", "minimax-01-fp8"]
INSTANCE_ID = random.choice(MODELS)


def post_json(base_url, path, data):
    resp = requests.post(f"{base_url}{path}", json=data, timeout=5)
    resp.raise_for_status()
    body = resp.json()
    code = body.get("header", {}).get("status", {}).get("code", "")
    if code != "OK":
        msg = body.get("header", {}).get("status", {}).get("message", "")
        raise RuntimeError(f"{path} failed: {code} - {msg}")
    return body


def parse_prometheus_counter(text, metric_name, label_filter=None):
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(rf"^{re.escape(metric_name)}(\{{[^}}]*\}})?\s+([\d.]+)", line)
        if m:
            if label_filter and label_filter not in line:
                continue
            total += float(m.group(2))
            found = True
    return total if found else None


def ensure_registered():
    """Register instance (idempotent)."""
    post_json(META_URL, "/api/registerInstance", {
        "trace_id": TRACE_ID,
        "instance_group": INSTANCE_GROUP,
        "instance_id": INSTANCE_ID,
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8",
            "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
        },
        "location_spec_infos": [{"name": "tp0", "size": 1024}],
    })


def write_blocks(keys):
    """Write blocks and mark all as successful."""
    resp = post_json(META_URL, "/api/startWriteCache", {
        "trace_id": TRACE_ID,
        "instance_id": INSTANCE_ID,
        "block_keys": keys,
        "write_timeout_seconds": 100,
    })
    session_id = resp.get("write_session_id", "")
    post_json(META_URL, "/api/finishWriteCache", {
        "trace_id": TRACE_ID,
        "instance_id": INSTANCE_ID,
        "write_session_id": session_id,
        "success_blocks": {"bool_masks": {"values": [True] * len(keys)}},
    })


def query_blocks(keys):
    """Query blocks via PrefixMatch, return hit count."""
    resp = post_json(META_URL, "/api/getCacheLocation", {
        "trace_id": TRACE_ID,
        "instance_id": INSTANCE_ID,
        "query_type": "QT_PREFIX_MATCH",
        "block_keys": keys,
    })
    return len(resp.get("locations", []))


def main():
    ensure_registered()

    # Generate random number of blocks (5-15)
    num_blocks = random.randint(5, 15)
    base = random.randint(1, 100000) * 100
    all_keys = [base + i for i in range(num_blocks)]

    # Write all blocks
    write_blocks(all_keys)

    # Generate queries with 80-100% hit rate
    # Randomly decide how many queries to make (2-5)
    num_queries = random.randint(2, 5)
    total_queried = 0
    total_hits = 0

    for _ in range(num_queries):
        # Random query size (5 to num_blocks + 2)
        query_size = random.randint(5, num_blocks + 2)
        # Most queries are 90-100% hits; some 80-90%
        hit_rate = random.choice([0.85, 0.90, 0.90, 0.95, 0.95, 1.0, 1.0, 1.0])
        num_hits = max(1, int(query_size * hit_rate))
        num_misses = query_size - num_hits

        # Pick hit keys as a contiguous prefix from written blocks
        hit_keys = all_keys[:num_hits] if num_hits <= len(all_keys) else all_keys
        # Miss keys go AFTER hits (PrefixMatch stops at first miss)
        miss_keys = [base + num_blocks + random.randint(1, 10000) for _ in range(num_misses)]

        query_keys = hit_keys + miss_keys

        hits = query_blocks(query_keys)
        total_queried += len(query_keys)
        total_hits += hits

    actual_rate = total_hits / total_queried * 100 if total_queried > 0 else 0
    print(f"queried={total_queried}, hits={total_hits}, rate={actual_rate:.1f}%")


if __name__ == "__main__":
    main()
