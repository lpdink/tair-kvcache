#!/usr/bin/env python3
"""
Integration test: Revisit interval histogram with controlled per-instance distributions.

Architecture: fire-and-forget worker threads.

Each instance has N worker threads. Each worker independently loops:
  1. Generate random cache keys
  2. Write them (StartWriteCache + FinishWriteCache)
  3. Sleep for strategy.next_interval() seconds
  4. Read them back (GetCacheLocation) — this triggers Observe()
  5. Repeat with new keys

This creates continuous traffic where at any moment, workers are at different
stages of their cycle. The per-instance distribution converges to the strategy's
target distribution as observations accumulate.

Models:
  - Qwen3.7-Max:     uniform 25% across ≤1s, 1-5s, 5-30s, 30-60s
  - GLM-5.1:         skewed 40/30/20/10% across same buckets
  - DeepSeek-V4-Pro: uniform random 1-70s

Usage:
  python3 feature_test/test_revisit_interval.py               # quick single-shot test
  python3 feature_test/test_revisit_interval.py --continuous   # worker threads, run forever
  python3 feature_test/test_revisit_interval.py --workers 20   # more workers = more throughput
"""

import sys
import re
import random
import time
import argparse
import threading
from abc import ABC, abstractmethod
from datetime import datetime

import requests

META_URL = "http://localhost:6382"
ADMIN_URL = "http://localhost:6492"
INSTANCE_GROUP = "default"
BLOCK_SIZE = 128

# Conservative sleep targets: well inside each bucket to absorb overhead
BUCKET_TARGETS = {
    "le1": 0.5,     # → ≤1s bucket
    "1to5": 3.0,    # → 1-5s bucket
    "5to30": 15.0,  # → 5-30s bucket
    "30to60": 45.0,  # → 30-60s bucket
}


# ---------------------------------------------------------------------------
# Strategy pattern: sleep interval selection per model
# ---------------------------------------------------------------------------

class SleepStrategy(ABC):
    @abstractmethod
    def next_interval(self) -> float:
        """Return sleep duration in seconds."""

    @abstractmethod
    def label(self) -> str:
        """Human-readable description."""


class UniformBucketStrategy(SleepStrategy):
    """25% each across 4 buckets."""

    def next_interval(self) -> float:
        key = random.choice(list(BUCKET_TARGETS.keys()))
        return BUCKET_TARGETS[key]

    def label(self) -> str:
        return "uniform 25% × {≤1s, 1-5s, 5-30s, 30-60s}"


class WeightedBucketStrategy(SleepStrategy):
    """Weighted distribution across 4 buckets."""

    def __init__(self, weights: dict[str, float]):
        self._keys = list(weights.keys())
        self._weights = list(weights.values())

    def next_interval(self) -> float:
        key = random.choices(self._keys, self._weights, k=1)[0]
        return BUCKET_TARGETS[key]

    def label(self) -> str:
        parts = [f"{k}:{int(v*100)}%" for k, v in zip(self._keys, self._weights)]
        return "weighted " + ", ".join(parts)


class RandomRangeStrategy(SleepStrategy):
    """Uniform random within a range."""

    def __init__(self, lo: float, hi: float):
        self._lo = lo
        self._hi = hi

    def next_interval(self) -> float:
        return random.uniform(self._lo, self._hi)

    def label(self) -> str:
        return f"uniform [{self._lo}s, {self._hi}s]"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODELS: dict[str, SleepStrategy] = {
    "Qwen3.7-Max": UniformBucketStrategy(),
    "GLM-5.1": WeightedBucketStrategy({"le1": 0.4, "1to5": 0.3, "5to30": 0.2, "30to60": 0.1}),
    "DeepSeek-V4-Pro": RandomRangeStrategy(1.0, 70.0),
}


# ---------------------------------------------------------------------------
# KVCM API helpers (thread-safe: each thread uses its own session)
# ---------------------------------------------------------------------------

_session = threading.local()


def get_session():
    if not hasattr(_session, "http"):
        _session.http = requests.Session()
    return _session.http


def post_json(path, data):
    resp = get_session().post(f"{META_URL}{path}", json=data, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    code = body.get("header", {}).get("status", {}).get("code", "")
    if code != "OK":
        msg = body.get("header", {}).get("status", {}).get("message", "")
        raise RuntimeError(f"{path} failed: {code} - {msg}")
    return body


def ensure_registered(instance_id: str):
    post_json("/api/registerInstance", {
        "trace_id": f"worker-{threading.current_thread().name}",
        "instance_group": INSTANCE_GROUP,
        "instance_id": instance_id,
        "block_size": BLOCK_SIZE,
        "model_deployment": {
            "model_name": "test_model", "dtype": "FP8",
            "use_mla": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
        },
        "location_spec_infos": [{"name": "tp0", "size": 1024}],
    })


def write_blocks(instance_id: str, keys: list[int]):
    resp = post_json("/api/startWriteCache", {
        "trace_id": f"worker-{threading.current_thread().name}",
        "instance_id": instance_id,
        "block_keys": keys,
        "write_timeout_seconds": 100,
    })
    session_id = resp.get("write_session_id", "")
    post_json("/api/finishWriteCache", {
        "trace_id": f"worker-{threading.current_thread().name}",
        "instance_id": instance_id,
        "write_session_id": session_id,
        "success_blocks": {"bool_masks": {"values": [True] * len(keys)}},
    })


def query_blocks(instance_id: str, keys: list[int]) -> int:
    resp = post_json("/api/getCacheLocation", {
        "trace_id": f"worker-{threading.current_thread().name}",
        "instance_id": instance_id,
        "query_type": "QT_PREFIX_MATCH",
        "block_keys": keys,
    })
    return len(resp.get("locations", []))


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def worker(instance_id: str, strategy: SleepStrategy, worker_id: int, stop_event: threading.Event):
    """
    One worker thread: loop forever writing → sleeping → reading.
    Each iteration uses unique random keys to avoid cache hits from previous iterations.
    """
    ensure_registered(instance_id)

    iteration = 0
    while not stop_event.is_set():
        iteration += 1
        # Generate unique keys: use worker_id + iteration to ensure no collision
        num_blocks = random.randint(3, 8)  # vary block count to avoid patterns
        base = random.randint(1, 1_000_000) * 1000
        keys = [base + i for i in range(num_blocks)]

        try:
            write_blocks(instance_id, keys)
        except Exception as e:
            print(f"  [{instance_id}/w{worker_id}] write error: {e}")
            time.sleep(1)
            continue

        sleep_time = strategy.next_interval()
        stop_event.wait(timeout=sleep_time)
        if stop_event.is_set():
            break

        try:
            hits = query_blocks(instance_id, keys)
        except Exception as e:
            print(f"  [{instance_id}/w{worker_id}] read error: {e}")
            continue

        # Log occasionally (every 5th iteration per worker)
        if iteration % 5 == 1:
            ts = datetime.now().strftime("%H:%M:%S")
            bucket = "≤1s" if sleep_time <= 1 else "1-5s" if sleep_time <= 5 else "5-30s" if sleep_time <= 30 else "30-60s" if sleep_time <= 60 else ">60s"
            print(f"  [{ts}] {instance_id}/w{worker_id}: iter={iteration}, sleep={sleep_time:.1f}s ({bucket}), hits={hits}")


# ---------------------------------------------------------------------------
# Verify metrics
# ---------------------------------------------------------------------------

def verify_metrics():
    """Print histogram observation counts from /metrics endpoint."""
    try:
        resp = requests.get(f"{ADMIN_URL}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"  metrics endpoint error: {e}")
        return

    for model in MODELS:
        count = 0.0
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"^kvcm_revisit_interval_seconds_count\{[^}]*\}\s+([\d.]+)", line)
            if m and f'instance_id="{model}"' in line:
                count = float(m.group(1))
        print(f"  {model}: {int(count)} observations")


def print_summary():
    """Print bucket distribution percentages from Prometheus."""
    try:
        resp = requests.get(f"{ADMIN_URL}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception:
        return

    for model in MODELS:
        buckets = {}
        total = 0.0
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"^kvcm_revisit_interval_seconds_bucket\{[^}]*\}\s+([\d.]+)", line)
            if m and f'instance_id="{model}"' in line:
                le_m = re.search(r'le="([^"]+)"', line)
                if le_m:
                    buckets[le_m.group(1)] = float(m.group(1))
            m = re.match(r"^kvcm_revisit_interval_seconds_count\{[^}]*\}\s+([\d.]+)", line)
            if m and f'instance_id="{model}"' in line:
                total = float(m.group(1))

        if total <= 0:
            print(f"  {model}: no observations yet")
            continue

        prev = 0
        parts = []
        for le in ["1", "5", "30", "60"]:
            val = buckets.get(le, 0)
            diff = val - prev
            pct = diff / total * 100
            parts.append(f"≤{le}s:{pct:.0f}%")
            prev = val

        over60 = total - buckets.get("60", 0)
        pct_over = over60 / total * 100
        parts.append(f">60s:{pct_over:.0f}%")

        print(f"  {model} ({int(total)} obs): " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single_cycle():
    """Quick single-shot test: one cycle per model, sequential."""
    print("Revisit Interval Histogram — Quick Test")
    print("=" * 60)

    for model in MODELS:
        ensure_registered(model)

    for model, strategy in MODELS.items():
        num_blocks = 5
        base = random.randint(1, 100000) * 100
        keys = [base + i for i in range(num_blocks)]
        write_blocks(model, keys)
        sleep_time = strategy.next_interval()
        time.sleep(sleep_time)
        hits = query_blocks(model, keys)
        print(f"  {model}: sleep={sleep_time:.1f}s, hits={hits}")

    verify_metrics()


def run_continuous(num_workers: int):
    """Run worker threads for all instances, printing stats periodically."""
    print("Revisit Interval Histogram — Continuous Traffic Generator")
    print("=" * 60)
    print(f"  Workers per instance: {num_workers}")
    print(f"  Total workers: {num_workers * len(MODELS)}")
    print(f"  Strategy:")
    for model, strategy in MODELS.items():
        print(f"    {model}: {strategy.label()}")
    print(f"\nPress Ctrl+C to stop\n")

    # Register all instances from main thread
    for model in MODELS:
        ensure_registered(model)

    stop_event = threading.Event()
    threads = []

    for model, strategy in MODELS.items():
        for w in range(num_workers):
            t = threading.Thread(
                target=worker,
                args=(model, strategy, w, stop_event),
                name=f"{model}-w{w}",
                daemon=True,
            )
            threads.append(t)
            t.start()

    # Stagger thread starts slightly to avoid thundering herd
    time.sleep(0.5)

    start_time = time.time()
    stats_cycle = 0

    try:
        while not stop_event.is_set():
            # Print stats every 30 seconds
            time.sleep(30)
            stats_cycle += 1
            elapsed = time.time() - start_time
            ts = datetime.now().strftime("%H:%M:%S")
            obs_per_sec = ""

            print(f"\n[{ts}] === Stats after {elapsed:.0f}s ({stats_cycle * 30}s interval) ===")
            print_summary()
            print()

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nStopping {len(threads)} workers (ran for {elapsed:.0f}s)...")
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        print(f"\nFinal metrics:")
        print_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Revisit interval histogram test")
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Run continuous worker threads until Ctrl+C")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="Number of worker threads per instance (default: 10)")
    args = parser.parse_args()

    if args.continuous:
        run_continuous(args.workers)
    else:
        run_single_cycle()


if __name__ == "__main__":
    main()
