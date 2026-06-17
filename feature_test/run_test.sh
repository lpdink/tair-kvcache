#!/bin/bash
# One-click integration test for per-group revisit interval bucket configuration.
#
# Usage:
#   cd tair-kvcache
#   bash feature_test/run_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo " Per-Group Revisit Bucket Integration Test"
echo "============================================"

# 1. Start environment
echo ""
echo "[1/4] Starting KVCM + Prometheus + Grafana..."
docker compose -f feature_test/docker-compose-test.yaml up -d
echo "  Waiting for KVCM to compile and start (~2 min)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:6492/metrics > /dev/null 2>&1; then
        echo "  KVCM ready."
        break
    fi
    sleep 2
done

# 2. Run test
echo ""
echo "[2/4] Running integration test..."
python3 feature_test/test_per_group_buckets.py
TEST_EXIT=$?

# 3. Show Grafana URL
echo ""
echo "[3/4] Grafana dashboard: http://localhost:12111 (admin/admin)"
echo "  Dashboard: KVCacheManager - Per-Group Revisit Interval Buckets"

# 4. Cleanup
echo ""
echo "[4/4] Stopping environment..."
docker compose -f feature_test/docker-compose-test.yaml down

exit $TEST_EXIT
