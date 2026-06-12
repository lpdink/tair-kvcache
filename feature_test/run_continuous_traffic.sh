#!/bin/bash
#
# Continuous traffic generator for KVCacheManager revisit interval histogram
# Runs worker threads that continuously write/read with controlled intervals
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${1:-10}"  # default 10 workers per instance, pass arg to override

exec python3 "${SCRIPT_DIR}/test_revisit_interval.py" --continuous --workers "${WORKERS}"
