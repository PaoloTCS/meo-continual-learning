#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
cd "$(dirname "$0")/.."

mkdir -p logs
log="logs/nightly_$(date +%Y%m%d_%H%M).log"

nohup bash scripts/run_all_local.sh > "$log" 2>&1 &
echo $! > run_all_local.pid
echo "$log" > run_all_local.logpath

echo "Launched: PID=$(cat run_all_local.pid), log=$log"

