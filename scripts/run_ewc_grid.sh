#!/usr/bin/env bash
set -euo pipefail
unset MallocStackLogging MallocStackLoggingNoCompact

SEED=${SEED:-42}
LAMBDAS=(${LAMBDAS:-0.03 0.1 0.3 1 3 10 30})

mkdir -p configs results/logs

for lambda in "${LAMBDAS[@]}"; do
  cat > "configs/ewc_lambda_${lambda}.yaml" <<EOF
seed: ${SEED}
device: auto
data: {root: ./data, num_tasks: 10, batch_size: 128, num_workers: 0}
model: {name: resnet50, pretrained: false, num_classes: 100}
training: {lr: 0.01, momentum: 0.9, weight_decay: 5e-4, epochs_per_task: 20}
method:
  type: ewc
  lambda_ewc: ${lambda}
  mode: online
  gamma: 0.9
  fisher_batches: 50
  fisher_batch_size: 64
EOF
  echo ">>> λ=${lambda}"
  .venv/bin/python -u -m src.train \
    --config "configs/ewc_lambda_${lambda}.yaml" \
    --output_dir "results/ewc_lambda_${lambda}"
done

echo ">>> Aggregating best λ"
.venv/bin/python scripts/aggregate_results.py --pattern "results/logs/ewc_seed${SEED}_lam*.json"
