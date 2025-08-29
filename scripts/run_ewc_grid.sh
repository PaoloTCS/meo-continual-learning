#!/bin/bash
# Run EWC lambda grid search on CIFAR-100

set -e

echo "Running EWC lambda grid search on CIFAR-100..."
echo "=============================================="

# Create results directory
mkdir -p results/logs

# Lambda values to test (from paper)
LAMBDAS=(10 50 100 200)

# Run EWC for each lambda
for lambda in "${LAMBDAS[@]}"; do
    echo "Running EWC with lambda=$lambda..."
    
    # Create config for this lambda
    cat > configs/ewc_lambda_${lambda}.yaml << EOF
seed: 42
model:
  name: "resnet50"
  pretrained: false
  num_classes: 100

data:
  root: "./data"
  num_tasks: 10
  batch_size: 128
  num_workers: 4

method:
  type: "ewc"
  lambda_ewc: ${lambda}
  gamma: 0.9
  mode: "online"
  fisher_batches: 200
  fisher_batch_size: 128

training:
  epochs_per_task: 20
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4
  scheduler: "cosine"

device: "mps"
EOF
    
    # Run training
    python src/train.py \
        --config configs/ewc_lambda_${lambda}.yaml \
        --output_dir results/ewc_lambda_${lambda}
    
    echo "Completed lambda=$lambda"
    echo "---"
done

echo "EWC grid search complete!"
echo "Results saved to results/logs/"
echo ""
echo "Running aggregation script..."
python scripts/aggregate_results.py
