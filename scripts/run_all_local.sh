#!/usr/bin/env bash
set -euxo pipefail

# --- location ---
cd "$(dirname "$0")/.."   # repo root: meo-continual-learning

# --- venv ---
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt || true
pip install pyyaml

# --- sanity: MPS present? ---
python - <<'PY'
import torch
print("Device:", "mps" if torch.backends.mps.is_available() else "cpu")
PY

mkdir -p results/logs logs

# --------------------------
# A) FINETUNE (full 10-task)
# --------------------------
.venv/bin/python -u -m src.train \
  --config configs/finetune_cifar100.yaml \
  --output_dir results/finetune_full | tee logs/finetune_full.log

# ------------------------------------
# B) EWC sweep (10 epochs) -> aggregate
# ------------------------------------
export SEED=42
export LAMBDAS="0.03 0.1 0.3 1 3 10 30"
export EPOCHS=10
bash scripts/run_ewc_grid.sh | tee logs/ewc_grid_10E.log

# summarize sweep
.venv/bin/python scripts/aggregate_results.py \
  --pattern "results/logs/ewc_seed${SEED}_lam*.json" | tee logs/ewc_aggregate.log || true

# ------------------------------------------------
# C) Extract best λ and re-run that λ at 20 epochs
# ------------------------------------------------
python - <<'PY'
import json, pathlib, sys
p=pathlib.Path("results/logs/ewc_grid_summary.json")
if not p.exists():
    print("No ewc_grid_summary.json found"); sys.exit(0)
s=json.load(open(p))
print("BEST_LAMBDA", s["best_lambda"])
open("best_lambda.txt","w").write(str(s["best_lambda"]))
PY

best=$(cat best_lambda.txt)
cp "configs/ewc_lambda_${best}.yaml" "configs/ewc_lambda_${best}_20E.yaml"
# bump epochs_per_task to 20 (BSD sed fallback included)
sed -i '' -e 's/epochs_per_task: 10/epochs_per_task: 20/' "configs/ewc_lambda_${best}_20E.yaml" || \
sed -i 's/epochs_per_task: 10/epochs_per_task: 20/' "configs/ewc_lambda_${best}_20E.yaml"

.venv/bin/python -u -m src.train \
  --config "configs/ewc_lambda_${best}_20E.yaml" \
  --output_dir "results/ewc_lambda_${best}_20E" | tee "logs/ewc_${best}_20E.log"

# ----------------------------------
# D) (Optional) one MEO confirmation
# ----------------------------------
if [ -f configs/meo_cifar100.yaml ]; then
  .venv/bin/python -u -m src.train \
    --config configs/meo_cifar100.yaml \
    --output_dir results/meo_full | tee logs/meo_full.log
fi

# -----------------------------------
# E) Write morning summary to a file
# -----------------------------------
python - <<'PY'
import json, glob, os, pathlib
out = pathlib.Path("results/summary_morning.txt")
lines = []
# finetune
f = "results/finetune_full/results.json"
if os.path.exists(f):
    d = json.load(open(f))
    lines += [f"FINETUNE_final_avg_acc={d.get('final_avg_accuracy')}"]
# ewc 20E
cands = sorted(glob.glob("results/ewc_lambda_*_20E/results.json"))
if cands:
    d = json.load(open(cands[-1]))
    lines += [f"EWC_20E_final_avg_acc={d.get('final_avg_accuracy')}"]
# meo
m = "results/meo_full/results.json"
if os.path.exists(m):
    d = json.load(open(m))
    lines += [f"MEO_final_avg_acc={d.get('final_avg_accuracy')}"]
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
PY

echo "All done. See results/summary_morning.txt and logs/*.log"