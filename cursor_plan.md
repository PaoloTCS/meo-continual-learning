# Cursor Plan: EWC in-protocol + Paper Update

## Goal
Run an EWC λ-grid on CIFAR-100 (10 tasks, ResNet-50) on Apple Silicon (MPS), choose best λ by final average accuracy, update `paper/MEO_Paper_v2_clean.tex` table and caption accordingly, rebuild the PDF.

## Scope
- Keep hyperparameters identical to the paper (20 epochs per task, etc.).
- Online EWC with diagonal Fisher.
- Log JSON results to `results/logs/`.
- Provide scripts: `scripts/run_ewc_grid.sh`, `scripts/aggregate_results.py`, `scripts/update_tex_after_ewc.py`.
- Optional: Makefile targets to run everything with `make`.

## Non-goals
- No refactor of training loop beyond flags/logging.
- No new figures beyond current placeholders.

## Acceptance Criteria
- Running `scripts/run_ewc_grid.sh` prints a table of λ vs final_avg_acc and the selected best λ.
- Running `python scripts/update_tex_after_ewc.py --tex paper/MEO_Paper_v2_clean.tex --ewc <BEST>` updates the TeX table and removes the asterisk note in the caption.
- `pdflatex paper/MEO_Paper_v2_clean.tex` builds a PDF showing the new EWC value and no asterisk note.

## Interfaces / Flags
- `src/train.py` supports:
  - `--method {finetune, meo, ewc}`
  - `--dataset cifar100 --tasks 10 --model resnet50`
  - `--epochs-per-task 20 --batch-size 128`
  - `--lr 0.01 --momentum 0.9 --weight-decay 5e-4 --schedule cosine`
  - `--seed 42 --device mps --num-workers 4`
  - `--ewc-mode online --ewc-lambda <int> --ewc-gamma 0.9`
  - `--logdir results/logs`

## Logging contract
At the end of training, `train.py`:
- prints `[FINAL] final_avg_acc=<float>`
- writes JSON `{ "method": "...", "lambda": <int or null>, "final_avg_acc": <float>, "seed": 42 }` to `results/logs/{method}_seed42[_lam<L>].json`.

## Makefile (optional)
Targets:
- `make ewc-grid`
- `make update-tex EWC=63.4`
- `make pdf`
