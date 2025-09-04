#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("drift_json", type=str, help="Path to drift.json")
    p.add_argument("out_png", type=str, help="Output PNG path")
    args = p.parse_args()

    with open(args.drift_json, "r") as f:
        drift = json.load(f)

    per_task = sorted(drift.get("per_task", []), key=lambda x: x.get("task", 0))
    plt.figure(figsize=(5.2, 3.2), dpi=200)

    if per_task:
        xs = [item["task"] + 1 for item in per_task]  # 1-based task index
        ys = [item["drift"] for item in per_task]
        plt.plot(xs, ys, marker="o", label="MEO (per task)")
        plt.xlabel("Task")
        plt.title("MEO drift (smoke run, per-task)")
        plt.xlim(min(xs) - 0.5, max(xs) + 0.5)
    else:
        # Fallback to per-epoch view
        per_epoch = drift.get("per_epoch", [])
        curves = {}
        for item in per_epoch:
            t = item["task"]
            curves.setdefault(t, []).append(item["drift"])
        for t, ys in sorted(curves.items()):
            xs = list(range(len(ys)))
            plt.plot(xs, ys, marker="o", label=f"Task {t}")
        plt.xlabel("Epoch")
        plt.title("MEO drift (smoke run, per-epoch)")

    plt.ylabel("Normalized drift")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved drift plot -> {out_path}")


if __name__ == "__main__":
    main()
