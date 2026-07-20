#!/usr/bin/env python3
"""Validate, aggregate and plot the final three-object fair experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "experiments/final-fair-training"
EVAL_ROOT = ROOT / "experiments/final-fair-evaluation"
OUT = ROOT / "deliverables/experiment-final"
METHODS = ("baseline", "wrist4")
OBJECTS = ("jar", "cellphone", "usb_stick")
OBJECT_LABELS = {"jar": "Jar", "cellphone": "Cellphone", "usb_stick": "USB stick"}
OBJECT_CODES = {
    "jar": "core-jar-8ec888ab36f1c5635afa616678601602",
    "cellphone": "sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6",
    "usb_stick": "sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4",
}
COLORS = {"baseline": "#4c78a8", "wrist4": "#e45756"}
MILESTONES = (1, 5, 10, 25, 50, 100)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_training() -> tuple[list[dict], dict]:
    rows = []
    payloads = {}
    for method in METHODS:
        for obj in OBJECTS:
            for seed in range(3):
                path = TRAIN_ROOT / method / obj / f"seed_{seed}" / "metrics.json"
                payload = read_json(path)
                result = payload["result"]
                payloads[(method, obj, seed)] = payload
                rows.append({
                    "method": method,
                    "object": obj,
                    "object_code": result["object_code"],
                    "seed": seed,
                    "wrist_weight": result["wrist_weight"],
                    "split_seed": result["split_seed"],
                    "split_hash": result["split_hash"],
                    "train_sequences": next(iter(result["split_summary"].values()))["train_sequences"],
                    "validation_sequences": next(iter(result["split_summary"].values()))["valid_sequences"],
                    "epochs_completed": result["epochs_completed"],
                    "best_epoch": result["best_epoch"],
                    "best_validation_loss": result["best_val_loss"],
                    "device_name": result["device_name"],
                    "torch": result["torch"],
                    "cuda_runtime": result["cuda_runtime"],
                })
    for obj in OBJECTS:
        object_rows = [row for row in rows if row["object"] == obj]
        if len({row["split_hash"] for row in object_rows}) != 1:
            raise AssertionError(f"Split hash mismatch for {obj}")
    return rows, payloads


def load_rollout(path: Path) -> dict:
    runs = read_json(path)["runs"]
    if len(runs) != 1:
        raise AssertionError(f"Expected one rollout run in {path}")
    return runs[0]


def rollout_row(run: dict, method: str, obj: str, seed: int, checkpoint: str) -> dict:
    return {
        "checkpoint": checkpoint,
        "method": method,
        "object": obj,
        "object_code": run["object_code"],
        "seed": seed,
        "epoch": run["epoch"],
        "success_count": run["success_count"],
        "trajectory_count": run["num_trajectories"],
        "success_rate": run["success_rate"],
        "mean_max_lift_m": run["mean_max_lift_m"],
        "mean_final_lift_m": run["mean_final_lift_m"],
        "checkpoint_evaluation_normalized_lift_score": run["mean_normalized_lift_score"],
    }


def load_evaluation() -> tuple[list[dict], list[dict], list[dict]]:
    best_rows, milestone_rows, trajectory_rows = [], [], []
    for method in METHODS:
        for obj in OBJECTS:
            for seed in range(3):
                path = EVAL_ROOT / "best" / method / obj / f"seed_{seed}" / "rollout_metrics.json"
                run = load_rollout(path)
                best_rows.append(rollout_row(run, method, obj, seed, "best"))
                for trajectory in run["trajectories"]:
                    trajectory_rows.append({
                        "checkpoint": "best", "method": method, "object": obj, "seed": seed,
                        **trajectory,
                    })
            for epoch in MILESTONES:
                path = EVAL_ROOT / "milestones" / method / obj / f"epoch_{epoch:03d}" / "rollout_metrics.json"
                run = load_rollout(path)
                milestone_rows.append(rollout_row(run, method, obj, 0, f"epoch_{epoch:03d}"))
                for trajectory in run["trajectories"]:
                    trajectory_rows.append({
                        "checkpoint": f"epoch_{epoch:03d}", "method": method, "object": obj,
                        "seed": 0, **trajectory,
                    })
    return best_rows, milestone_rows, trajectory_rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / "figures" / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss(training_payloads: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, obj in zip(axes, OBJECTS):
        for method in METHODS:
            history = training_payloads[(method, obj, 0)]["history"]
            epoch = [row["epoch"] for row in history]
            ax.plot(epoch, [row["train_loss"] for row in history], color=COLORS[method],
                    alpha=0.55, linestyle=":", label=f"{method} train")
            ax.plot(epoch, [row["val_loss"] for row in history], color=COLORS[method],
                    label=f"{method} validation")
        ax.set_title(OBJECT_LABELS[obj])
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Weighted BC loss")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Seed 0 train/validation BC loss")
    save_figure(fig, "bc_loss_curves")


def plot_checkpoint_curves(milestones: list[dict], best: list[dict], field: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, obj in zip(axes, OBJECTS):
        for method in METHODS:
            rows = sorted(
                (row for row in milestones if row["object"] == obj and row["method"] == method),
                key=lambda row: int(row["epoch"]),
            )
            x = [int(row["epoch"]) for row in rows]
            y = [row[field] for row in rows]
            ax.plot(x, y, marker="o", color=COLORS[method], label=method)
            train = read_json(TRAIN_ROOT / method / obj / "seed_0" / "metrics.json")["result"]
            best_row = next(row for row in best if row["object"] == obj and row["method"] == method and row["seed"] == 0)
            ax.scatter([train["best_epoch"]], [best_row[field]], marker="*", s=120,
                       color=COLORS[method], edgecolor="black", linewidth=0.5)
        ax.set_title(OBJECT_LABELS[obj])
        ax.set_xlabel("Epoch (* = best validation checkpoint)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend()
    fig.suptitle("Seed 0 checkpoint evaluation")
    save_figure(fig, stem)


def plot_best_bars(best: list[dict]) -> None:
    x = np.arange(len(OBJECTS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for index, method in enumerate(METHODS):
        means, stds = [], []
        for obj in OBJECTS:
            values = [row["success_rate"] for row in best if row["method"] == method and row["object"] == obj]
            means.append(np.mean(values) * 100)
            stds.append(np.std(values, ddof=1) * 100)
        positions = x + (index - 0.5) * width
        bars = ax.bar(positions, means, width, yerr=stds, capsize=4, color=COLORS[method], label=method)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x, [OBJECT_LABELS[obj] for obj in OBJECTS])
    ax.set_ylabel("Success rate (%) — mean ± sample SD, 3 seeds")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Best-checkpoint closed-loop success")
    save_figure(fig, "three_object_success_bars")


def aggregate(best: list[dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        for obj in (*OBJECTS, "aggregate"):
            selected = [row for row in best if row["method"] == method and (obj == "aggregate" or row["object"] == obj)]
            if obj == "aggregate":
                rates = np.asarray([
                    sum(row["success_count"] for row in selected if row["seed"] == seed)
                    / sum(row["trajectory_count"] for row in selected if row["seed"] == seed)
                    for seed in range(3)
                ], dtype=float)
                lift = np.asarray([
                    np.mean([
                        row["checkpoint_evaluation_normalized_lift_score"]
                        for row in selected if row["seed"] == seed
                    ])
                    for seed in range(3)
                ], dtype=float)
            else:
                rates = np.asarray([row["success_rate"] for row in selected], dtype=float)
                lift = np.asarray([row["checkpoint_evaluation_normalized_lift_score"] for row in selected], dtype=float)
            rows.append({
                "method": method,
                "object": obj,
                "seed_count": len(rates),
                "successes": sum(row["success_count"] for row in selected),
                "trajectories": sum(row["trajectory_count"] for row in selected),
                "mean_success_rate": float(rates.mean()),
                "sample_std_success_rate": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
                "mean_checkpoint_evaluation_normalized_lift_score": float(lift.mean()),
                "sample_std_normalized_lift_score": float(lift.std(ddof=1)) if len(lift) > 1 else 0.0,
            })
    return rows


def write_config(training_rows: list[dict]) -> None:
    split_manifest = {}
    for obj in OBJECTS:
        result = read_json(TRAIN_ROOT / "baseline" / obj / "seed_0" / "metrics.json")["result"]
        split_manifest[obj] = {
            "object_code": OBJECT_CODES[obj],
            "split_hash": result["split_hash"],
            "split_seed": result["split_seed"],
            "split_summary": result["split_summary"],
        }
    (OUT / "configs" / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    yaml_text = """protocol: three-object-fair-bc-v1
objects:
  jar: core-jar-8ec888ab36f1c5635afa616678601602
  cellphone: sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6
  usb_stick: sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4
split:
  unit: complete_sequence
  train_fraction: 0.8
  validation_fraction: 0.2
  split_seed: 0
training:
  seeds: [0, 1, 2]
  batch_size: 256
  learning_rate: 0.0002
  max_epochs: 200
  validation_patience: 30
  checkpoint_selection: minimum_validation_loss
  milestone_epochs: [1, 5, 10, 25, 50, 100]
methods:
  baseline: {wrist_weight: 2, orientation_weight: 1, finger_weight: 1, l1_weight: 1}
  wrist4: {wrist_weight: 4, orientation_weight: 1, finger_weight: 1, l1_weight: 1}
execution:
  gpu: NVIDIA GeForce RTX 5070 Ti
  compute_capability: sm_120
  training: PyTorch 2.7.1+cu128 on CUDA GPU
  evaluation_policy: PyTorch 1.12.1+cu113 on CPU
  evaluation_physics: Isaac Gym Preview 4 GPU PhysX on cuda:0 with CPU tensor pipeline
evaluation:
  trajectories_per_object: 20
  normalized_lift_score: clip(max_lift_m / 0.30, 0, 1)
"""
    (OUT / "configs" / "fair_experiment.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    for directory in (OUT / "figures", OUT / "results", OUT / "configs", OUT / "renders"):
        directory.mkdir(parents=True, exist_ok=True)
    training_rows, training_payloads = load_training()
    best_rows, milestone_rows, trajectory_rows = load_evaluation()
    aggregate_rows = aggregate(best_rows)
    write_csv(OUT / "results" / "training_summary.csv", training_rows)
    write_csv(OUT / "results" / "best_rollout_summary.csv", best_rows)
    write_csv(OUT / "results" / "milestone_rollout_summary.csv", milestone_rows)
    write_csv(OUT / "results" / "trajectory_metrics.csv", trajectory_rows)
    write_csv(OUT / "results" / "aggregate_summary.csv", aggregate_rows)
    summary = {
        "protocol": "three-object-fair-bc-v1",
        "reference_results": {
            "official_other_object_checkpoint": {"successes": 1, "trajectories": 60, "rate": 1 / 60},
            "preliminary_joint_three_object_model": {"successes": 11, "trajectories": 60, "rate": 11 / 60},
        },
        "aggregate": aggregate_rows,
        "best_runs": best_rows,
    }
    (OUT / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    write_config(training_rows)
    plot_loss(training_payloads)
    plot_checkpoint_curves(milestone_rows, best_rows, "success_rate", "Success rate", "checkpoint_success_curves")
    plot_checkpoint_curves(
        milestone_rows, best_rows, "checkpoint_evaluation_normalized_lift_score",
        "Checkpoint evaluation normalized lift score", "checkpoint_normalized_lift_curves",
    )
    plot_best_bars(best_rows)
    print(json.dumps({"aggregate": aggregate_rows, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
