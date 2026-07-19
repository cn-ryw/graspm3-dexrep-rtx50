#!/usr/bin/env python3
"""Render the Gate D BC loss history and three-object rollout comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text())
    history = payload["history"]
    result = payload["result"]
    epochs = np.asarray([row["epoch"] for row in history])
    train_loss = np.asarray([row["train_loss"] for row in history])
    val_loss = np.asarray([row["val_loss"] for row in history])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].semilogy(epochs, train_loss, label="train BC loss", linewidth=1.6)
    axes[0].semilogy(epochs, val_loss, label="validation BC loss", linewidth=1.6)
    best_epoch = result["best_epoch"]
    best_loss = result["best_val_loss"]
    axes[0].scatter([best_epoch], [best_loss], marker="*", s=120, zorder=5, label="best checkpoint")
    axes[0].set(title="Three-object BC training", xlabel="epoch", ylabel="weighted BC loss (log scale)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    labels = ["Jar", "Cellphone", "USB stick"]
    baseline = np.asarray([0.05, 0.0, 0.0]) * 100
    trained = np.asarray([0.20, 0.0, 0.35]) * 100
    positions = np.arange(len(labels))
    width = 0.36
    axes[1].bar(positions - width / 2, baseline, width, label="official checkpoint")
    axes[1].bar(positions + width / 2, trained, width, label="Gate D best checkpoint")
    axes[1].set(
        title="Isaac Gym rollout (20 sequences/object)",
        ylabel="success rate (%)",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 40),
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / ("gate_d_training_and_success." + suffix), dpi=180)
    print("plot=PASS")


if __name__ == "__main__":
    main()
