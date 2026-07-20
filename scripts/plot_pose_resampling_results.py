#!/usr/bin/env python3
"""Plot the independent pose-hard resampling development and test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = json.loads(args.development_summary.read_text(encoding="utf-8"))
    test = json.loads(args.test_summary.read_text(encoding="utf-8"))
    labels = ["Jar", "Cellphone", "USB stick", "Aggregate"]
    keys = ["jar", "cellphone", "usb_stick"]
    baseline_test = [test["objects"][key]["baseline_success_rate"] * 100 for key in keys]
    method_test = [test["objects"][key]["method_success_rate"] * 100 for key in keys]
    baseline_test.append(test["aggregate"]["baseline_success_rate"] * 100)
    method_test.append(test["aggregate"]["method_success_rate"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = ("#4C78A8", "#E45756")
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, baseline_test, width, label="Uniform baseline", color=colors[0])
    axes[0].bar(x + width / 2, method_test, width, label="Pose-hard resampling", color=colors[1])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Frozen independent test")
    axes[0].set_ylim(0, max(35, max(baseline_test + method_test) + 7))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    for index, value in enumerate(baseline_test):
        axes[0].text(index - width / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(method_test):
        axes[0].text(index + width / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)

    split_labels = ["Development", "Independent test"]
    baseline_aggregate = [
        development["aggregate"]["baseline_success_rate"] * 100,
        test["aggregate"]["baseline_success_rate"] * 100,
    ]
    method_aggregate = [
        development["aggregate"]["method_success_rate"] * 100,
        test["aggregate"]["method_success_rate"] * 100,
    ]
    y = np.arange(2)
    axes[1].bar(y - width / 2, baseline_aggregate, width, color=colors[0])
    axes[1].bar(y + width / 2, method_aggregate, width, color=colors[1])
    axes[1].axhline(25, color="#666666", linestyle="--", linewidth=1, label="25% aggregate target")
    axes[1].set_xticks(y, split_labels)
    axes[1].set_ylabel("Aggregate success rate (%)")
    axes[1].set_title("Result replicated across two manifests")
    axes[1].set_ylim(0, 30)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)
    for index, value in enumerate(baseline_aggregate):
        axes[1].text(index - width / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(method_aggregate):
        axes[1].text(index + width / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=8)

    fig.suptitle("Pose-stratified hard-trajectory resampling: negative result", fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "pose_resampling_comparison.png", dpi=180)
    svg_path = args.output_dir / "pose_resampling_comparison.svg"
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
