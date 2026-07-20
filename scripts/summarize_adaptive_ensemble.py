#!/usr/bin/env python3
"""Summarize the bounded object-routing and three-seed ensemble experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECTS = ("jar", "cellphone", "usb_stick")
DISPLAY_NAMES = ("Jar", "Cellphone", "USB stick")


def load_run(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    runs = payload.get("runs", [])
    if len(runs) != 1:
        raise ValueError(f"Expected one run in {path}, found {len(runs)}")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--fair-summary", type=Path, required=True)
    parser.add_argument("--single-latency", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    ensemble_runs = {
        name: load_run(args.evaluation_root / name / "rollout_metrics.json")
        for name in OBJECTS
    }
    single_run = load_run(args.single_latency)

    fair = {}
    with args.fair_summary.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            fair[(row["method"], row["object"])] = row

    rows = []
    for name, display_name in zip(OBJECTS, DISPLAY_NAMES):
        run = ensemble_runs[name]
        rows.append(
            {
                "object": name,
                "display_name": display_name,
                "selected_method": "wrist4" if name != "usb_stick" else "baseline",
                "ensemble_size": int(run["ensemble_size"]),
                "successes": int(run["success_count"]),
                "trajectories": int(run["num_trajectories"]),
                "success_rate": float(run["success_rate"]),
                "checkpoint_evaluation_normalized_lift_score": float(
                    run["mean_normalized_lift_score"]
                ),
                "policy_inference_mean_ms": float(run["policy_inference_mean_ms"]),
                "policy_inference_p95_ms": float(run["policy_inference_p95_ms"]),
            }
        )

    total_successes = sum(row["successes"] for row in rows)
    total_trajectories = sum(row["trajectories"] for row in rows)
    weighted_lift = sum(
        row["checkpoint_evaluation_normalized_lift_score"] * row["trajectories"]
        for row in rows
    ) / total_trajectories
    mean_ensemble_latency = float(np.mean([row["policy_inference_mean_ms"] for row in rows]))
    single_latency = float(single_run["policy_inference_mean_ms"])

    summary = {
        "status": "exploratory_post_hoc_deployment_optimization",
        "protocol_note": (
            "The same fixed 20 trajectories were reused and the object-specific method was "
            "selected after inspecting the fair experiment. This is not an independent fair-test claim."
        ),
        "routing": {"jar": "wrist4", "cellphone": "wrist4", "usb_stick": "baseline"},
        "ensemble": "mean of unclamped deterministic 28-D actions from seed 0/1/2 best checkpoints",
        "objects": rows,
        "aggregate": {
            "successes": total_successes,
            "trajectories": total_trajectories,
            "success_rate": total_successes / total_trajectories,
            "checkpoint_evaluation_normalized_lift_score": weighted_lift,
        },
        "latency": {
            "single_policy_reference_mean_ms": single_latency,
            "three_policy_ensemble_mean_ms": mean_ensemble_latency,
            "slowdown_ratio": mean_ensemble_latency / single_latency,
            "sixty_hz_budget_ms": 1000.0 / 60.0,
        },
        "decision": (
            "Keep the ensemble as an optional stability-oriented deployment route; do not replace "
            "the fair baseline/wrist4 conclusion until an independent perturbation holdout is evaluated."
        ),
    }
    with (args.results_dir / "adaptive_ensemble_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    with (args.results_dir / "adaptive_ensemble_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(OBJECTS))
    width = 0.25
    baseline = np.array([float(fair[("baseline", name)]["mean_success_rate"]) for name in OBJECTS])
    baseline_std = np.array(
        [float(fair[("baseline", name)]["sample_std_success_rate"]) for name in OBJECTS]
    )
    wrist4 = np.array([float(fair[("wrist4", name)]["mean_success_rate"]) for name in OBJECTS])
    wrist4_std = np.array(
        [float(fair[("wrist4", name)]["sample_std_success_rate"]) for name in OBJECTS]
    )
    ensemble = np.array([row["success_rate"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width, baseline * 100, width, yerr=baseline_std * 100, capsize=4, label="Baseline (3-seed mean)")
    ax.bar(x, wrist4 * 100, width, yerr=wrist4_std * 100, capsize=4, label="Wrist=4 (3-seed mean)")
    bars = ax.bar(x + width, ensemble * 100, width, label="Adaptive ensemble (1 fixed replay)")
    for bar, value in zip(bars, ensemble):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x, DISPLAY_NAMES)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title("Post-hoc adaptive ensemble: fixed-trajectory exploratory replay")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.figures_dir / f"adaptive_ensemble_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
