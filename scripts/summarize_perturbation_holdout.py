#!/usr/bin/env python3
"""Build paired statistics and figures for the frozen perturbation holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECTS = ("jar", "cellphone", "usb_stick")
DISPLAY_NAMES = {"jar": "Jar", "cellphone": "Cellphone", "usb_stick": "USB stick"}


def load_run(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if len(payload.get("runs", [])) != 1:
        raise ValueError(f"Expected one rollout run in {path}")
    return payload["runs"][0]


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int = 20000) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    hashes = set()
    for object_name in OBJECTS:
        for seed in range(3):
            run = load_run(
                args.evaluation_root / f"member_seed_{seed}" / object_name / "rollout_metrics.json"
            )
            runs[(object_name, f"seed_{seed}")] = run
            hashes.add(run["holdout_manifest_sha256"])
        run = load_run(args.evaluation_root / "ensemble" / object_name / "rollout_metrics.json")
        runs[(object_name, "ensemble")] = run
        hashes.add(run["holdout_manifest_sha256"])
    if len(hashes) != 1:
        raise ValueError(f"Mismatched holdout manifests: {sorted(hashes)}")

    case_rows = []
    for object_name in OBJECTS:
        member_trajectories = [
            runs[(object_name, f"seed_{seed}")]["trajectories"] for seed in range(3)
        ]
        ensemble_trajectories = runs[(object_name, "ensemble")]["trajectories"]
        for trajectory_id in range(20):
            ensemble_row = ensemble_trajectories[trajectory_id]
            perturbation = ensemble_row["holdout_perturbation"]
            row = {
                "object": object_name,
                "trajectory_id": trajectory_id,
                **{key: perturbation[key] for key in (
                    "position_x_m", "position_y_m", "yaw_deg", "friction", "mass_kg"
                )},
                "ensemble_success": int(ensemble_row["success"]),
                "ensemble_normalized_lift": ensemble_row["normalized_lift_score"],
            }
            for seed, trajectories in enumerate(member_trajectories):
                row[f"seed_{seed}_success"] = int(trajectories[trajectory_id]["success"])
                row[f"seed_{seed}_normalized_lift"] = trajectories[trajectory_id][
                    "normalized_lift_score"
                ]
            case_rows.append(row)

    with (args.results_dir / "perturbation_holdout_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=case_rows[0].keys())
        writer.writeheader()
        writer.writerows(case_rows)

    summaries = []
    for object_name in (*OBJECTS, "aggregate"):
        selected = case_rows if object_name == "aggregate" else [
            row for row in case_rows if row["object"] == object_name
        ]
        summary = {"object": object_name, "trajectories": len(selected)}
        for seed in range(3):
            summary[f"seed_{seed}_successes"] = sum(row[f"seed_{seed}_success"] for row in selected)
            summary[f"seed_{seed}_success_rate"] = summary[f"seed_{seed}_successes"] / len(selected)
        member_successes = sum(summary[f"seed_{seed}_successes"] for seed in range(3))
        summary["member_successes"] = member_successes
        summary["member_trials"] = len(selected) * 3
        summary["member_mean_success_rate"] = member_successes / (len(selected) * 3)
        summary["ensemble_successes"] = sum(row["ensemble_success"] for row in selected)
        summary["ensemble_success_rate"] = summary["ensemble_successes"] / len(selected)
        summary["ensemble_minus_member_mean_pp"] = 100 * (
            summary["ensemble_success_rate"] - summary["member_mean_success_rate"]
        )
        summaries.append(summary)

    with (args.results_dir / "perturbation_holdout_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    success_differences = np.asarray([
        row["ensemble_success"]
        - np.mean([row[f"seed_{seed}_success"] for seed in range(3)])
        for row in case_rows
    ])
    lift_differences = np.asarray([
        row["ensemble_normalized_lift"]
        - np.mean([row[f"seed_{seed}_normalized_lift"] for seed in range(3)])
        for row in case_rows
    ])
    rng = np.random.default_rng(20260720)
    paired = {
        "success_rate_difference_pp": float(success_differences.mean() * 100),
        "success_rate_difference_95pct_cluster_bootstrap_pp": [
            value * 100 for value in bootstrap_interval(success_differences, rng)
        ],
        "normalized_lift_difference": float(lift_differences.mean()),
        "normalized_lift_difference_95pct_cluster_bootstrap": bootstrap_interval(
            lift_differences, rng
        ),
        "ensemble_unique_successes_no_member_succeeded": sum(
            row["ensemble_success"]
            and not any(row[f"seed_{seed}_success"] for seed in range(3))
            for row in case_rows
        ),
        "ensemble_failures_at_least_one_member_succeeded": sum(
            not row["ensemble_success"]
            and any(row[f"seed_{seed}_success"] for seed in range(3))
            for row in case_rows
        ),
        "per_member_paired": {},
    }
    for seed in range(3):
        wins = sum(
            row["ensemble_success"] and not row[f"seed_{seed}_success"] for row in case_rows
        )
        losses = sum(
            not row["ensemble_success"] and row[f"seed_{seed}_success"] for row in case_rows
        )
        paired["per_member_paired"][f"seed_{seed}"] = {
            "ensemble_wins": wins,
            "ensemble_losses": losses,
            "ties": len(case_rows) - wins - losses,
        }

    aggregate = summaries[-1]
    payload = {
        "status": "frozen_independent_perturbation_holdout",
        "holdout_manifest_sha256": next(iter(hashes)),
        "protocol": runs[("jar", "ensemble")]["holdout_protocol"],
        "comparison": (
            "Frozen object routing and seed-0/1/2 action-mean ensemble versus all three "
            "constituent members on identical paired perturbations."
        ),
        "summaries": summaries,
        "paired_analysis": paired,
        "decision": (
            "The ensemble improves the average and worst-seed deployment result but does not beat "
            "the best member; the bootstrap interval includes zero, so retain it as a stability "
            "option rather than claiming a statistically established success-rate improvement."
        ),
        "aggregate": aggregate,
    }
    with (args.results_dir / "perturbation_holdout_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)

    labels = [DISPLAY_NAMES[name] for name in OBJECTS] + ["Aggregate"]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for index, method in enumerate(("seed_0", "seed_1", "seed_2", "ensemble")):
        key = f"{method}_success_rate"
        values = [summary[key] * 100 for summary in summaries]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=method.replace("_", " ").title())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 43)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, fontsize=9)
    ax.set_title("Frozen pose/friction/mass holdout: members vs action-mean ensemble")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.figures_dir / f"perturbation_holdout_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
