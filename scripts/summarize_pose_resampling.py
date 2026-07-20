#!/usr/bin/env python3
"""Summarize paired baseline and pose-hard rollout matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


OBJECTS = ("jar", "cellphone", "usb_stick")
SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path, expected_manifest_hash: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("runs", [])) != 1:
        raise ValueError(f"Expected one run in {path}")
    run = payload["runs"][0]
    if run.get("holdout_manifest_sha256") != expected_manifest_hash:
        raise ValueError(f"Manifest hash mismatch in {path}")
    trajectory_ids = [int(row["trajectory_id"]) for row in run["trajectories"]]
    if trajectory_ids != list(range(20)):
        raise ValueError(f"Expected paired trajectory ids 0..19 in {path}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    manifest_hash = sha256(args.manifest)
    result = {
        "protocol": "pose_stratified_hard_resampling_evaluation_v1",
        "label": args.label,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "targets": {"aggregate_success_rate": 0.25, "usb_stick_success_rate": 0.15},
        "objects": {},
    }
    csv_rows = []
    paired_success_clusters = []
    paired_lift_clusters = []
    paired_wins = 0
    paired_losses = 0
    paired_ties = 0
    for object_name in OBJECTS:
        method_successes = 0
        baseline_successes = 0
        method_lifts = []
        baseline_lifts = []
        seeds = []
        object_success_deltas = []
        object_lift_deltas = []
        for seed in SEEDS:
            baseline_path = args.baseline_root / f"member_seed_{seed}" / object_name / "rollout_metrics.json"
            method_path = args.method_root / f"member_seed_{seed}" / object_name / "rollout_metrics.json"
            baseline = load_run(baseline_path, manifest_hash)
            method = load_run(method_path, manifest_hash)
            baseline_successes += int(baseline["success_count"])
            method_successes += int(method["success_count"])
            baseline_lifts.append(float(baseline["mean_normalized_lift_score"]))
            method_lifts.append(float(method["mean_normalized_lift_score"]))
            seeds.append({
                "seed": seed,
                "baseline_successes": int(baseline["success_count"]),
                "method_successes": int(method["success_count"]),
                "baseline_success_rate": float(baseline["success_rate"]),
                "method_success_rate": float(method["success_rate"]),
            })
            baseline_trajectories = baseline["trajectories"]
            method_trajectories = method["trajectories"]
            success_deltas = np.asarray([
                int(method_row["success"]) - int(baseline_row["success"])
                for baseline_row, method_row in zip(baseline_trajectories, method_trajectories)
            ])
            lift_deltas = np.asarray([
                float(method_row["normalized_lift_score"])
                - float(baseline_row["normalized_lift_score"])
                for baseline_row, method_row in zip(baseline_trajectories, method_trajectories)
            ])
            paired_wins += int(np.sum(success_deltas > 0))
            paired_losses += int(np.sum(success_deltas < 0))
            paired_ties += int(np.sum(success_deltas == 0))
            object_success_deltas.append(success_deltas)
            object_lift_deltas.append(lift_deltas)
        paired_success_clusters.extend(np.stack(object_success_deltas).mean(axis=0).tolist())
        paired_lift_clusters.extend(np.stack(object_lift_deltas).mean(axis=0).tolist())
        row = {
            "object": object_name,
            "trials": 60,
            "baseline_successes": baseline_successes,
            "baseline_success_rate": baseline_successes / 60,
            "method_successes": method_successes,
            "method_success_rate": method_successes / 60,
            "delta_percentage_points": (method_successes - baseline_successes) / 60 * 100,
            "baseline_mean_normalized_lift_score": float(np.mean(baseline_lifts)),
            "method_mean_normalized_lift_score": float(np.mean(method_lifts)),
        }
        result["objects"][object_name] = {**row, "seeds": seeds}
        csv_rows.append(row)

    aggregate_baseline = sum(row["baseline_successes"] for row in csv_rows)
    aggregate_method = sum(row["method_successes"] for row in csv_rows)
    aggregate = {
        "object": "aggregate",
        "trials": 180,
        "baseline_successes": aggregate_baseline,
        "baseline_success_rate": aggregate_baseline / 180,
        "method_successes": aggregate_method,
        "method_success_rate": aggregate_method / 180,
        "delta_percentage_points": (aggregate_method - aggregate_baseline) / 180 * 100,
        "baseline_mean_normalized_lift_score": float(np.mean([
            row["baseline_mean_normalized_lift_score"] for row in csv_rows
        ])),
        "method_mean_normalized_lift_score": float(np.mean([
            row["method_mean_normalized_lift_score"] for row in csv_rows
        ])),
    }
    csv_rows.append(aggregate)
    result["aggregate"] = aggregate
    rng = np.random.default_rng(20260720)
    success_clusters = np.asarray(paired_success_clusters)
    lift_clusters = np.asarray(paired_lift_clusters)
    bootstrap_indices = rng.integers(
        0, len(success_clusters), size=(20000, len(success_clusters)),
    )
    success_bootstrap = success_clusters[bootstrap_indices].mean(axis=1) * 100.0
    lift_bootstrap = lift_clusters[bootstrap_indices].mean(axis=1)
    result["paired_analysis"] = {
        "cluster_definition": "object-trajectory cluster averaged across three fixed training seeds",
        "bootstrap_seed": 20260720,
        "bootstrap_replicates": 20000,
        "success_rate_delta_percentage_points": aggregate["delta_percentage_points"],
        "success_rate_delta_95_percent_ci": np.quantile(
            success_bootstrap, [0.025, 0.975]
        ).tolist(),
        "normalized_lift_delta": aggregate["method_mean_normalized_lift_score"]
        - aggregate["baseline_mean_normalized_lift_score"],
        "normalized_lift_delta_95_percent_ci": np.quantile(
            lift_bootstrap, [0.025, 0.975]
        ).tolist(),
        "paired_wins": paired_wins,
        "paired_losses": paired_losses,
        "paired_ties": paired_ties,
    }
    result["target_status"] = {
        "aggregate_at_least_25_percent": aggregate["method_success_rate"] >= 0.25,
        "usb_stick_at_least_15_percent": (
            result["objects"]["usb_stick"]["method_success_rate"] >= 0.15
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"pose_resampling_{args.label}_summary.json"
    csv_path = args.output_dir / f"pose_resampling_{args.label}_summary.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result["aggregate"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(result["target_status"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
