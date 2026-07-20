#!/usr/bin/env python3
"""Select evidence-backed best-checkpoint cases and render final media."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "experiments/final-fair-evaluation/best"
OUT = ROOT / "deliverables/experiment-final/renders"
RENDERER = ROOT / "scripts/render_rollout_states.py"
OBJECTS = ("jar", "cellphone", "usb_stick")
METHODS = ("baseline", "wrist4")


def load_candidate(method: str, obj: str, seed: int) -> dict:
    source = EVAL_ROOT / method / obj / f"seed_{seed}"
    metrics_path = source / "rollout_metrics.json"
    run = json.loads(metrics_path.read_text(encoding="utf-8"))["runs"][0]
    return {
        "method": method,
        "object": obj,
        "seed": seed,
        "source": source,
        "metrics": metrics_path,
        "state": source / run["state_file"],
        "run": run,
    }


def render(candidate: dict, role: str) -> dict:
    target = OUT / candidate["object"] / role
    subprocess.run([
        sys.executable, str(RENDERER),
        "--input", str(candidate["state"]),
        "--metrics", str(candidate["metrics"]),
        "--output-dir", str(target),
        "--fps", "10",
    ], check=True)
    return {
        "object": candidate["object"],
        "role": role,
        "method": candidate["method"],
        "seed": candidate["seed"],
        "success_count": candidate["run"]["success_count"],
        "trajectory_count": candidate["run"]["num_trajectories"],
        "mean_normalized_lift_score": candidate["run"]["mean_normalized_lift_score"],
        "source_metrics": str(candidate["metrics"].relative_to(ROOT)),
    }


def main() -> None:
    candidates = [load_candidate(method, obj, seed)
                  for method in METHODS for obj in OBJECTS for seed in range(3)]
    manifest = []
    for obj in OBJECTS:
        object_candidates = [candidate for candidate in candidates if candidate["object"] == obj]
        success_candidates = [candidate for candidate in object_candidates if candidate["run"]["success_count"] > 0]
        if success_candidates:
            selected_success = max(
                success_candidates,
                key=lambda candidate: (candidate["run"]["success_count"],
                                       candidate["run"]["mean_normalized_lift_score"]),
            )
            manifest.append(render(selected_success, "selected_success_source"))
        failure_candidates = [
            candidate for candidate in object_candidates
            if candidate["run"]["success_count"] < candidate["run"]["num_trajectories"]
        ]
        if not failure_candidates:
            raise RuntimeError(f"No failure trajectory available for {obj}")
        selected_failure = max(
            failure_candidates,
            key=lambda candidate: candidate["run"]["mean_normalized_lift_score"],
        )
        if not success_candidates or selected_failure["source"] != selected_success["source"]:
            manifest.append(render(selected_failure, "selected_failure_source"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
