#!/usr/bin/env python3
"""Generate a frozen Latin-hypercube perturbation manifest before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOUNDS = {
    "position_x_m": (-0.008, 0.008),
    "position_y_m": (-0.008, 0.008),
    "yaw_deg": (-8.0, 8.0),
    "friction": (0.8, 1.2),
    "mass_kg": (0.16, 0.24),
}


def stratified_dimension(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    unit_values = (np.arange(count) + rng.random(count)) / count
    return low + (high - low) * unit_values[rng.permutation(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dimensions = {
        name: stratified_dimension(rng, args.count, *bounds)
        for name, bounds in BOUNDS.items()
    }
    trajectories = []
    for trajectory_id in range(args.count):
        row = {"trajectory_id": trajectory_id}
        row.update({name: round(float(values[trajectory_id]), 9) for name, values in dimensions.items()})
        trajectories.append(row)

    payload = {
        "protocol": "frozen_combined_pose_friction_mass_holdout_v1",
        "seed": args.seed,
        "generation": "independent Latin-hypercube dimensions; generated before all holdout rollouts",
        "bounds": {name: list(bounds) for name, bounds in BOUNDS.items()},
        "trajectory_count": args.count,
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
