#!/usr/bin/env python3
"""Build a training-only pose-stratified hard-trajectory sampler manifest.

Difficulty comes exclusively from baseline rollouts on a dedicated development
perturbation manifest. The historical holdout is never read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBJECTS = {
    "jar": "core-jar-8ec888ab36f1c5635afa616678601602",
    "cellphone": "sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6",
    "usb_stick": "sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_strata(
    rotation_matrices: np.ndarray,
    train_indices: list[int],
    threshold_deg: float,
) -> dict[int, int]:
    centers: list[np.ndarray] = []
    assignments: dict[int, int] = {}
    for sequence_index in sorted(train_indices):
        rotation = rotation_matrices[sequence_index]
        distances = [rotation_distance_deg(center, rotation) for center in centers]
        if distances and min(distances) <= threshold_deg:
            stratum = int(np.argmin(distances))
        else:
            stratum = len(centers)
            centers.append(rotation)
        assignments[sequence_index] = stratum
    return assignments


def normalize_capped_weights(raw_weights: list[float], maximum: float) -> np.ndarray:
    weights = np.asarray(raw_weights, dtype=np.float64)
    if maximum < 1.0:
        raise ValueError("Maximum sequence weight must be at least 1")
    weights /= weights.mean()
    for _ in range(len(weights)):
        over = weights > maximum
        if not np.any(over):
            break
        weights[over] = maximum
        under = ~over
        if not np.any(under):
            break
        remaining_total = len(weights) - float(weights[over].sum())
        weights[under] *= remaining_total / float(weights[under].sum())
    if not np.isclose(weights.mean(), 1.0, atol=1e-12):
        raise RuntimeError("Unable to normalize capped sampling weights")
    if float(weights.max()) > maximum + 1e-12:
        raise RuntimeError("Normalized sampling weights exceed the configured cap")
    return weights


def split_for_object(sequence_count: int, valid_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    valid_count = max(1, min(sequence_count - 1, int(round(sequence_count * valid_fraction))))
    indices = rng.permutation(sequence_count)
    return indices[valid_count:].tolist(), indices[:valid_count].tolist()


def split_hash(
    object_code: str,
    sequence_count: int,
    train_indices: list[int],
    valid_indices: list[int],
) -> str:
    summary = {
        object_code: {
            "total_sequences": sequence_count,
            "train_sequences": len(train_indices),
            "valid_sequences": len(valid_indices),
            "train_indices": train_indices,
            "valid_indices": valid_indices,
        }
    }
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_development_outcomes(
    evaluation_root: Path,
    object_name: str,
    object_code: str,
    seeds: list[int],
    expected_manifest_hash: str,
) -> tuple[dict[int, list[bool]], dict[str, str]]:
    outcomes: dict[int, list[bool]] = {}
    evidence_hashes = {}
    for seed in seeds:
        path = evaluation_root / f"member_seed_{seed}" / object_name / "rollout_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing development baseline result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("runs", [])) != 1:
            raise ValueError(f"Expected exactly one rollout run in {path}")
        run = payload["runs"][0]
        if run.get("object_code") != object_code:
            raise ValueError(f"Object mismatch in {path}")
        if run.get("holdout_manifest_sha256") != expected_manifest_hash:
            raise ValueError(f"Development manifest hash mismatch in {path}")
        for trajectory in run["trajectories"]:
            trajectory_id = int(trajectory["trajectory_id"])
            outcomes.setdefault(trajectory_id, []).append(bool(trajectory["success"]))
        evidence_hashes[str(path)] = sha256(path)
    if any(len(values) != len(seeds) for values in outcomes.values()):
        raise ValueError(f"Incomplete paired development outcomes for {object_name}")
    return outcomes, evidence_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--development-evaluation-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--pose-threshold-deg", type=float, default=15.0)
    parser.add_argument("--hardness-strength", type=float, default=1.0)
    parser.add_argument("--max-sequence-weight", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite frozen sampler manifest: {args.output}")
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError("--valid-fraction must be between 0 and 1")
    if args.pose_threshold_deg <= 0 or args.hardness_strength < 0:
        raise ValueError("Pose threshold must be positive and hardness strength non-negative")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    development_manifest_hash = sha256(args.development_manifest)

    objects = {}
    evidence_hashes = {}
    for object_name, object_code in OBJECTS.items():
        cache_path = args.data_dir / f"{object_code}.npy"
        payload = np.load(cache_path, allow_pickle=True).item()
        rotations = np.asarray(payload["obj_rotmat"], dtype=np.float64)
        success_indices = np.asarray(payload["success_idx"], dtype=np.int64)
        sequence_count = len(rotations)
        if success_indices.shape != (sequence_count,):
            raise ValueError(f"Invalid success_idx shape in {cache_path}")
        train_indices, valid_indices = split_for_object(
            sequence_count, args.valid_fraction, args.split_seed,
        )
        assignments = pose_strata(rotations, train_indices, args.pose_threshold_deg)
        stratum_counts = {
            stratum: list(assignments.values()).count(stratum)
            for stratum in sorted(set(assignments.values()))
        }
        outcomes, object_evidence = load_development_outcomes(
            args.development_evaluation_root,
            object_name,
            object_code,
            seeds,
            development_manifest_hash,
        )
        evidence_hashes.update(object_evidence)

        raw_weights = []
        rows = []
        stratum_count = len(stratum_counts)
        for sequence_index in train_indices:
            raw_trajectory_id = int(success_indices[sequence_index])
            successes = outcomes.get(raw_trajectory_id)
            if successes is None:
                raise ValueError(
                    f"No development outcome for raw trajectory {raw_trajectory_id} ({object_name})"
                )
            success_rate = float(np.mean(successes))
            failure_rate = 1.0 - success_rate
            stratum = assignments[sequence_index]
            pose_balance = len(train_indices) / (stratum_count * stratum_counts[stratum])
            hardness = 1.0 + args.hardness_strength * failure_rate
            raw_weight = pose_balance * hardness
            raw_weights.append(raw_weight)
            rows.append({
                "sequence_index": sequence_index,
                "raw_trajectory_id": raw_trajectory_id,
                "pose_stratum": stratum,
                "pose_stratum_size": stratum_counts[stratum],
                "development_successes": int(sum(successes)),
                "development_trials": len(successes),
                "development_failure_rate": failure_rate,
                "pose_balance_factor": pose_balance,
                "hardness_factor": hardness,
                "object_rotation_matrix": rotations[sequence_index].round(9).tolist(),
            })

        normalized = normalize_capped_weights(raw_weights, args.max_sequence_weight)
        for row, weight in zip(rows, normalized):
            row["sampling_weight"] = float(weight)

        objects[object_code] = {
            "object_name": object_name,
            "split_hash": split_hash(
                object_code, sequence_count, train_indices, valid_indices,
            ),
            "sequence_count": sequence_count,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "pose_strata_count": stratum_count,
            "pose_stratum_sizes": {str(key): value for key, value in stratum_counts.items()},
            "train_sequences": rows,
        }

    result = {
        "protocol": "pose_stratified_hard_resampling_v1",
        "description": (
            "Complete-sequence pose balancing multiplied by development-baseline "
            "failure weighting; validation sequences and historical holdout are excluded"
        ),
        "development_manifest": str(args.development_manifest),
        "development_manifest_sha256": development_manifest_hash,
        "development_evidence_sha256": evidence_hashes,
        "parameters": {
            "seeds": seeds,
            "split_seed": args.split_seed,
            "valid_fraction": args.valid_fraction,
            "pose_threshold_deg": args.pose_threshold_deg,
            "hardness_strength": args.hardness_strength,
            "max_sequence_weight": args.max_sequence_weight,
            "samples_per_epoch": "unchanged from uniform baseline",
            "replacement": True,
        },
        "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"sha256={sha256(args.output)}")
    for object_code, data in objects.items():
        weights = [row["sampling_weight"] for row in data["train_sequences"]]
        print(
            f"{data['object_name']}: strata={data['pose_strata_count']} "
            f"weight_min={min(weights):.4f} weight_max={max(weights):.4f} "
            f"split_hash={data['split_hash']}"
        )


if __name__ == "__main__":
    main()
