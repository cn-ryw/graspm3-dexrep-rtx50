#!/usr/bin/env python3
"""Inspect one official GraspM3 record and export human-readable previews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


EXPECTED_KEYS = {"obj_rotmat", "obj_scale", "grasp_seqs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export summaries and plots for one trusted official GraspM3 .npy file."
    )
    parser.add_argument("object_id", help="Object ID, with or without the .npy suffix")
    parser.add_argument("--trajectory", type=int, default=0, help="Trajectory index (default: 0)")
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataset"))
    parser.add_argument("--mesh-root", type=Path, default=Path("/meshdata"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    return parser.parse_args()


def finite_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "all_finite": bool(np.isfinite(array).all()),
    }


def load_record(path: Path) -> dict[str, np.ndarray]:
    # GraspM3 intentionally stores a dictionary in a scalar object array. Only
    # load records from the official dataset directory; the wrapper mounts it
    # read-only inside a network-disabled, capability-dropped container.
    payload = np.load(path, allow_pickle=True)
    if payload.shape != () or payload.dtype != object:
        raise ValueError(f"Expected a scalar object array, got shape={payload.shape} dtype={payload.dtype}")
    record = payload.item()
    if not isinstance(record, dict):
        raise TypeError(f"Expected dict payload, got {type(record).__name__}")
    missing = EXPECTED_KEYS - set(record)
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")

    arrays = {key: np.asarray(record[key]) for key in EXPECTED_KEYS}
    rotations = arrays["obj_rotmat"]
    scales = arrays["obj_scale"]
    grasps = arrays["grasp_seqs"]
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected obj_rotmat shape: {rotations.shape}")
    if scales.ndim != 1:
        raise ValueError(f"Unexpected obj_scale shape: {scales.shape}")
    if grasps.ndim != 3 or grasps.shape[2] != 28:
        raise ValueError(f"Unexpected grasp_seqs shape: {grasps.shape}")
    if not (rotations.shape[0] == scales.shape[0] == grasps.shape[0]):
        raise ValueError("Trajectory counts disagree across record arrays")
    if not all(np.issubdtype(value.dtype, np.number) for value in arrays.values()):
        raise TypeError("Expected numeric arrays inside the record")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Record contains NaN or infinity")
    return arrays


def export_csv(path: Path, trajectory: np.ndarray) -> None:
    columns = [
        "frame",
        "hand_tx_relative",
        "hand_ty_relative",
        "hand_tz_relative",
        "hand_world_x",
        "hand_world_y",
        "hand_world_z",
        "hand_rot_x",
        "hand_rot_y",
        "hand_rot_z",
    ] + [f"joint_{index:02d}" for index in range(22)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for frame, vector in enumerate(trajectory):
            relative = vector[:3]
            world = relative + np.array([0.0, 0.0, 1.0], dtype=relative.dtype)
            writer.writerow([frame, *relative.tolist(), *world.tolist(), *vector[3:].tolist()])


def plot_trajectory(path: Path, trajectory: np.ndarray, object_id: str, index: int) -> None:
    frames = np.arange(trajectory.shape[0])
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for dimension, label in enumerate(("x", "y", "z")):
        axes[0].plot(frames, trajectory[:, dimension], label=f"t{label} relative")
    axes[0].plot(frames, trajectory[:, 2] + 1.0, "--", label="world z = relative z + 1")
    axes[0].set_ylabel("translation (m)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.25)

    for dimension, label in enumerate(("rx", "ry", "rz"), start=3):
        axes[1].plot(frames, trajectory[:, dimension], label=label)
    axes[1].set_ylabel("global rotation")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    heatmap = axes[2].imshow(
        trajectory[:, 6:].T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
    )
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("joint index (0–21)")
    figure.colorbar(heatmap, ax=axes[2], label="joint angle")
    figure.suptitle(f"{object_id} — trajectory {index}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_obj(path: Path, max_faces: int = 12_000) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) >= 4:
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                indices = [int(field.split("/", 1)[0]) for field in line.split()[1:]]
                if len(indices) >= 3:
                    first = indices[0]
                    for position in range(1, len(indices) - 1):
                        faces.append([first, indices[position], indices[position + 1]])
    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.size == 0 or face_array.size == 0:
        raise ValueError(f"No renderable geometry found in {path}")
    face_array = np.where(face_array > 0, face_array - 1, len(vertex_array) + face_array)
    if len(face_array) > max_faces:
        selected = np.linspace(0, len(face_array) - 1, max_faces, dtype=np.int64)
        face_array = face_array[selected]
    return vertex_array, face_array


def plot_mesh(path: Path, obj_path: Path, title: str) -> dict[str, Any]:
    vertices, faces = load_obj(obj_path)
    triangles = vertices[faces]
    figure = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles,
        linewidths=0.05,
        edgecolors=(0.1, 0.1, 0.1, 0.16),
        facecolors=(0.16, 0.55, 0.78, 0.78),
    )
    axis.add_collection3d(collection)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(title)
    axis.view_init(elev=24, azim=42)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "path": str(obj_path),
        "vertices": int(len(vertices)),
        "triangles_rendered": int(len(faces)),
        "bounds_min": minimum.tolist(),
        "bounds_max": maximum.tolist(),
    }


def main() -> None:
    args = parse_args()
    object_id = Path(args.object_id).stem
    if not object_id or Path(object_id).name != object_id:
        raise ValueError("object_id must be a single file-style identifier")
    record_path = (args.dataset_root / f"{object_id}.npy").resolve()
    dataset_root = args.dataset_root.resolve()
    if record_path.parent != dataset_root or not record_path.is_file():
        raise FileNotFoundError(f"Record not found: {record_path}")

    arrays = load_record(record_path)
    grasps = arrays["grasp_seqs"]
    if not 0 <= args.trajectory < grasps.shape[0]:
        raise IndexError(f"trajectory must be in [0, {grasps.shape[0] - 1}]")

    output = args.output_dir / object_id / f"trajectory_{args.trajectory:03d}"
    output.mkdir(parents=True, exist_ok=True)
    trajectory = grasps[args.trajectory]
    determinants = np.linalg.det(arrays["obj_rotmat"])
    summary: dict[str, Any] = {
        "object_id": object_id,
        "source": str(record_path),
        "selected_trajectory": args.trajectory,
        "semantics": {
            "grasp_vector": "[translation_xyz, global_rotation_xyz, 22_joint_angles]",
            "translation_reference_world_point": [0.0, 0.0, 1.0],
        },
        "obj_rotmat": finite_stats(arrays["obj_rotmat"]),
        "obj_rotmat_determinant": finite_stats(determinants),
        "obj_scale": finite_stats(arrays["obj_scale"]),
        "grasp_seqs": finite_stats(grasps),
        "selected_trajectory_stats": finite_stats(trajectory),
    }

    export_csv(output / "trajectory.csv", trajectory)
    plot_trajectory(output / "trajectory_channels.png", trajectory, object_id, args.trajectory)
    mesh_path = args.mesh_root / object_id / "coacd" / "decomposed.obj"
    if mesh_path.is_file():
        summary["mesh"] = plot_mesh(output / "mesh_preview.png", mesh_path, object_id)
    else:
        summary["mesh"] = {"path": str(mesh_path), "available": False}

    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(f"object_id={object_id}")
    print(f"trajectories={grasps.shape[0]} frames={grasps.shape[1]} dimensions={grasps.shape[2]}")
    print(f"selected_trajectory={args.trajectory}")
    print(f"output={output}")
    print("GRASPM3_VIEW=PASS")


if __name__ == "__main__":
    main()
