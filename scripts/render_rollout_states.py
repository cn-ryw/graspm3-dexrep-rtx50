#!/usr/bin/env python3
"""Render evaluator rollout states to deterministic GIF, MP4 and key frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


CASE_LABELS = ("start", "contact", "lift", "end")
BOUNDS = ((-0.28, 0.28), (-0.28, 0.28), (0.42, 1.18))
OBJECT_LABELS = {
    "core-jar-8ec888ab36f1c5635afa616678601602": "Jar",
    "sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6": "Cellphone",
    "sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4": "USB stick",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Evaluator rollout_states.npz")
    parser.add_argument("--metrics", type=Path, required=True, help="rollout_metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def load_run(metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run in {metrics_path}, got {len(runs)}")
    return runs[0]


def render_frame(hand: np.ndarray, obj: np.ndarray, title: str) -> Image.Image:
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], s=0.35, c="#d95f9f", alpha=0.80)
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=0.70, c="#4c78a8", alpha=0.92)
    xx, yy = np.meshgrid(np.linspace(BOUNDS[0][0], BOUNDS[0][1], 2),
                         np.linspace(BOUNDS[1][0], BOUNDS[1][1], 2))
    ax.plot_surface(xx, yy, np.full_like(xx, 0.60), color="#bdbdbd", alpha=0.18, shade=False)
    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(*BOUNDS[1])
    ax.set_zlim(*BOUNDS[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba[:, :, :3].copy())
    plt.close(fig)
    return image


def save_mp4(images: list[Image.Image], path: Path, fps: int) -> None:
    width, height = images[0].size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 writer: {path}")
    for image in images:
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()


def render_case(
    data: np.lib.npyio.NpzFile, run: dict, case: str, output_dir: Path, fps: int,
    source_state_file: Path,
) -> None:
    prefix = f"visual_{case}"
    hand_points = data[prefix + "_hand_points"]
    object_points = data[prefix + "_object_points"]
    frame_steps = data[prefix + "_frame_steps"].astype(int)
    key_steps = data[prefix + "_key_steps"].astype(int)
    trajectory_id = int(data[prefix + "_trajectory_id"])
    trajectory = next(item for item in run["trajectories"] if item["trajectory_id"] == trajectory_id)
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    method_label = run["method"].replace(
        "holdout_adaptive_ensemble_", "holdout ensemble / "
    ).replace("adaptive_ensemble_", "ensemble / ")
    for index, step in enumerate(frame_steps):
        title = (
            f"{OBJECT_LABELS.get(run['object_code'], run['object_code'])} | "
            f"{method_label}\nseed {run['seed']} | {case} | step {step}"
        )
        frames.append(render_frame(hand_points[index], object_points[index], title))

    gif_path = case_dir / f"{case}.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
        loop=0, optimize=True,
    )
    save_mp4(frames, case_dir / f"{case}.mp4", fps)

    key_images = []
    for label, step in zip(CASE_LABELS, key_steps):
        frame_index = int(np.argmin(np.abs(frame_steps - step)))
        image = frames[frame_index].copy()
        image.save(case_dir / f"{label}.png")
        key_images.append((label, int(frame_steps[frame_index]), image))

    sheet = Image.new("RGB", (1280, 960), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, step, image) in enumerate(key_images):
        x = (index % 2) * 640
        y = (index // 2) * 480
        sheet.paste(image, (x, y))
        draw.rectangle((x + 8, y + 8, x + 168, y + 38), fill="white")
        draw.text((x + 14, y + 14), f"{label} / step {step}", fill="black")
    sheet.save(case_dir / "contact_sheet.png")

    case_metrics = {
        "object_code": run["object_code"],
        "method": run["method"],
        "seed": run["seed"],
        "epoch": run["epoch"],
        "case": case,
        "trajectory": trajectory,
        "source_state_file": source_state_file.name,
        "render": {
            "fps": fps,
            "camera": {"elevation_deg": 24, "azimuth_deg": -58},
            "bounds_m": {"x": BOUNDS[0], "y": BOUNDS[1], "z": BOUNDS[2]},
            "colors": {"hand": "#d95f9f", "object": "#4c78a8"},
            "key_steps": dict(zip(CASE_LABELS, map(int, key_steps))),
        },
    }
    (case_dir / "metrics.json").write_text(
        json.dumps(case_metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(f"rendered {case}: {gif_path}")


def main() -> None:
    args = parse_args()
    run = load_run(args.metrics)
    with np.load(args.input) as data:
        cases = [case for case in ("success", "failure") if f"visual_{case}_hand_points" in data]
        if not cases:
            raise RuntimeError("No visual payload found; evaluate with DEXGRASP_EXPORT_VISUALS=1")
        for case in cases:
            render_case(data, run, case, args.output_dir, args.fps, args.input)


if __name__ == "__main__":
    main()
