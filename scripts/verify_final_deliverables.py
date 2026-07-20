#!/usr/bin/env python3
"""Fail fast when the final experiment evidence package is incomplete or unsafe."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/experiment-final"


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    required = [
        OUT / "灵巧手抓取实验报告.md",
        OUT / "灵巧手抓取实验报告.pdf",
        OUT / "figures/bc_loss_curves.png",
        OUT / "figures/checkpoint_success_curves.png",
        OUT / "figures/checkpoint_normalized_lift_curves.png",
        OUT / "figures/three_object_success_bars.png",
        OUT / "results/summary.json",
        OUT / "results/best_rollout_summary.csv",
        OUT / "results/milestone_rollout_summary.csv",
        OUT / "results/trajectory_metrics.csv",
        OUT / "configs/fair_experiment.yaml",
        OUT / "configs/split_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise AssertionError("Missing required files: " + ", ".join(missing))
    pdf_info = subprocess.run(
        ["pdfinfo", str(OUT / "灵巧手抓取实验报告.pdf")],
        check=True, capture_output=True, text=True,
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)$", pdf_info, re.MULTILINE)
    assert page_match and 1 <= int(page_match.group(1)) <= 2, pdf_info

    best = csv_rows(OUT / "results/best_rollout_summary.csv")
    milestones = csv_rows(OUT / "results/milestone_rollout_summary.csv")
    trajectories = csv_rows(OUT / "results/trajectory_metrics.csv")
    assert len(best) == 18, len(best)
    assert len(milestones) == 36, len(milestones)
    assert len(trajectories) == 1080, len(trajectories)
    assert all(int(row["trajectory_count"]) == 20 for row in best + milestones)

    split_manifest = json.loads((OUT / "configs/split_manifest.json").read_text(encoding="utf-8"))
    assert set(split_manifest) == {"jar", "cellphone", "usb_stick"}
    for obj, manifest in split_manifest.items():
        assert len(manifest["split_hash"]) == 64, (obj, manifest["split_hash"])

    gifs = sorted((OUT / "renders").glob("**/*.gif"))
    mp4s = sorted((OUT / "renders").glob("**/*.mp4"))
    success_gifs = [path for path in gifs if path.name == "success.gif"]
    failure_gifs = [path for path in gifs if path.name == "failure.gif"]
    assert success_gifs, "No success GIF"
    assert failure_gifs, "No failure GIF"
    for obj in ("jar", "cellphone", "usb_stick"):
        assert list((OUT / "renders" / obj).glob("**/failure.gif")), f"No failure GIF for {obj}"
    for path in gifs:
        with Image.open(path) as image:
            assert getattr(image, "n_frames", 1) > 1, path
    for path in mp4s:
        video = cv2.VideoCapture(str(path))
        assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) > 1, path
        video.release()

    forbidden_extensions = {".ckpt", ".npy", ".npz"}
    forbidden = [path for path in OUT.rglob("*") if path.suffix.lower() in forbidden_extensions]
    assert not forbidden, forbidden
    secret_pattern = re.compile(
        r"/home/[^/\s]+|[A-Za-z0-9._%+-]+@(gmail|outlook|qq)\.com|" + "sk" + r"-[A-Za-z0-9_-]{10,}"
    )
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".yaml", ".yml", ".svg"}:
            assert not secret_pattern.search(path.read_text(encoding="utf-8")), path

    print(json.dumps({
        "best_runs": len(best),
        "milestone_runs": len(milestones),
        "trajectory_rows": len(trajectories),
        "gif_count": len(gifs),
        "mp4_count": len(mp4s),
        "report_pages": int(page_match.group(1)),
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
