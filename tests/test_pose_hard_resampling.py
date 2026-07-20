from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_pose_hard_sampler_manifest",
    ROOT / "scripts" / "build_pose_hard_sampler_manifest.py",
)
trainer = None
if importlib.util.find_spec("torch") is not None:
    trainer = load_module(
        "train_bc_modern_torch",
        ROOT / "scripts" / "train_bc_modern_torch.py",
    )


class PoseHardResamplingTests(unittest.TestCase):
    def test_pose_strata_respects_rotation_threshold(self) -> None:
        identity = np.eye(3)
        angle_10 = np.radians(10)
        angle_40 = np.radians(40)
        rotate_10 = np.array([
            [np.cos(angle_10), -np.sin(angle_10), 0],
            [np.sin(angle_10), np.cos(angle_10), 0],
            [0, 0, 1],
        ])
        rotate_40 = np.array([
            [np.cos(angle_40), -np.sin(angle_40), 0],
            [np.sin(angle_40), np.cos(angle_40), 0],
            [0, 0, 1],
        ])
        assignments = builder.pose_strata(
            np.stack([identity, rotate_10, rotate_40]), [0, 1, 2], 15.0,
        )
        self.assertEqual(assignments, {0: 0, 1: 0, 2: 1})

    def test_capped_weights_keep_unit_mean_and_cap(self) -> None:
        weights = builder.normalize_capped_weights([0.1, 0.1, 10.0, 1.0], 3.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertLessEqual(float(weights.max()), 3.0)
        self.assertTrue(np.all(weights > 0))

    @unittest.skipIf(trainer is None, "PyTorch is intentionally isolated in the training container")
    def test_training_loader_rejects_wrong_split_hash(self) -> None:
        payload = {
            "protocol": "pose_stratified_hard_resampling_v1",
            "parameters": {"pose_threshold_deg": 15.0, "hardness_strength": 1.0},
            "objects": {
                "object": {
                    "split_hash": "expected",
                    "train_sequences": [{"sequence_index": 3, "sampling_weight": 1.0}],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split hash"):
                trainer.load_pose_hard_frame_weights(
                    path, "wrong", [("object", 3)],
                )

    @unittest.skipIf(trainer is None, "PyTorch is intentionally isolated in the training container")
    def test_training_loader_expands_sequence_weights_to_frames(self) -> None:
        payload = {
            "protocol": "pose_stratified_hard_resampling_v1",
            "parameters": {"pose_threshold_deg": 15.0, "hardness_strength": 1.0},
            "objects": {
                "object": {
                    "split_hash": "same",
                    "train_sequences": [
                        {"sequence_index": 3, "sampling_weight": 0.5},
                        {"sequence_index": 7, "sampling_weight": 1.5},
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            weights, metadata = trainer.load_pose_hard_frame_weights(
                path, "same", [("object", 3), ("object", 3), ("object", 7)],
            )
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.5])
        self.assertEqual(metadata["strategy"], "pose_stratified_hard_resampling")


if __name__ == "__main__":
    unittest.main()
