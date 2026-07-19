#!/usr/bin/env python3
"""Compute per-object BC errors for a modern compatible checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from train_bc_modern_torch import DexRepBC, load_sequences


@torch.no_grad()
def metrics(model, observations, actions, device):
    obs = torch.from_numpy(observations.reshape(-1, 2460)).to(device)
    target = torch.from_numpy(actions.reshape(-1, 28)).to(device)
    prediction = model(obs)
    wrist = float(functional.mse_loss(prediction[:, :3], target[:, :3]))
    orientation = float(functional.mse_loss(prediction[:, 3:6], target[:, 3:6]))
    finger = float(functional.mse_loss(prediction[:, 6:], target[:, 6:]))
    l1 = float(functional.l1_loss(prediction, target))
    return {
        "frames": len(obs),
        "weighted_bc_loss": 2 * wrist + orientation + finger + l1,
        "wrist_mse": wrist,
        "orientation_mse": orientation,
        "finger_mse": finger,
        "action_mae": l1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    model = DexRepBC().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rng = np.random.default_rng(args.seed)
    result = {}
    for path in sorted(args.data_dir.glob("*.npy")):
        observations, actions = load_sequences(path)
        sequence_count = len(observations)
        valid_count = max(1, min(sequence_count - 1, int(round(sequence_count * args.valid_fraction))))
        indices = rng.permutation(sequence_count)
        valid_indices = indices[:valid_count]
        train_indices = indices[valid_count:]
        result[path.stem] = {
            "train_sequences": len(train_indices),
            "valid_sequences": len(valid_indices),
            "train": metrics(model, observations[train_indices], actions[train_indices], device),
            "valid": metrics(model, observations[valid_indices], actions[valid_indices], device),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("PER_OBJECT_RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
