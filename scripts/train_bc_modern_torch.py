#!/usr/bin/env python3
"""Train the official DexRep BC MLP without importing Isaac Gym or Lightning.

The state_dict keys intentionally match LitBCModel so the output can be loaded by
the original PyTorch 1.12 inference process.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset


def observation_mask(pro_dim: int = 100) -> np.ndarray:
    fingertip_velocity = np.arange(84, 149).reshape(5, 13)[:, -6:].reshape(-1)
    hand_velocity = np.arange(28, 56)
    hand_force = np.arange(56, 84)
    fingertip_force = np.arange(149, 179)
    object_velocity = np.arange(216, 222)
    removed = np.concatenate(
        [hand_force, fingertip_force, fingertip_velocity, object_velocity, hand_velocity]
    )
    if pro_dim != 100:
        raise ValueError("This compatibility trainer currently supports pro_dim=100 only")
    return ~np.isin(np.arange(2582), removed)


def load_sequences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mask = observation_mask()
    payload = np.load(path, allow_pickle=True).item()
    observations = payload["obs"][..., mask].astype(np.float32)
    actions = payload["vis_unscale_actions"].astype(np.float32)
    # Match GraspM3DexRepDataset.__getitem__: its final flattened frame maps
    # back by 39 positions (frame 69 -> frame 30 for the 70-frame records).
    if observations.shape[1] > 39:
        observations[:, -1] = observations[:, -40]
        actions[:, -1] = actions[:, -40]
    return observations, actions


def load_split(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for path in paths:
        obs, act = load_sequences(path)
        observations.append(obs.reshape(-1, 2460))
        actions.append(act.reshape(-1, 28))
    return torch.from_numpy(np.concatenate(observations)), torch.from_numpy(np.concatenate(actions))


def load_sequence_split(
    paths: list[Path], valid_fraction: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    train_observations: list[np.ndarray] = []
    train_actions: list[np.ndarray] = []
    valid_observations: list[np.ndarray] = []
    valid_actions: list[np.ndarray] = []
    split_summary = {}
    for path in paths:
        observations, actions = load_sequences(path)
        sequence_count = len(observations)
        valid_count = max(1, min(sequence_count - 1, int(round(sequence_count * valid_fraction))))
        indices = rng.permutation(sequence_count)
        valid_indices = indices[:valid_count]
        train_indices = indices[valid_count:]
        train_observations.append(observations[train_indices].reshape(-1, 2460))
        train_actions.append(actions[train_indices].reshape(-1, 28))
        valid_observations.append(observations[valid_indices].reshape(-1, 2460))
        valid_actions.append(actions[valid_indices].reshape(-1, 28))
        split_summary[path.stem] = {
            "total_sequences": sequence_count,
            "train_sequences": len(train_indices),
            "valid_sequences": len(valid_indices),
        }
    return (
        torch.from_numpy(np.concatenate(train_observations)),
        torch.from_numpy(np.concatenate(train_actions)),
        torch.from_numpy(np.concatenate(valid_observations)),
        torch.from_numpy(np.concatenate(valid_actions)),
        split_summary,
    )


class DexRepBC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_std = nn.Parameter(np.log(0.8) * torch.ones(28))
        self.bn_pnl = nn.BatchNorm1d(1280)
        self.dexrep_sensor_enc = nn.Linear(1080, 128)
        self.dexrep_pointL_enc = nn.Linear(1280, 128)
        self.state_enc = nn.Linear(100, 128)
        self.actor = nn.Sequential(
            nn.Linear(384, 1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.ELU(),
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 28),
        )
        self.critic = nn.Sequential(
            nn.Linear(384, 1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.ELU(),
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for encoder in (self.state_enc, self.dexrep_sensor_enc, self.dexrep_pointL_enc):
            nn.init.orthogonal_(encoder.weight, gain=np.sqrt(2))
        actor_scales = [np.sqrt(2)] * 4 + [0.01]
        critic_scales = [np.sqrt(2)] * 4 + [1.0]
        for layer, gain in zip((m for m in self.actor if isinstance(m, nn.Linear)), actor_scales):
            nn.init.orthogonal_(layer.weight, gain=gain)
        for layer, gain in zip((m for m in self.critic if isinstance(m, nn.Linear)), critic_scales):
            nn.init.orthogonal_(layer.weight, gain=gain)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[:, :100]
        sensor = observations[:, 100:1180]
        point = observations[:, 1180:2460]
        state_embedding = self.state_enc(state)
        sensor_embedding = functional.normalize(self.dexrep_sensor_enc(sensor), dim=-1)
        point_embedding = functional.normalize(self.dexrep_pointL_enc(self.bn_pnl(point)), dim=-1)
        return self.actor(torch.cat([state_embedding, sensor_embedding, point_embedding], dim=1))


def bc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    wrist = functional.mse_loss(prediction[:, :3], target[:, :3])
    orientation = functional.mse_loss(prediction[:, 3:6], target[:, 3:6])
    finger = functional.mse_loss(prediction[:, 6:], target[:, 6:])
    return 2 * wrist + orientation + finger + functional.l1_loss(prediction, target)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for observations, actions in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        total += float(bc_loss(model(observations), actions)) * observations.shape[0]
        count += observations.shape[0]
    return total / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--valid-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    return parser.parse_args()


def make_checkpoint(model: nn.Module, epoch: int, global_step: int, device: torch.device) -> dict:
    return {
        "state_dict": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()},
        "epoch": epoch,
        "global_step": global_step,
        "modern_torch_metadata": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    split_summary = None
    if args.data_dir is not None:
        data_paths = sorted(args.data_dir.glob("*.npy"))
        if not data_paths:
            raise RuntimeError("The data directory must contain .npy files")
        train_obs, train_actions, valid_obs, valid_actions, split_summary = load_sequence_split(
            data_paths, args.valid_fraction, args.seed,
        )
        train_paths = data_paths
        valid_paths = data_paths
    else:
        if args.train_dir is None or args.valid_dir is None:
            raise RuntimeError("Provide either --data-dir or both --train-dir and --valid-dir")
        train_paths = sorted(args.train_dir.glob("*.npy"))
        valid_paths = sorted(args.valid_dir.glob("*.npy"))
        if not train_paths or not valid_paths:
            raise RuntimeError("Both train and validation directories must contain .npy files")
        train_obs, train_actions = load_split(train_paths)
        valid_obs, valid_actions = load_split(valid_paths)
    train_loader = DataLoader(
        TensorDataset(train_obs, train_actions), batch_size=args.batch_size,
        shuffle=True, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        TensorDataset(valid_obs, valid_actions), batch_size=args.batch_size,
        shuffle=False, pin_memory=device.type == "cuda",
    )

    model = DexRepBC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.perf_counter()
    last_train_loss = float("nan")
    last_valid_loss = float("nan")
    best_valid_loss = float("inf")
    best_epoch = -1
    best_checkpoint = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for observations, actions in train_loader:
            observations = observations.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bc_loss(model(observations), actions)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * observations.shape[0]
            count += observations.shape[0]
        last_train_loss = total / count
        last_valid_loss = validate(model, valid_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": last_train_loss, "val_loss": last_valid_loss})
        if last_valid_loss < best_valid_loss:
            best_valid_loss = last_valid_loss
            best_epoch = epoch
            best_checkpoint = make_checkpoint(model, epoch, (epoch + 1) * len(train_loader), device)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={last_train_loss:.6f} "
            f"val_loss={last_valid_loss:.6f} best_val_loss={best_valid_loss:.6f}", flush=True,
        )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"early_stop epoch={epoch + 1} patience={args.patience}", flush=True)
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = make_checkpoint(model, history[-1]["epoch"] - 1, len(history) * len(train_loader), device)
    torch.save(checkpoint, args.output, _use_new_zipfile_serialization=False)
    best_output = args.best_output or args.output.with_name("best.ckpt")
    torch.save(best_checkpoint, best_output, _use_new_zipfile_serialization=False)
    metrics_output = args.metrics_output or args.output.with_name("metrics.json")
    result = {
        "device": str(device),
        "device_name": checkpoint["modern_torch_metadata"]["device"],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "train_files": len(train_paths),
        "valid_files": len(valid_paths),
        "train_frames": len(train_obs),
        "valid_frames": len(valid_obs),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "train_loss": last_train_loss,
        "val_loss": last_valid_loss,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_valid_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "best_checkpoint": str(best_output),
        "split_summary": split_summary,
    }
    metrics_output.write_text(json.dumps({"result": result, "history": history}, indent=2, sort_keys=True))
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
