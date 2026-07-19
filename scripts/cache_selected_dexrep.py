#!/usr/bin/env python3
"""Cache DexRep observations for selected raw GraspM3 objects.

This script runs inside the legacy Isaac Gym container. Source trajectories are
read-only; processed dictionaries are written to a separate output directory.
"""

import json
import os
from pathlib import Path

import numpy as np

from data_preprocess import worker_run
from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from utils.process_marl import get_AgentIndex


def main():
    set_np_formatting()
    input_dir = Path(os.environ["DEXGRASP_CACHE_INPUT"])
    output_dir = Path(os.environ["DEXGRASP_CACHE_OUTPUT"])
    object_codes = [
        item for item in os.environ["DEXGRASP_CACHE_OBJECTS"].split(",") if item
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    args = get_args()
    args.seed = int(os.environ.get("DEXGRASP_SEED", "0"))
    args.rl_device = os.environ.get("DEXGRASP_RL_DEVICE", args.rl_device)
    args.sim_device = os.environ.get("DEXGRASP_SIM_DEVICE", args.sim_device)
    args.headless = True
    cfg, cfg_train, _ = load_cfg(args)
    if args.num_objs != -1:
        cfg["env"]["num_objs"] = args.num_objs
    cfg["env"]["seq_start_rot_uniform"] = False
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(args.seed, cfg_train.get("torch_deterministic", False))
    agent_index = get_AgentIndex(cfg)

    results = []
    for proc_id, object_code in enumerate(object_codes):
        source = input_dir / (object_code + ".npy")
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = np.load(str(source), allow_pickle=True).item()
        payload["obj_code"] = object_code
        processed = worker_run(
            args, proc_id, [payload], cfg, cfg_train, sim_params, agent_index,
            str(output_dir), save_npy=True,
        )[0]
        output = output_dir / (object_code + ".npy")
        results.append({
            "object_code": object_code,
            "raw_sequences": int(len(payload["grasp_seqs"])),
            "cached_sequences": int(len(processed["grasp_seqs"])),
            "frames": int(processed["obs"].shape[1]) if len(processed["obs"]) else 0,
            "output": str(output),
            "output_bytes": output.stat().st_size,
        })

    print("CACHE_RESULT_JSON=" + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
