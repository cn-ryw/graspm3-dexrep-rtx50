#!/usr/bin/env bash
set -euo pipefail

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${control_dir}/experiments/gate-d-three-object-dexrep-cache"
output_root="${control_dir}/experiments/pose-stratified-hard-resampling-training"
sampler_manifest="${control_dir}/configs/pose_stratified_hard_sampler_v1.json"
log_dir="${control_dir}/reports/logs"
image_name="vt-dexterity-pytorch-smoke:torch2.7.1-cu128"

[[ -s "${sampler_manifest}" ]] || { echo "Sampler manifest not found: ${sampler_manifest}" >&2; exit 2; }

declare -A object_codes=(
  [jar]="core-jar-8ec888ab36f1c5635afa616678601602"
  [cellphone]="sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6"
  [usb_stick]="sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4"
)

mkdir -p "${output_root}" "${log_dir}"
for object_name in jar cellphone usb_stick; do
  object_code="${object_codes[${object_name}]}"
  [[ -s "${data_dir}/${object_code}.npy" ]] || { echo "DexRep cache not found for ${object_name}" >&2; exit 2; }
  for seed in 0 1 2; do
    output_dir="${output_root}/${object_name}/seed_${seed}"
    if [[ -s "${output_dir}/metrics.json" && -s "${output_dir}/best.ckpt" ]]; then
      echo "skip completed pose-hard/${object_name}/seed_${seed}"
      continue
    fi
    mkdir -p "${output_dir}"
    log_file="${log_dir}/pose-hard-train-${object_name}-seed${seed}.log"
    echo "train pose-hard/${object_name}/seed_${seed}"
    docker run --rm --gpus all --network none --ipc host \
      --env PYTHONDONTWRITEBYTECODE=1 \
      --mount "type=bind,src=${data_dir},dst=/dataset,readonly" \
      --mount "type=bind,src=${control_dir}/scripts,dst=/scripts,readonly" \
      --mount "type=bind,src=${sampler_manifest},dst=/config/sampler.json,readonly" \
      --mount "type=bind,src=${output_dir},dst=/output" \
      "${image_name}" \
      python3 -u /scripts/train_bc_modern_torch.py \
        --data-dir /dataset \
        --object-code "${object_code}" \
        --valid-fraction 0.2 \
        --split-seed 0 \
        --seed "${seed}" \
        --wrist-weight 2 \
        --sampler-manifest /config/sampler.json \
        --epochs 200 \
        --patience 30 \
        --milestone-epochs 1,5,10,25,50,100 \
        --batch-size 256 \
        --lr 2e-4 \
        --output /output/last.ckpt \
        --best-output /output/best.ckpt \
        --metrics-output /output/metrics.json \
      2>&1 | tee "${log_file}"
  done
done
