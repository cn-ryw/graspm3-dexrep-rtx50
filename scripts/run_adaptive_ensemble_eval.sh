#!/usr/bin/env bash
set -euo pipefail

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_root="${control_dir}/experiments/final-fair-training"
output_root="${control_dir}/experiments/adaptive-ensemble-evaluation"

declare -A object_codes=(
  [jar]="core-jar-8ec888ab36f1c5635afa616678601602"
  [cellphone]="sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6"
  [usb_stick]="sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4"
)
declare -A selected_methods=(
  [jar]="wrist4"
  [cellphone]="wrist4"
  [usb_stick]="baseline"
)

for object_name in jar cellphone usb_stick; do
  method="${selected_methods[${object_name}]}"
  checkpoint_csv=""
  for seed in 0 1 2; do
    checkpoint="${training_root}/${method}/${object_name}/seed_${seed}/best.ckpt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Checkpoint not found: ${checkpoint}" >&2
      exit 2
    fi
    if [[ -n "${checkpoint_csv}" ]]; then
      checkpoint_csv+=","
    fi
    checkpoint_csv+="${checkpoint}"
  done
  output_dir="${output_root}/${object_name}"
  if [[ -s "${output_dir}/rollout_metrics.json" ]]; then
    echo "skip completed adaptive ensemble ${object_name}"
    continue
  fi
  "${control_dir}/scripts/run_final_eval.sh" \
    --checkpoints "${checkpoint_csv}" \
    --object-code "${object_codes[${object_name}]}" \
    --method "adaptive_ensemble_${method}" \
    --seed 0 \
    --epoch ensemble_best \
    --output-dir "${output_dir}" \
    --export-visuals
done
