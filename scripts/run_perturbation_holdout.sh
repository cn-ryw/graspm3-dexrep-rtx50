#!/usr/bin/env bash
set -euo pipefail

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_root="${control_dir}/experiments/final-fair-training"
output_root="${control_dir}/experiments/perturbation-holdout"
manifest="${control_dir}/configs/perturbation_holdout_v1.json"
jobs="${HOLDOUT_EVAL_JOBS:-2}"

if [[ ! -s "${manifest}" ]]; then
  echo "Frozen holdout manifest not found: ${manifest}" >&2
  exit 2
fi

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

run_member() {
  local object_name="$1"
  local member="$2"
  local method="${selected_methods[${object_name}]}"
  local output_dir="${output_root}/member_seed_${member}/${object_name}"
  if [[ -s "${output_dir}/rollout_metrics.json" ]]; then
    echo "skip completed holdout member seed=${member} object=${object_name}"
    return
  fi
  "${control_dir}/scripts/run_final_eval.sh" \
    --checkpoint "${training_root}/${method}/${object_name}/seed_${member}/best.ckpt" \
    --object-code "${object_codes[${object_name}]}" \
    --method "holdout_${method}_member" \
    --seed "${member}" \
    --epoch best \
    --output-dir "${output_dir}" \
    --holdout-manifest "${manifest}"
}

run_ensemble() {
  local object_name="$1"
  local method="${selected_methods[${object_name}]}"
  local checkpoints=""
  local member
  for member in 0 1 2; do
    [[ -z "${checkpoints}" ]] || checkpoints+=","
    checkpoints+="${training_root}/${method}/${object_name}/seed_${member}/best.ckpt"
  done
  local output_dir="${output_root}/ensemble/${object_name}"
  if [[ -s "${output_dir}/rollout_metrics.json" ]]; then
    echo "skip completed holdout ensemble object=${object_name}"
    return
  fi
  "${control_dir}/scripts/run_final_eval.sh" \
    --checkpoints "${checkpoints}" \
    --object-code "${object_codes[${object_name}]}" \
    --method "holdout_adaptive_ensemble_${method}" \
    --seed 0 \
    --epoch ensemble_best \
    --output-dir "${output_dir}" \
    --holdout-manifest "${manifest}" \
    --export-visuals
}

tasks=()
for object_name in jar cellphone usb_stick; do
  for member in 0 1 2; do
    tasks+=("member:${object_name}:${member}")
  done
  tasks+=("ensemble:${object_name}")
done

active_jobs=0
for task in "${tasks[@]}"; do
  IFS=: read -r kind object_name member <<< "${task}"
  if [[ "${kind}" == member ]]; then
    run_member "${object_name}" "${member}" &
  else
    run_ensemble "${object_name}" &
  fi
  ((active_jobs += 1))
  if ((active_jobs >= jobs)); then
    wait -n
    ((active_jobs -= 1))
  fi
done
wait
