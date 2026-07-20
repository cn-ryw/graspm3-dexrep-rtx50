#!/usr/bin/env bash
set -euo pipefail

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_root="${control_dir}/experiments/final-fair-training"
evaluation_root="${control_dir}/experiments/final-fair-evaluation"
max_jobs="${FINAL_EVAL_JOBS:-1}"
active_jobs=0
if ((max_jobs < 1)); then
  echo "FINAL_EVAL_JOBS must be at least 1" >&2
  exit 2
fi

declare -A object_codes=(
  [jar]="core-jar-8ec888ab36f1c5635afa616678601602"
  [cellphone]="sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6"
  [usb_stick]="sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4"
)

run_eval() {
  local checkpoint="$1" object_code="$2" method="$3" seed="$4" epoch="$5" output_dir="$6" export_visuals="$7"
  if [[ -s "${output_dir}/rollout_metrics.json" ]]; then
    echo "skip completed ${output_dir#${evaluation_root}/}"
    return
  fi
  local visual_arg=()
  if [[ "${export_visuals}" == 1 ]]; then
    visual_arg=(--export-visuals)
  fi
  "${control_dir}/scripts/run_final_eval.sh" \
    --checkpoint "${checkpoint}" \
    --object-code "${object_code}" \
    --method "${method}" \
    --seed "${seed}" \
    --epoch "${epoch}" \
    --output-dir "${output_dir}" \
    "${visual_arg[@]}"
}

schedule_eval() {
  run_eval "$@" &
  active_jobs=$((active_jobs + 1))
  if ((active_jobs >= max_jobs)); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
}

for method in baseline wrist4; do
  for object_name in jar cellphone usb_stick; do
    object_code="${object_codes[${object_name}]}"
    for seed in 0 1 2; do
      train_dir="${training_root}/${method}/${object_name}/seed_${seed}"
      schedule_eval \
        "${train_dir}/best.ckpt" "${object_code}" "${method}" "${seed}" best \
        "${evaluation_root}/best/${method}/${object_name}/seed_${seed}" 1
    done

    train_dir="${training_root}/${method}/${object_name}/seed_0"
    for milestone in 1 5 10 25 50 100; do
      padded="$(printf '%03d' "${milestone}")"
      schedule_eval \
        "${train_dir}/milestones/epoch_${padded}.ckpt" "${object_code}" "${method}" 0 "${milestone}" \
        "${evaluation_root}/milestones/${method}/${object_name}/epoch_${padded}" 0
    done
  done
done

while ((active_jobs > 0)); do
  wait -n
  active_jobs=$((active_jobs - 1))
done
