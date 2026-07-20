#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --manifest FILE --label development|test" >&2
}

manifest=""
label=""
while (($#)); do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "${manifest}" && -n "${label}" ]] || { usage; exit 2; }
[[ "${label}" == development || "${label}" == test ]] || { usage; exit 2; }
[[ -s "${manifest}" ]] || { echo "Evaluation manifest not found: ${manifest}" >&2; exit 2; }

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training_root="${control_dir}/experiments/pose-stratified-hard-resampling-training"
output_root="${control_dir}/experiments/pose-stratified-hard-resampling-${label}"
jobs="${POSE_RESAMPLING_EVAL_JOBS:-2}"

declare -A object_codes=(
  [jar]="core-jar-8ec888ab36f1c5635afa616678601602"
  [cellphone]="sem-CellPhone-4e2f684b3cebdbc344f470fdd42caac6"
  [usb_stick]="sem-USBStick-e2e75212fefa0ecf30e9a79a761377a4"
)

run_one() {
  local object_name="$1"
  local seed="$2"
  local output_dir="${output_root}/member_seed_${seed}/${object_name}"
  if [[ -s "${output_dir}/rollout_metrics.json" ]]; then
    echo "skip completed pose-hard ${label} seed=${seed} object=${object_name}"
    return
  fi
  "${control_dir}/scripts/run_final_eval.sh" \
    --checkpoint "${training_root}/${object_name}/seed_${seed}/best.ckpt" \
    --object-code "${object_codes[${object_name}]}" \
    --method "pose_stratified_hard_resampling_${label}" \
    --seed "${seed}" \
    --epoch best \
    --output-dir "${output_dir}" \
    --holdout-manifest "${manifest}"
}

active_jobs=0
for object_name in jar cellphone usb_stick; do
  for seed in 0 1 2; do
    run_one "${object_name}" "${seed}" &
    ((active_jobs += 1))
    if ((active_jobs >= jobs)); then
      wait -n
      ((active_jobs -= 1))
    fi
  done
done
wait
