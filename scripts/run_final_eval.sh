#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 (--checkpoint FILE | --checkpoints FILE,FILE,...) --object-code CODE --method NAME --seed N --epoch LABEL --output-dir DIR [--holdout-manifest FILE] [--export-visuals]" >&2
}

checkpoint=""
checkpoints=""
object_code=""
method=""
seed=""
epoch=""
output_dir=""
holdout_manifest=""
export_visuals=0
while (($#)); do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --checkpoints) checkpoints="$2"; shift 2 ;;
    --object-code) object_code="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --holdout-manifest) holdout_manifest="$2"; shift 2 ;;
    --export-visuals) export_visuals=1; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${object_code}" || -z "${method}" || -z "${seed}" || -z "${epoch}" || -z "${output_dir}" ]]; then
  usage
  exit 2
fi
if [[ -n "${checkpoint}" && -n "${checkpoints}" ]] || [[ -z "${checkpoint}" && -z "${checkpoints}" ]]; then
  usage
  exit 2
fi
if [[ -n "${checkpoint}" ]]; then
  checkpoints="${checkpoint}"
fi

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${DEXGRASP_REPO_DIR:-${control_dir}/../DexGraspMotionChallenge2025}"
isaacgym_dir="${ISAACGYM_DIR:-${control_dir}/../../software/isaacgym}"
graspm3_dir="${GRASPM3_DIR:-${control_dir}/../../datasets/GraspM3}"
extension_cache="${TORCH_EXTENSION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/vt-dexterity/gate-c-torch-extensions}"
image_name="vt-dexterity-gate-c-demo:official-f41dc7d"

for required_dir in "${repo_dir}" "${isaacgym_dir}" "${graspm3_dir}/dataset" "${graspm3_dir}/meshdata"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory not found: ${required_dir}" >&2
    exit 2
  fi
done

IFS=',' read -r -a checkpoint_files <<< "${checkpoints}"
checkpoint_mounts=()
runtime_checkpoints=()
for index in "${!checkpoint_files[@]}"; do
  checkpoint_file="${checkpoint_files[${index}]}"
  if [[ ! -f "${checkpoint_file}" ]]; then
    echo "Checkpoint not found: ${checkpoint_file}" >&2
    exit 2
  fi
  checkpoint_file="$(realpath "${checkpoint_file}")"
  container_checkpoint="/workspace/checkpoints/member_${index}.ckpt"
  checkpoint_mounts+=(--mount "type=bind,src=${checkpoint_file},dst=${container_checkpoint},readonly")
  runtime_checkpoints+=("${container_checkpoint}")
done
runtime_checkpoint_csv="$(IFS=,; echo "${runtime_checkpoints[*]}")"
mkdir -p "${extension_cache}" "${output_dir}"
output_dir="$(realpath "${output_dir}")"

holdout_args=()
if [[ -n "${holdout_manifest}" ]]; then
  if [[ ! -f "${holdout_manifest}" ]]; then
    echo "Holdout manifest not found: ${holdout_manifest}" >&2
    exit 2
  fi
  holdout_manifest="$(realpath "${holdout_manifest}")"
  holdout_args+=(
    --env DEXGRASP_HOLDOUT_MANIFEST=/workspace/holdout/manifest.json
    --mount "type=bind,src=${holdout_manifest},dst=/workspace/holdout/manifest.json,readonly"
  )
fi

docker run --rm --gpus all --network none --ipc host \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/scripts:/opt/isaacgym/python:/workspace/project:/workspace/project/dexgrasp:/workspace/project/pytorch_kinematics \
  --env TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
  --env DEXGRASP_INFER_OBJECTS="${object_code}" \
  --env DEXGRASP_INFER_BATCH_SIZE=1 \
  --env DEXGRASP_CHECKPOINTS="${runtime_checkpoint_csv}" \
  --env DEXGRASP_ROLLOUT_OUTPUT=/output \
  --env DEXGRASP_METHOD="${method}" \
  --env DEXGRASP_EVAL_SEED="${seed}" \
  --env DEXGRASP_EPOCH="${epoch}" \
  --env DEXGRASP_EXPORT_VISUALS="${export_visuals}" \
  "${holdout_args[@]}" \
  --mount "type=bind,src=${isaacgym_dir},dst=/opt/isaacgym,readonly" \
  --mount "type=bind,src=${repo_dir},dst=/workspace/project" \
  --mount "type=bind,src=${control_dir}/scripts,dst=/scripts,readonly" \
  --mount "type=bind,src=${graspm3_dir}/dataset,dst=/workspace/project/dexgrasp/dataset/train,readonly" \
  --mount "type=bind,src=${graspm3_dir}/meshdata,dst=/workspace/project/assets/meshdata,readonly" \
  --mount "type=bind,src=${extension_cache},dst=/opt/torch_extensions" \
  "${checkpoint_mounts[@]}" \
  --mount "type=bind,src=${output_dir},dst=/output" \
  --workdir /workspace/project/dexgrasp \
  "${image_name}" \
  python3.8 -u bc_env_infer.py \
    --task ShadowHandGraspDexRepIjrr \
    --algo ppo1 \
    --seed "${seed}" \
    --rl_device cpu \
    --sim_device cuda:0 \
    --pipeline cpu \
    --headless \
  2>&1 | tee "${output_dir}/eval.log"

test -s "${output_dir}/rollout_metrics.json"
