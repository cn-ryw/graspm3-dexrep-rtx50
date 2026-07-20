#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --checkpoint FILE --object-code CODE --method NAME --seed N --epoch LABEL --output-dir DIR [--export-visuals]" >&2
}

checkpoint=""
object_code=""
method=""
seed=""
epoch=""
output_dir=""
export_visuals=0
while (($#)); do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --object-code) object_code="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --export-visuals) export_visuals=1; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${checkpoint}" || -z "${object_code}" || -z "${method}" || -z "${seed}" || -z "${epoch}" || -z "${output_dir}" ]]; then
  usage
  exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
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

checkpoint="$(realpath "${checkpoint}")"
mkdir -p "${extension_cache}" "${output_dir}"
output_dir="$(realpath "${output_dir}")"

docker run --rm --gpus all --network none --ipc host \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/scripts:/opt/isaacgym/python:/workspace/project:/workspace/project/dexgrasp:/workspace/project/pytorch_kinematics \
  --env TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
  --env DEXGRASP_INFER_OBJECTS="${object_code}" \
  --env DEXGRASP_INFER_BATCH_SIZE=1 \
  --env DEXGRASP_CHECKPOINT=/workspace/checkpoint.ckpt \
  --env DEXGRASP_ROLLOUT_OUTPUT=/output \
  --env DEXGRASP_METHOD="${method}" \
  --env DEXGRASP_EVAL_SEED="${seed}" \
  --env DEXGRASP_EPOCH="${epoch}" \
  --env DEXGRASP_EXPORT_VISUALS="${export_visuals}" \
  --mount "type=bind,src=${isaacgym_dir},dst=/opt/isaacgym,readonly" \
  --mount "type=bind,src=${repo_dir},dst=/workspace/project" \
  --mount "type=bind,src=${control_dir}/scripts,dst=/scripts,readonly" \
  --mount "type=bind,src=${graspm3_dir}/dataset,dst=/workspace/project/dexgrasp/dataset/train,readonly" \
  --mount "type=bind,src=${graspm3_dir}/meshdata,dst=/workspace/project/assets/meshdata,readonly" \
  --mount "type=bind,src=${extension_cache},dst=/opt/torch_extensions" \
  --mount "type=bind,src=${checkpoint},dst=/workspace/checkpoint.ckpt,readonly" \
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
