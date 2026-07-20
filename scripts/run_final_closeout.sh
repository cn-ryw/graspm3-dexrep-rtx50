#!/usr/bin/env bash
set -euo pipefail

control_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${control_dir}/scripts/run_final_training_matrix.sh"
FINAL_EVAL_JOBS="${FINAL_EVAL_JOBS:-2}" \
  "${control_dir}/scripts/run_final_evaluation_matrix.sh"
python "${control_dir}/scripts/build_final_results.py"
python "${control_dir}/scripts/render_final_cases.py"
python "${control_dir}/scripts/build_final_report.py"
python "${control_dir}/scripts/verify_final_deliverables.py"
