#!/usr/bin/env bash
# Watch a pretrained policy roll out live in the SAPIEN viewer.
# Usage: ./watch_rollout.sh <X_VLA|Pi_05> <task> [seed] [num_episodes] [task_config]
#   e.g. ./watch_rollout.sh Pi_05 lift_pot 0 3
# task_config defaults to demo_clean_viewer (demo_clean with render_freq: 5);
# lower render_freq in that file for a smoother view, raise it for faster rollouts.
set -euo pipefail

policy=${1:?policy: X_VLA or Pi_05}
task=${2:?task name}
seed=${3:-0}
episodes=${4:-1}
task_config=${5:-demo_clean_viewer}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DISPLAY="${DISPLAY:-:1}"
export CUDA_HOME="$ROOT/.cuda"
export PATH="$ROOT/.cuda/bin:$PATH"
export PYTHONWARNINGS=ignore::UserWarning
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

port=$(bash XPolicyLab/utils/get_free_port.sh)
log="$(mktemp -t policy_server.XXXXXX.log)"

case "$policy" in
  X_VLA)
    ckpt=RoboTwin-cotrain-aloha_agilex-ee-0; action=ee
    server_py="$ROOT/.venv_xvla/bin/python"; server_cwd="$ROOT"
    extra=() ;;
  Pi_05)
    ckpt=RoboTwin-lerobot_v30-aloha_agilex-joint-0; action=joint
    server_py="$ROOT/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
    server_cwd="$ROOT/XPolicyLab/policy/Pi_05"
    # deploy.yml defaults (59999 / arx_x5_sim) do not match the downloaded checkpoint
    extra=(checkpoint_num=29999 repo_id=RoboTwin_lerobot_v30)
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ;;
  *) echo "unknown policy: $policy (X_VLA or Pi_05)" >&2; exit 1 ;;
esac

(cd "$server_cwd" && exec "$server_py" "$ROOT/XPolicyLab/setup_policy_server.py" \
  --config_path "$ROOT/XPolicyLab/policy/$policy/deploy.yml" --overrides \
  port="$port" host=localhost bench_name=RoboTwin task_name="$task" ckpt_name="$ckpt" \
  env_cfg_type=aloha_agilex seed="$seed" policy_name="$policy" action_type="$action" \
  action_dim=14 "${extra[@]}") >"$log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
echo "[watch] $policy server pid $server_pid on port $port (log: $log)"

# The eval client retries its connection while the model loads.
.venv/bin/python scripts/eval_policy_xpolicylab.py \
  --task_name "$task" --env_cfg_type aloha_agilex --policy_name "$policy" \
  --port "$port" --seed "$seed" --test_num "$episodes" --task_config "$task_config" \
  --additional_info "ckpt_name=$ckpt,action_type=$action" 2>&1 \
  | grep --line-buffered -av "OIDN\|Warn\|sparse_coo"
