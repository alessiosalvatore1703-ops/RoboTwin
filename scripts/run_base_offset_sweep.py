"""Sweep base-pose offsets over tasks, all conditions of a task in one process on one sim and one policy client.

Conditions: the zero-offset baseline, then each axis alone (x, y, yaw) at signed levels, then optionally a
joint random draw (Mobi-pi's pose-noise protocol). All share --seed, so conditions are paired by episode
seed. Per-episode rows land in <out_dir>/<policy>/<task>/<condition>.jsonl; a condition whose log already
holds --test_num rows is skipped, an incomplete one is rerun from scratch.

An experiment is a YAML config (--config, see env_cfg/base_offset/pose_noise.yml): with `levels` every level is
one joint-random-noise condition; without it the axis-sweep flags below build the conditions. CLI flags override
config keys. Unknown config keys are rejected, a silently ignored key would be a config typo.

Process startup (imports, renderer, sim and planner build) and the policy-model load dominate a short
condition, so they are paid once per task, not once per condition. Evals are deliberately serial: on a 16 GB
GPU (RTX 5080, X_VLA, lift_pot) 3 parallel workers filled GPU memory and the planner failed to build, and
2 workers peaked at 15.8 GB, were no faster per episode than 1 and wedged the renderer in one condition.

Within an axis-sign group levels run by increasing |offset|; with --early_stop the larger ones are skipped
once a smaller one succeeds 0 times. --server_cmd (with a {port} placeholder) starts and stops the policy
server; without it an already-running server at --port is used.
"""
import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
for path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "scripts", ROBOTWIN_ROOT / "description" / "utils"):
    sys.path.insert(0, str(path))


def level_groups(args):
    """One group per configured noise level; a zero level is the fixed zero-offset ID baseline."""
    groups = {}
    for level in args.levels:
        if level["xy_m"] == 0 and level["yaw_deg"] == 0:
            knobs = {"base_dx": 0.0, "base_dy": 0.0, "base_dyaw": 0.0}
        else:
            knobs = {"base_noise_xy_m": level["xy_m"], "base_noise_yaw_rad": math.radians(level["yaw_deg"]),
                     "noise_seed": args.noise_seed}
        groups[level["name"]] = [(level["name"], knobs)]
    return groups


def condition_groups(args):
    """{group: [(condition_name, knobs)]}, each group ordered by increasing offset magnitude."""
    if args.levels:
        return level_groups(args)
    groups = {"nominal": [("nominal", {"base_dx": 0.0, "base_dy": 0.0, "base_dyaw": 0.0})]}
    levels = {"x": ("base_dx", args.xy_levels_m, 1.0), "y": ("base_dy", args.xy_levels_m, 1.0),
              "yaw": ("base_dyaw", args.yaw_levels_deg, math.pi / 180)}
    for axis in args.axes:
        key, values, scale = levels[axis]
        for sign in (-1, 1):
            signed = sorted((v for v in values if v * sign > 0), key=abs)
            if signed:
                groups[f"{axis}{'+' if sign > 0 else '-'}"] = [(f"{axis}_{v:+g}", {key: v * scale}) for v in signed]
    if args.joint:
        groups["joint"] = [("joint", {"base_noise_xy_m": args.joint_xy_m,
                                      "base_noise_yaw_rad": math.radians(args.joint_yaw_deg),
                                      "noise_seed": args.noise_seed})]
    return groups


def parse_eval_args(args, task):
    """usr_args exactly as the eval script's own CLI would build them, so the harness sees a normal run."""
    import eval_policy_xpolicylab as eval_module
    info = {"ckpt_name": args.ckpt_name, "action_type": args.action_type,
            "render_each_action": args.render_each_action}
    argv = ["eval", "--task_name", task, "--env_cfg_type", args.env_cfg_type, "--policy_name", args.policy_name,
            "--port", str(args.port), "--seed", str(args.seed), "--test_num", str(args.test_num),
            "--task_config", args.task_config, "--additional_info", ",".join(f"{k}={v}" for k, v in info.items())]
    saved, sys.argv = sys.argv, argv
    try:
        return eval_module.parse_args()
    finally:
        sys.argv = saved


def run_task(args, task, groups):
    import eval_policy_xpolicylab as eval_module
    run_dir = args.out_dir / args.policy_name / task
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items() if k != "levels"}, "levels": args.levels,
         "conditions": [name for conds in groups.values() for name, _ in conds]}, indent=2) + "\n")

    base_usr_args = parse_eval_args(args, task)
    base_usr_args.setdefault("evaluation_id", f"robotwin-{task}-{args.policy_name}-sweep")
    base_usr_args.setdefault("action_case_id", f"{task}-case")
    task_args, _ = eval_module.load_task_args(base_usr_args)
    instruction_type = eval_module.resolve_eval_instruction(base_usr_args, task_args)
    task_args["eval_instruction"] = instruction_type
    base_usr_args["instruction_type"] = instruction_type
    base_usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    base_usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])
    task_env = eval_module.class_decorator(task)
    model_client = eval_module.build_policy_client(base_usr_args)
    try:
        for group, conditions in groups.items():
            for name, knobs in conditions:
                log = run_dir / f"{name}.jsonl"
                rows = len(log.read_text().splitlines()) if log.exists() else 0
                if rows < args.test_num:
                    log.unlink(missing_ok=True)
                    print(f"== {task}/{name}", flush=True)
                    eval_module.eval_remote_policy(
                        task, task_env, task_args, model_client, 100000 * (1 + args.seed),
                        usr_args={**base_usr_args, **knobs, "episode_log": str(log), "condition_name": name},
                        test_num=args.test_num, instruction_type=instruction_type)
                else:
                    print(f"skip {task}/{name}: {rows} rows", flush=True)
                successes = sum(json.loads(line)["status"] == "success" for line in log.read_text().splitlines())
                if successes == 0 and args.early_stop:
                    print(f"early stop {task}/{group}: 0 successes at {name}", flush=True)
                    break
    finally:
        eval_module.close_policy_client(model_client)


def main():
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--config", type=Path)
    config_path = preparser.parse_known_args()[0].config
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    levels = config.pop("levels", None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="experiment YAML; keys are the flags below plus `levels`")
    parser.add_argument("--task_names", nargs="+", default=["lift_pot", "open_laptop", "stack_blocks_two"])
    parser.add_argument("--policy_name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ckpt_name")
    parser.add_argument("--server_cmd", help="policy-server command with a {port} placeholder")
    parser.add_argument("--action_type", default="ee")
    parser.add_argument("--env_cfg_type", default="aloha_agilex")
    parser.add_argument("--task_config", default="demo_clean_novideo")
    parser.add_argument("--render_each_action", type=lambda v: v.lower() != "false", default=False,
                        help="render+send an observation after every chunk action (default: only per chunk)")
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axes", nargs="+", choices=("x", "y", "yaw"), default=["x", "y", "yaw"])
    parser.add_argument("--xy_levels_m", nargs="+", type=float, default=[-0.2, -0.1, 0.1, 0.2])
    parser.add_argument("--yaw_levels_deg", nargs="+", type=float, default=[-25, -10, 10, 25])
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--joint_xy_m", type=float, default=0.2)
    parser.add_argument("--joint_yaw_deg", type=float, default=25)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=ROBOTWIN_ROOT / "eval_result" / "base_offset")
    parser.add_argument("--dry_run", action="store_true")
    unknown = set(config) - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f"unknown config keys {sorted(unknown)}")
    parser.set_defaults(**config)
    args = parser.parse_args()
    for required in ("policy_name", "port", "ckpt_name"):
        if getattr(args, required) is None:
            parser.error(f"{required} is required (flag or config key)")
    args.levels = levels
    args.out_dir = (ROBOTWIN_ROOT / args.out_dir).resolve()

    groups = condition_groups(args)
    if args.dry_run:
        for task in args.task_names:
            for group, conditions in groups.items():
                print(task, group, [name for name, _ in conditions])
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        shutil.copy(args.config, args.out_dir / args.config.name)
    server = None
    if args.server_cmd:
        server = subprocess.Popen(shlex.split(args.server_cmd.format(port=args.port)), cwd=ROBOTWIN_ROOT,
                                  stdout=open(args.out_dir / f"server_{args.port}.log", "w"),
                                  stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for task in args.task_names:
            run_task(args, task, groups)
    finally:
        if server is not None:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
