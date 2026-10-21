"""Map which base offsets are geometrically valid (no robot penetration of table/wall/objects) without any policy.

Sets up one seed of --task_name per offset and applies the same penetration check the eval uses, so the
printed boundary is exactly what the eval will mark `infeasible`. No rendering, no policy server.
"""
import argparse
import math
import sys
from pathlib import Path

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
for path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "scripts", ROBOTWIN_ROOT / "description" / "utils"):
    sys.path.insert(0, str(path))

import eval_policy_xpolicylab as eval_module  # noqa: E402
from envs.utils.base_offset import robot_penetrations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", default="lift_pot")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--xy_levels_m", nargs="+", type=float, default=[-.8, -.6, -.4, -.2, .2, .4, .6, .8])
    parser.add_argument("--yaw_levels_deg", nargs="+", type=float, default=[-90, -60, -30, 30, 60, 90])
    args = parser.parse_args()

    task_args, _ = eval_module.load_task_args(
        {"task_name": args.task_name, "policy_name": "none", "task_config": "demo_clean"})
    task_args.update(eval_mode=True, render_freq=0)
    env = eval_module.class_decorator(args.task_name)
    offsets = [(f"x {v:+.2f} m", {"dx": v, "dy": 0.0, "dyaw": 0.0}) for v in args.xy_levels_m]
    offsets += [(f"y {v:+.2f} m", {"dx": 0.0, "dy": v, "dyaw": 0.0}) for v in args.xy_levels_m]
    offsets += [(f"yaw {v:+.0f} deg", {"dx": 0.0, "dy": 0.0, "dyaw": math.radians(v)}) for v in args.yaw_levels_deg]
    for label, offset in [("nominal", {"dx": 0.0, "dy": 0.0, "dyaw": 0.0})] + offsets:
        try:
            env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True,
                           **{**task_args, "base_pose_offset": offset})
            penetrations = robot_penetrations(env.scene, env.robot.left_entity.get_links())
            env.close_env()
            verdict = "valid" if not penetrations else "INVALID " + ", ".join(
                f"{l}/{o} {d:.3f}m" for l, o, d in penetrations[:2])
        except Exception as error:
            eval_module.safe_close_env(env)
            verdict = f"INVALID setup raised {type(error).__name__}: {error}"
        print(f"{label:14s} {verdict}", flush=True)


if __name__ == "__main__":
    main()
