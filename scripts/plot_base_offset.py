"""Plot base-offset sweeps in Mobi-pi's pose-noise style: heading arrows per episode, plus per-axis success curves.

Colours follow plot_pose_noise.py: green = zero-offset baseline, blue = success, red = failure,
red x = infeasible (the sim could not set the offset up). Takes run dirs written by run_base_offset_sweep.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

AXES = (("dx", "x offset [m]", 1.0), ("dy", "y offset [m]", 1.0), ("dyaw", "yaw offset [deg]", 180 / np.pi))


def load_records(run_dir):
    records = [json.loads(line) for path in sorted(run_dir.glob("*.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"No episode records in {run_dir}")
    return records


def nominal_pose(records):
    """World base pose at zero offset, recovered from any measured row (infeasible rows carry no pose)."""
    row = next(r for r in records if "x" in r)
    return row["x"] - row["dx"], row["y"] - row["dy"], row["yaw"] - row["dyaw"]


def color_for(record):
    return "green" if record["condition"] == "id" else "blue" if record["status"] == "success" else "red"


def wilson(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def plot_poses(ax, records, id_records=()):
    """`id_records` are the nominal-level rollouts, drawn green on top of a noisy level's arrows so each
    panel shows the training pose next to the perturbed ones; all of them share the one nominal pose."""
    x0, y0, yaw0 = nominal_pose(records)
    records = [r for r in records if r["condition"] != "id"] + list(id_records) + \
        [r for r in records if r["condition"] == "id"]
    for record in records:
        x, y, yaw = x0 + record["dx"], y0 + record["dy"], yaw0 + record["dyaw"]
        color = color_for(record)
        if record["status"] == "infeasible":
            ax.scatter(x, y, marker="x", color=color, s=35)
        ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, angles="xy", scale_units="xy", scale=8,
                  width=.005, alpha=.6)
    noisy = [r for r in records if r["condition"] != "id"]
    count = {status: sum(r["status"] == status for r in noisy) for status in ("success", "failure", "infeasible")}
    ax.legend(handles=[
        Line2D([], [], color="green", label=f"ID pose ({len(records) - len(noisy)} rollouts)"),
        Line2D([], [], color="blue", label=f"Noisy success ({count['success']})"),
        Line2D([], [], color="red", label=f"Noisy failure ({count['failure']})"),
        Line2D([], [], color="red", marker="x", linestyle="none", label=f"Noisy infeasible ({count['infeasible']})"),
    ], fontsize=8)
    ax.set_title(f"Noisy success {count['success']}/{len(noisy)}\nBase starting poses (world frame)")
    # Quiver autoscaling considers arrow origins but can clip heading tips (same fix as plot_pose_noise.py).
    poses = np.asarray([(x0 + r["dx"], y0 + r["dy"], yaw0 + r["dyaw"]) for r in records])
    tips = poses[:, :2] + np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2]))) / 8
    points = np.vstack((poses[:, :2], tips))
    margin = np.maximum(.02, .1 * np.ptp(points, axis=0))
    ax.set_xlim(points[:, 0].min() - margin[0], points[:, 0].max() + margin[0])
    ax.set_ylim(points[:, 1].min() - margin[1], points[:, 1].max() + margin[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=.2)


def plot_axis_curve(ax, records, key, label, scale):
    """Success rate against one axis, using only rows where the other two axes are at zero."""
    others = [k for k, _, _ in AXES if k != key]
    rows = [r for r in records if all(r[k] == 0 for k in others)]
    levels = sorted({r[key] for r in rows})
    rates, lows, highs = [], [], []
    for level in levels:
        outcomes = [r["status"] == "success" for r in rows if r[key] == level]
        low, high = wilson(sum(outcomes), len(outcomes))
        rates.append(np.mean(outcomes))
        lows.append(low)
        highs.append(high)
    levels = np.asarray(levels) * scale
    ax.plot(levels, rates, color="blue", marker="o")
    ax.fill_between(levels, lows, highs, color="blue", alpha=.15)
    baseline = [r["status"] == "success" for r in rows if r[key] == 0]
    if baseline:
        ax.axhline(np.mean(baseline), color="green", linestyle="--", label="ID baseline")
        ax.legend(fontsize=8)
    ax.set_ylim(-.02, 1.02)
    ax.set_xlabel(label)
    ax.set_ylabel("success rate (95% Wilson)")
    ax.set_title(f"{label.split(' ')[0]} sweep")
    ax.grid(alpha=.2)


def plot_level_summary(ax, records, levels):
    """Success rate among feasible rollouts per noise level, 95% Wilson interval, ID baseline dashed green."""
    rates, lows, highs, labels = [], [], [], []
    for level in levels:
        feasible = [r for r in records if r.get("level") == level["name"] and r["status"] != "infeasible"]
        successes = sum(r["status"] == "success" for r in feasible)
        low, high = wilson(successes, len(feasible))
        rates.append(successes / len(feasible) if feasible else np.nan)
        lows.append(low)
        highs.append(high)
        labels.append("No noise" if level["xy_m"] == level["yaw_deg"] == 0
                      else f"±{level['xy_m'] * 100:g} cm\n±{level['yaw_deg']:g}°")
    positions = np.arange(len(levels))
    ax.errorbar(positions, rates, yerr=[np.array(rates) - lows, np.array(highs) - rates], color="blue",
                marker="o", capsize=4)
    ax.axhline(rates[0], color="green", linestyle="--", label="ID baseline")
    ax.legend(fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylim(-.02, 1.02)
    ax.set_ylabel("success rate among feasible (95% Wilson)")
    ax.set_title("Success vs noise level")
    ax.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        records = load_records(run_dir)
        levels = json.loads((run_dir / "manifest.json").read_text()).get("levels") if (run_dir / "manifest.json").exists() else None
        if levels:
            # The zero-noise level is a single repeated pose, so it gets no panel of its own: its rollouts
            # are overlaid green on every noisy panel instead.
            id_records = [r for r in records if r["condition"] == "id"]
            noisy_levels = [level for level in levels if level["xy_m"] or level["yaw_deg"]]
            fig, axes = plt.subplots(1, len(noisy_levels) + 1, figsize=(5.5 * (len(noisy_levels) + 1), 5.5))
            for ax, level in zip(axes, noisy_levels):
                level_records = [r for r in records if r.get("level") == level["name"]]
                plot_poses(ax, level_records, id_records)
                ax.set_title(f"{level['name']}: " + ax.get_title())
            plot_level_summary(axes[-1], records, levels)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            plot_poses(axes[0], records)
            for ax, (key, label, scale) in zip(axes[1:], AXES):
                plot_axis_curve(ax, records, key, label, scale)
        fig.suptitle(f"{records[0]['policy']} / {records[0]['task']}")
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            path = run_dir / f"base_offset{suffix}"
            fig.savefig(path, dpi=180)
            print(f"saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
