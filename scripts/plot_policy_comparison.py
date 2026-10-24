"""Per-task comparison of policies: success rate vs. position noise, one line per policy.

Takes a sweep out_dir holding one run dir per policy (as written by run_base_offset_sweep.py with a `levels` config)
and writes comparison/<task>.png / .pdf plus comparison/all_tasks.png / .pdf (all tasks in one row) into it. The x axis is the position-noise half-width in cm at its true
spacing; each tick also shows the yaw noise that comes with that level.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_success_vs_noise import INK, MUTED, GRID, TASK_ORDER
from summarize_base_offset import LEADERBOARD, LEADERBOARD_LABEL, tally

# Each policy keeps its color/marker in every plot (categorical slots 1-3, validated for CVD separation).
XLABEL = "Position noise, uniform in x and y [cm]   (yaw noise of the same level in brackets)"
POLICIES = [("Pi_05", "π0.5", "#2a78d6", "o"), ("FastWAM", "FastWAM", "#eb6834", "s"), ("OLA_SEM", "UniWAM", "#1baf7a", "^")]


def draw_task(ax, out_dir, task, tick_size=None):
    """Draw one task's policy lines into ax; returns the feasible-rollout counts of the plotted cells."""
    runs = [(run, label, color, marker) for run, label, color, marker in POLICIES
            if (out_dir / run / task / "manifest.json").exists()]
    levels = json.loads((out_dir / runs[0][0] / task / "manifest.json").read_text())["levels"]
    x = np.array([100 * level["xy_m"] for level in levels])
    feasible = []
    for run, label, color, marker in runs:
        paths = [out_dir / run / task / f"{level['name']}.jsonl" for level in levels]
        counts = [tally(path) if path.exists() else (0, 0, 0) for path in paths]  # a level not run is left out
        feasible += [f for _, f, _ in counts if f]
        rate = np.array([100 * s / f if f else np.nan for s, f, _ in counts])
        ax.plot(x, rate, color=color, lw=2, marker=marker, ms=7, mec="white", mew=1.5, label=label)
        if task in LEADERBOARD.get(run, {}):  # hollow marker: the published no-noise score
            ax.plot(0, LEADERBOARD[run][task], ls="none", marker=marker, ms=11, mfc="none", mec=color, mew=1.5)
    ax.set_xticks(x, [f"{'±' if v else ''}{v:g}\n({'±' if v else ''}{level['yaw_deg']:g}°)" for v, level in zip(x, levels)],
                  fontsize=tick_size)
    ax.set_xlim(x.min() - 0.8, x.max() + 0.8)
    ax.set_ylim(0, 105)
    ax.set_title(task.replace("_", " "), color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    return feasible


def legend_entries(ax):
    handles, labels = ax.get_legend_handles_labels()
    reference = Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=MUTED, mew=1.5)
    return handles + [reference], labels + [LEADERBOARD_LABEL]


def n_note(feasible):
    n = f"{min(feasible)}" if min(feasible) == max(feasible) else f"{min(feasible)}–{max(feasible)}"
    return f"n = {n} rollouts per point"


def save(fig, out_dir, name):
    (out_dir / "comparison").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / "comparison" / f"{name}.{suffix}")
    plt.close(fig)
    print(out_dir / "comparison" / f"{name}.png")


def plot_task(out_dir, task):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    feasible = draw_task(ax, out_dir, task)
    ax.set_xlabel(XLABEL, color=INK)
    ax.set_ylabel("Success rate [%]", color=INK)
    ax.legend(*legend_entries(ax), frameon=False, loc="upper right", fontsize=9)
    fig.text(0.01, 0.01, n_note(feasible), fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out_dir, task)


def plot_row(out_dir, tasks):
    """All tasks side by side in one horizontal image, sharing the y axis and one legend."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4.2), dpi=200, sharey=True)
    feasible = []
    for ax, task in zip(np.atleast_1d(axes), tasks):
        feasible += draw_task(ax, out_dir, task, tick_size=8)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Success rate [%]", color=INK)
    handles, labels = legend_entries(first)
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=len(labels), fontsize=10)
    fig.supxlabel(XLABEL, color=INK, fontsize=10)
    fig.text(0.005, 0.01, n_note(feasible), fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, out_dir, "all_tasks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    names = {path.parent.name for run, *_ in POLICIES for path in (args.out_dir / run).glob("*/manifest.json")}
    tasks = [t for t in TASK_ORDER if t in names] + sorted(names - set(TASK_ORDER))
    for task in tasks:
        plot_task(args.out_dir, task)
    plot_row(args.out_dir, tasks)


if __name__ == "__main__":
    main()
