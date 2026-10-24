"""Success rate vs. noise level, one line per task, for a level-config base-offset sweep.

Takes the same policy run dirs as summarize_base_offset.py and writes success_vs_noise.png / .pdf beside them.
Error bars are 95% Wilson intervals over the feasible rollouts of each cell.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from summarize_base_offset import LEADERBOARD, LEADERBOARD_LABEL, level_label, tally

# Categorical slots 1-6 in fixed order, validated for CVD separation; markers add a non-color cue.
STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D"), ("#e87ba4", "v"), ("#008300", "P")]
# Each task keeps its style across plots, whichever subset of tasks a run holds; unlisted tasks follow, in name order.
TASK_ORDER = ["lift_pot", "open_laptop", "stack_blocks_two", "click_bell", "beat_block_hammer", "handover_block"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def wilson(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def plot(policy_dir, title):
    tasks = sorted(path.parent for path in policy_dir.glob("*/manifest.json"))
    order = TASK_ORDER + sorted(t.name for t in tasks if t.name not in TASK_ORDER)
    tasks.sort(key=lambda t: order.index(t.name))
    levels = json.loads((tasks[0] / "manifest.json").read_text())["levels"]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    feasible = []
    for i, task_dir in enumerate(tasks):
        color, marker = STYLES[order.index(task_dir.name)]  # more tasks than styles: split the plot instead
        paths = [task_dir / f"{level['name']}.jsonl" for level in levels]
        counts = [tally(path) if path.exists() else (0, 0, 0) for path in paths]  # a level not run is left out
        feasible += [f for _, f, _ in counts if f]
        rate = np.array([100 * s / f if f else np.nan for s, f, _ in counts])
        lo, hi = np.array([wilson(s, f) for s, f, _ in counts]).T * 100
        xs = x + (i - (len(tasks) - 1) / 2) * 0.06  # small dodge so error bars don't overlap
        ax.errorbar(xs, rate, yerr=[rate - lo, hi - rate], color=color, lw=2, marker=marker, ms=7,
                    mec="white", mew=1.5, elinewidth=1, capsize=3, label=task_dir.name.replace("_", " "))
        reference = LEADERBOARD.get(policy_dir.name, {}).get(task_dir.name)
        if reference is not None:  # hollow marker: the published no-noise score
            ax.plot(xs[0], reference, ls="none", marker=marker, ms=11, mfc="none", mec=color, mew=1.5)
    ax.set_xticks(x, [level_label(level) for level in levels])
    ax.set_xlim(-0.3, len(levels) - 0.7)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Base-pose noise (uniform in ±xy, ±yaw)", color=INK)
    ax.set_ylabel("Success rate [%]", color=INK)
    ax.set_title(title or policy_dir.name, color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    handles, labels = ax.get_legend_handles_labels()
    if policy_dir.name in LEADERBOARD:
        handles.append(Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=MUTED, mew=1.5))
        labels.append(LEADERBOARD_LABEL.replace(" (", "\n("))
    ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    n = f"{min(feasible)}" if min(feasible) == max(feasible) else f"{min(feasible)}–{max(feasible)}"
    fig.text(0.01, 0.01, f"n = {n} rollouts per point; error bars: 95% Wilson interval", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(policy_dir / f"success_vs_noise.{suffix}")
    print(policy_dir / "success_vs_noise.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dirs", nargs="+", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()
    for policy_dir in args.policy_dirs:
        plot(policy_dir, args.title)


if __name__ == "__main__":
    main()
