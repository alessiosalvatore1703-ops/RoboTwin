"""Success-rate table per task and noise level, among feasible rollouts (infeasible poses leave the denominator).

Takes policy run dirs written by run_base_offset_sweep.py with a `levels` config (<dir>/<task>/manifest.json
plus one <level>.jsonl per level); prints a Markdown table and writes summary.md / summary.csv beside them.
"""
import argparse
import csv
import json
from pathlib import Path

# RoboTwin 2.0 leaderboard, clean2clean (easy) = demo_clean, 100 trials per task, co-train setting; keyed by run dir.
# Source: https://robotwin-platform.github.io/data/robotwin_leaderboard.json (updated 2026-10-21).
LEADERBOARD = {
    "Pi_05": {"lift_pot": 98, "open_laptop": 93, "stack_blocks_two": 91, "click_bell": 31,
              "beat_block_hammer": 93, "handover_block": 30},
    "FastWAM": {"lift_pot": 91, "open_laptop": 84, "stack_blocks_two": 99, "click_bell": 95,
                "beat_block_hammer": 82, "handover_block": 66},
    "OLA_SEM": {"lift_pot": 100, "open_laptop": 90, "stack_blocks_two": 91, "click_bell": 100,
                "beat_block_hammer": 65, "handover_block": 24},
}
LEADERBOARD_LABEL = "RoboTwin leaderboard (no noise)"


def level_label(level):
    if level["xy_m"] == 0 and level["yaw_deg"] == 0:
        return "No noise"
    return f"±{level['xy_m'] * 100:g} cm / ±{level['yaw_deg']:g}°"


def tally(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return (sum(r["status"] == "success" for r in rows), sum(r["status"] != "infeasible" for r in rows),
            sum(r["status"] == "infeasible" for r in rows))


def cell(successes, feasible, infeasible):
    text = f"{100 * successes / feasible:.1f}% ({successes}/{feasible})" if feasible else "n/a (0 feasible)"
    return text + (f", {infeasible} infeasible" if infeasible else "")


def summarize(policy_dir):
    tasks = sorted(path.parent for path in policy_dir.glob("*/manifest.json"))
    levels = json.loads((tasks[0] / "manifest.json").read_text())["levels"]
    if not levels:
        raise SystemExit(f"{policy_dir}: manifest has no `levels`; this table is for level-config runs")
    reference = LEADERBOARD.get(policy_dir.name, {})
    table, pooled = [], {level["name"]: [0, 0, 0] for level in levels}
    for task_dir in tasks:
        row = [task_dir.name]
        for level in levels:
            path = task_dir / f"{level['name']}.jsonl"
            counts = tally(path) if path.exists() else (0, 0, 0)
            pooled[level["name"]] = [a + b for a, b in zip(pooled[level["name"]], counts)]
            row.append(cell(*counts) if path.exists() else "not run")
        if reference:
            row.insert(2, f"{reference[task_dir.name]:g}%" if task_dir.name in reference else "n/a")
        table.append(row)
    table.append(["All tasks"] + [cell(*pooled[level["name"]]) for level in levels])
    header = ["Task"] + [level_label(level) for level in levels]
    if reference:
        # Unweighted mean over tasks; the pooled rollouts are equally split across tasks, so this is comparable.
        known = [reference[t.name] for t in tasks if t.name in reference]
        table[-1].insert(2, f"{sum(known) / len(known):.1f}%")
        header.insert(2, "Leaderboard (no noise)")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    markdown = "\n".join(lines) + "\n\nSuccess rate among feasible rollouts; counts are successes / feasible.\n"
    if reference:
        markdown += "Leaderboard: RoboTwin 2.0 clean2clean (demo_clean, 100 trials per task, co-train).\n"
    (policy_dir / "summary.md").write_text(markdown)
    with open(policy_dir / "summary.csv", "w", newline="") as file:
        csv.writer(file).writerows([header] + table)
    print(f"{policy_dir.name}\n{markdown}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dirs", nargs="+", type=Path)
    for policy_dir in parser.parse_args().policy_dirs:
        summarize(policy_dir)


if __name__ == "__main__":
    main()
