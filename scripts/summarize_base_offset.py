"""Success-rate table per task and noise level, among feasible rollouts (infeasible poses leave the denominator).

Takes policy run dirs written by run_base_offset_sweep.py with a `levels` config (<dir>/<task>/manifest.json
plus one <level>.jsonl per level); prints a Markdown table and writes summary.md / summary.csv beside them.
"""
import argparse
import csv
import json
from pathlib import Path


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
    table, pooled = [], {level["name"]: [0, 0, 0] for level in levels}
    for task_dir in tasks:
        row = [task_dir.name]
        for level in levels:
            path = task_dir / f"{level['name']}.jsonl"
            counts = tally(path) if path.exists() else (0, 0, 0)
            pooled[level["name"]] = [a + b for a, b in zip(pooled[level["name"]], counts)]
            row.append(cell(*counts) if path.exists() else "not run")
        table.append(row)
    table.append(["All tasks"] + [cell(*pooled[level["name"]]) for level in levels])
    header = ["Task"] + [level_label(level) for level in levels]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    markdown = "\n".join(lines) + "\n\nSuccess rate among feasible rollouts; counts are successes / feasible.\n"
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
