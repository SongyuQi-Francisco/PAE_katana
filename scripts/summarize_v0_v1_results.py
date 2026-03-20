#!/usr/bin/env python3
import argparse
import glob
import json
import os
from datetime import datetime


DOMAINS = ["amazon", "yelp", "goodreads"]
METRICS = [
    ("HR@1", "top_1_hit_rate"),
    ("HR@3", "top_3_hit_rate"),
    ("HR@5", "top_5_hit_rate"),
    ("NDCG@1", "ndcg_at_1"),
    ("NDCG@3", "ndcg_at_3"),
    ("NDCG@5", "ndcg_at_5"),
]


def load_latest_result(results_root, method_name, domain, version):
    pattern = os.path.join(results_root, method_name, f"{domain}_{version}_*_metrics.json")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None, None
    latest = max(candidates, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as file:
        return json.load(file), latest


def weighted_metric(rows, metric_key):
    numerator = 0.0
    denominator = 0
    for row in rows:
        total = row["total_scenarios"]
        numerator += row[metric_key] * total
        denominator += total
    return numerator / denominator if denominator else None


def format_value(value):
    return "-" if value is None else f"{value:.4f}"


def format_delta(value):
    return "-" if value is None else f"{value:+.4f}"


def build_section(results_root, phase_name, baseline_method, baseline_version, evolved_method, evolved_version):
    rows = []
    files_used = []

    for domain in DOMAINS:
        baseline_data, baseline_path = load_latest_result(results_root, baseline_method, domain, baseline_version)
        evolved_data, evolved_path = load_latest_result(results_root, evolved_method, domain, evolved_version)
        if not baseline_data or not evolved_data:
            continue

        baseline_metrics = baseline_data["metrics"]
        evolved_metrics = evolved_data["metrics"]
        row = {
            "domain": domain,
            "total_scenarios": baseline_metrics.get("total_scenarios", 0),
        }

        for _, metric_key in METRICS:
            baseline_value = baseline_metrics.get(metric_key)
            evolved_value = evolved_metrics.get(metric_key)
            row[f"baseline_{metric_key}"] = baseline_value
            row[f"evolved_{metric_key}"] = evolved_value
            row[f"delta_{metric_key}"] = (
                evolved_value - baseline_value
                if baseline_value is not None and evolved_value is not None
                else None
            )

        rows.append(row)
        files_used.append((domain, baseline_path, evolved_path))

    if not rows:
        return None

    total_tasks = sum(row["total_scenarios"] for row in rows)
    lines = [f"## {phase_name}", "", "| Domain | Tasks | Metric | V0 | V1 | Δ |", "|---|---:|---|---:|---:|---:|"]

    for row in rows:
        for metric_label, metric_key in METRICS:
            lines.append(
                f"| {row['domain']} | {row['total_scenarios']} | {metric_label} | "
                f"{format_value(row[f'baseline_{metric_key}'])} | "
                f"{format_value(row[f'evolved_{metric_key}'])} | "
                f"{format_delta(row[f'delta_{metric_key}'])} |"
            )

    for metric_label, metric_key in [("HR@1", "top_1_hit_rate"), ("NDCG@5", "ndcg_at_5")]:
        baseline_weighted = weighted_metric(rows, f"baseline_{metric_key}")
        evolved_weighted = weighted_metric(rows, f"evolved_{metric_key}")
        delta = None
        if baseline_weighted is not None and evolved_weighted is not None:
            delta = evolved_weighted - baseline_weighted
        lines.append(
            f"| weighted_avg | {total_tasks} | {metric_label} | "
            f"{format_value(baseline_weighted)} | {format_value(evolved_weighted)} | {format_delta(delta)} |"
        )

    lines.append("")
    lines.append("### Files Used")
    for domain, baseline_path, evolved_path in files_used:
        lines.append(f"- {domain} V0: `{baseline_path}`")
        lines.append(f"- {domain} V1: `{evolved_path}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize latest timestamped V0 vs V1 results.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/comparisons")
    args = parser.parse_args()

    specs = [
        ("Gradual Dev-Val", "v0a_devval_baseline", "v0a_devval", "v1a_devval_evolved", "v1a_devval"),
        ("Cognitive Dev-Val", "v0b_devval_baseline", "v0b_devval", "v1b_devval_evolved", "v1b_devval"),
        ("Gradual Track2", "v0a_track2_baseline", "v0a_track2", "v1a_track2_evolved", "v1a_track2"),
        ("Cognitive Track2", "v0b_track2_baseline", "v0b_track2", "v1b_track2_evolved", "v1b_track2"),
    ]

    sections = []
    for spec in specs:
        section = build_section(args.results_root, *spec)
        if section:
            sections.append(section)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"v0_v1_summary_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# V0 vs V1 Quantitative Summary\n\n")
        file.write(f"Generated at: {timestamp}\n\n")
        if sections:
            file.write("\n\n".join(sections))
        else:
            file.write("No matching timestamped result files found.\n")

    print(output_path)


if __name__ == "__main__":
    main()
