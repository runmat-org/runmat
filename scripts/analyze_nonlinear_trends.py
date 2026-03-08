#!/usr/bin/env python3
import json
import os
import statistics
import sys
from pathlib import Path


NONLINEAR_FIXTURES = {
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_proxy_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
}


def threshold_value(record, name):
    for item in record.get("threshold_assertions", []):
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("observed")
    return None


def load_reports(rolling_dir: Path):
    if not rolling_dir.exists():
        return []
    reports = []
    for path in sorted(rolling_dir.glob("analysis_benchmark_report_rolling_*.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except Exception:
            continue
    return reports


def collect_metrics(reports, window):
    fixture_samples = {fixture: [] for fixture in NONLINEAR_FIXTURES}
    for report in reports[-window:]:
        for record in report.get("records", []):
            fixture = record.get("fixture_id")
            if fixture not in NONLINEAR_FIXTURES:
                continue
            fixture_samples[fixture].append(
                {
                    "gpu_run_ms": record.get("gpu_run_ms"),
                    "gpu_speedup_ratio": record.get("gpu_speedup_ratio"),
                    "failed_increments": threshold_value(
                        record,
                        "nonlinear_failed_increments",
                    )
                    if fixture == "nonlinear_assembly_gpu_provider"
                    else threshold_value(record, "nonlinear_stress_failed_increments")
                    if fixture == "nonlinear_assembly_stress_gpu_provider"
                    else threshold_value(record, "nonlinear_softening_failed_increments")
                    if fixture == "nonlinear_softening_proxy_gpu_provider"
                    else threshold_value(record, "nonlinear_path_mix_total_increments"),
                    "publishable": bool(record.get("publishable", False)),
                }
            )
    return fixture_samples


def summarize(samples):
    lines = ["## Nonlinear Trend Summary", ""]
    lines.append("| Fixture | Samples | Median GPU ms | Median speedup | Publishable rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for fixture, values in sorted(samples.items()):
        if not values:
            lines.append(f"| {fixture} | 0 | - | - | - |")
            continue
        gpu = [v["gpu_run_ms"] for v in values if isinstance(v["gpu_run_ms"], (int, float))]
        speedup = [
            v["gpu_speedup_ratio"]
            for v in values
            if isinstance(v["gpu_speedup_ratio"], (int, float))
        ]
        publishable_rate = sum(1 for v in values if v["publishable"]) / len(values)
        lines.append(
            "| {} | {} | {} | {} | {:.2f} |".format(
                fixture,
                len(values),
                f"{statistics.median(gpu):.3f}" if gpu else "-",
                f"{statistics.median(speedup):.3f}" if speedup else "-",
                publishable_rate,
            )
        )
    return "\n".join(lines)


def main():
    rolling_dir = Path(
        os.getenv("RUNMAT_ANALYSIS_BASELINE_DIR", "target/runmat-analysis-artifacts/rolling")
    )
    window = int(os.getenv("RUNMAT_ANALYSIS_TREND_WINDOW", "8"))
    slowdown_limit = float(os.getenv("RUNMAT_ANALYSIS_TREND_MAX_SLOWDOWN_RATIO", "1.25"))
    protected_only = (
        os.getenv("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED", "false").lower() == "true"
    )
    protected = {
        part.strip()
        for part in os.getenv("RUNMAT_ANALYSIS_PROTECTED_BRANCHES", "main,master,release")
        .split(",")
        if part.strip()
    }
    ref_name = os.getenv("GITHUB_REF_NAME", "")

    reports = load_reports(rolling_dir)
    if not reports:
        print("No rolling reports found; skipping nonlinear trend checks")
        return 0

    samples = collect_metrics(reports, max(window, 1))
    summary = summarize(samples)
    print(summary)

    warnings = []
    for fixture, values in samples.items():
        if len(values) < 2:
            continue
        gpu = [v["gpu_run_ms"] for v in values if isinstance(v["gpu_run_ms"], (int, float))]
        if len(gpu) < 2:
            continue
        latest = gpu[-1]
        baseline = statistics.median(gpu[:-1])
        if baseline > 0 and latest / baseline > slowdown_limit:
            warnings.append(
                f"{fixture}: latest gpu_run_ms slowdown ratio {latest / baseline:.3f} exceeds {slowdown_limit:.3f}"
            )

    if warnings:
        print("\nTrend warnings:")
        for warning in warnings:
            print(f"- {warning}")

    should_fail = bool(warnings) and (
        not protected_only or ref_name in protected or any(ref_name.startswith(f"{b}/") for b in protected)
    )
    if should_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
