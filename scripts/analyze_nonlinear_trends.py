#!/usr/bin/env python3
import json
import os
import statistics
import sys
from pathlib import Path

from scripts.evaluate_prep_calibration_drift import (
    evaluate_report_drift,
    evaluate_rolling_drift,
    load_evidence,
    recommend_profile_shifts,
)


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
                    "prep_acceptance_score": record.get("prep_acceptance_score"),
                    "thermo_coupling_enabled": record.get("thermo_coupling_enabled"),
                    "thermo_effective_modulus_scale": record.get(
                        "thermo_effective_modulus_scale"
                    ),
                    "thermo_constitutive_material_spread_ratio": record.get(
                        "thermo_constitutive_material_spread_ratio"
                    ),
                    "thermo_assignment_heterogeneity_index": record.get(
                        "thermo_assignment_heterogeneity_index"
                    ),
                    "thermo_spatial_coverage_ratio": record.get(
                        "thermo_spatial_coverage_ratio"
                    ),
                    "thermo_field_extrapolation_ratio": record.get(
                        "thermo_field_extrapolation_ratio"
                    ),
                    "thermo_transient_severity": record.get("thermo_transient_severity"),
                    "thermo_nonlinear_severity": record.get("thermo_nonlinear_severity"),
                    "electro_thermal_coupling_enabled": record.get(
                        "electro_thermal_coupling_enabled"
                    ),
                    "electro_joule_heating_scale": record.get("electro_joule_heating_scale"),
                    "electro_conductivity_spread_ratio": record.get(
                        "electro_conductivity_spread_ratio"
                    ),
                    "electro_transient_severity": record.get("electro_transient_severity"),
                    "electro_nonlinear_severity": record.get("electro_nonlinear_severity"),
                }
            )
    return fixture_samples


def summarize(samples):
    lines = ["## Nonlinear Trend Summary", ""]
    lines.append(
        "| Fixture | Samples | Median GPU ms | Median speedup | Publishable rate | Median acceptance score | Thermo enabled rate | Median thermo modulus scale | Median thermo spread ratio | Median thermo heterogeneity | Median thermo transient sev | Median thermo nonlinear sev | Electro enabled rate | Median electro joule scale | Median electro spread ratio | Median electro transient sev | Median electro nonlinear sev |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for fixture, values in sorted(samples.items()):
        if not values:
            lines.append(f"| {fixture} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        gpu = [v["gpu_run_ms"] for v in values if isinstance(v["gpu_run_ms"], (int, float))]
        speedup = [
            v["gpu_speedup_ratio"]
            for v in values
            if isinstance(v["gpu_speedup_ratio"], (int, float))
        ]
        publishable_rate = sum(1 for v in values if v["publishable"]) / len(values)
        acceptance_scores = [
            v["prep_acceptance_score"]
            for v in values
            if isinstance(v.get("prep_acceptance_score"), (int, float))
        ]
        thermo_enabled_values = [
            v["thermo_coupling_enabled"]
            for v in values
            if isinstance(v.get("thermo_coupling_enabled"), bool)
        ]
        thermo_transient_values = [
            v["thermo_transient_severity"]
            for v in values
            if isinstance(v.get("thermo_transient_severity"), (int, float))
        ]
        thermo_modulus_values = [
            v["thermo_effective_modulus_scale"]
            for v in values
            if isinstance(v.get("thermo_effective_modulus_scale"), (int, float))
        ]
        thermo_nonlinear_values = [
            v["thermo_nonlinear_severity"]
            for v in values
            if isinstance(v.get("thermo_nonlinear_severity"), (int, float))
        ]
        thermo_spread_values = [
            v["thermo_constitutive_material_spread_ratio"]
            for v in values
            if isinstance(v.get("thermo_constitutive_material_spread_ratio"), (int, float))
        ]
        thermo_heterogeneity_values = [
            v["thermo_assignment_heterogeneity_index"]
            for v in values
            if isinstance(v.get("thermo_assignment_heterogeneity_index"), (int, float))
        ]
        electro_enabled_values = [
            v["electro_thermal_coupling_enabled"]
            for v in values
            if isinstance(v.get("electro_thermal_coupling_enabled"), bool)
        ]
        electro_joule_values = [
            v["electro_joule_heating_scale"]
            for v in values
            if isinstance(v.get("electro_joule_heating_scale"), (int, float))
        ]
        electro_spread_values = [
            v["electro_conductivity_spread_ratio"]
            for v in values
            if isinstance(v.get("electro_conductivity_spread_ratio"), (int, float))
        ]
        electro_transient_values = [
            v["electro_transient_severity"]
            for v in values
            if isinstance(v.get("electro_transient_severity"), (int, float))
        ]
        electro_nonlinear_values = [
            v["electro_nonlinear_severity"]
            for v in values
            if isinstance(v.get("electro_nonlinear_severity"), (int, float))
        ]
        lines.append(
            "| {} | {} | {} | {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                fixture,
                len(values),
                f"{statistics.median(gpu):.3f}" if gpu else "-",
                f"{statistics.median(speedup):.3f}" if speedup else "-",
                publishable_rate,
                f"{statistics.median(acceptance_scores):.3f}" if acceptance_scores else "-",
                (
                    f"{sum(1 for v in thermo_enabled_values if v) / len(thermo_enabled_values):.3f}"
                    if thermo_enabled_values
                    else "-"
                ),
                (
                    f"{statistics.median(thermo_modulus_values):.3f}"
                    if thermo_modulus_values
                    else "-"
                ),
                (
                    f"{statistics.median(thermo_spread_values):.3f}"
                    if thermo_spread_values
                    else "-"
                ),
                (
                    f"{statistics.median(thermo_heterogeneity_values):.3f}"
                    if thermo_heterogeneity_values
                    else "-"
                ),
                (
                    f"{statistics.median(thermo_transient_values):.3f}"
                    if thermo_transient_values
                    else "-"
                ),
                (
                    f"{statistics.median(thermo_nonlinear_values):.3f}"
                    if thermo_nonlinear_values
                    else "-"
                ),
                (
                    f"{sum(1 for v in electro_enabled_values if v) / len(electro_enabled_values):.3f}"
                    if electro_enabled_values
                    else "-"
                ),
                (
                    f"{statistics.median(electro_joule_values):.3f}"
                    if electro_joule_values
                    else "-"
                ),
                (
                    f"{statistics.median(electro_spread_values):.3f}"
                    if electro_spread_values
                    else "-"
                ),
                (
                    f"{statistics.median(electro_transient_values):.3f}"
                    if electro_transient_values
                    else "-"
                ),
                (
                    f"{statistics.median(electro_nonlinear_values):.3f}"
                    if electro_nonlinear_values
                    else "-"
                ),
            )
        )
    return "\n".join(lines)


def main():
    rolling_dir = Path(
        os.getenv("RUNMAT_ANALYSIS_BASELINE_DIR", "target/runmat-analysis-artifacts/rolling")
    )
    window = int(os.getenv("RUNMAT_ANALYSIS_TREND_WINDOW", "8"))
    slowdown_limit = float(os.getenv("RUNMAT_ANALYSIS_TREND_MAX_SLOWDOWN_RATIO", "1.25"))
    thermo_spread_trend_limit = float(
        os.getenv("RUNMAT_ANALYSIS_THERMO_MAX_SPREAD_TREND_RATIO", "1.2")
    )
    thermo_heterogeneity_trend_limit = float(
        os.getenv("RUNMAT_ANALYSIS_THERMO_MAX_HETEROGENEITY_TREND_RATIO", "1.2")
    )
    thermo_coverage_drop_limit = float(
        os.getenv("RUNMAT_ANALYSIS_THERMO_MAX_COVERAGE_DROP_RATIO", "1.25")
    )
    thermo_extrapolation_trend_limit = float(
        os.getenv("RUNMAT_ANALYSIS_THERMO_MAX_FIELD_EXTRAPOLATION_TREND_RATIO", "1.25")
    )
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

    evidence_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_EVIDENCE",
            "scripts/prep_calibration_evidence.json",
        )
    )
    evidence = load_evidence(evidence_path)
    if evidence is not None:
        latest_report = reports[-1]
        drift_rows = evaluate_report_drift(latest_report, evidence)
        if drift_rows:
            max_drift = max(row.get("drift_ratio", 0.0) for row in drift_rows)
            drift_slopes = evaluate_rolling_drift(reports[-max(window, 1) :], evidence)
            max_drift_slope = max(drift_slopes.values()) if drift_slopes else 0.0
            recommendations = recommend_profile_shifts(
                latest_report,
                reports[-max(window, 1) :],
                evidence,
                drift_trigger=float(
                    os.getenv("RUNMAT_RELEASE_READINESS_PREP_RETRAIN_TRIGGER_DRIFT", "0.1")
                ),
            )
            recommendation_pressure = (
                len(recommendations) / len(NONLINEAR_FIXTURES) if NONLINEAR_FIXTURES else 0.0
            )
            print("\nCalibration drift summary:")
            print(f"- max drift ratio: {max_drift:.3f}")
            print(f"- max drift slope: {max_drift_slope:.4f}")
            print(
                f"- recommendation pressure: {recommendation_pressure:.3f} ({len(recommendations)} fixtures)"
            )

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

        spread = [
            v["thermo_constitutive_material_spread_ratio"]
            for v in values
            if isinstance(v.get("thermo_constitutive_material_spread_ratio"), (int, float))
        ]
        if len(spread) >= 2:
            baseline_spread = statistics.median(spread[:-1]) if len(spread) > 1 else spread[0]
            latest_spread = spread[-1]
            if baseline_spread > 0:
                spread_ratio = latest_spread / baseline_spread
                if spread_ratio > thermo_spread_trend_limit:
                    warnings.append(
                        f"THERMO_SPREAD_TREND_WORSENING:{fixture}: ratio {spread_ratio:.3f} exceeds {thermo_spread_trend_limit:.3f}"
                    )

        heterogeneity = [
            v["thermo_assignment_heterogeneity_index"]
            for v in values
            if isinstance(v.get("thermo_assignment_heterogeneity_index"), (int, float))
        ]
        if len(heterogeneity) >= 2:
            baseline_heterogeneity = (
                statistics.median(heterogeneity[:-1])
                if len(heterogeneity) > 1
                else heterogeneity[0]
            )
            latest_heterogeneity = heterogeneity[-1]
            if baseline_heterogeneity > 0:
                heterogeneity_ratio = latest_heterogeneity / baseline_heterogeneity
                if heterogeneity_ratio > thermo_heterogeneity_trend_limit:
                    warnings.append(
                        f"THERMO_HETEROGENEITY_TREND_WORSENING:{fixture}: ratio {heterogeneity_ratio:.3f} exceeds {thermo_heterogeneity_trend_limit:.3f}"
                    )

        coverage = [
            v["thermo_spatial_coverage_ratio"]
            for v in values
            if isinstance(v.get("thermo_spatial_coverage_ratio"), (int, float))
        ]
        if len(coverage) >= 2:
            latest_coverage = coverage[-1]
            baseline_coverage = statistics.median(coverage[:-1])
            if latest_coverage > 0:
                coverage_drop_ratio = baseline_coverage / latest_coverage
                if coverage_drop_ratio > thermo_coverage_drop_limit:
                    warnings.append(
                        f"THERMO_FIELD_COVERAGE_WORSENING:{fixture}: drop ratio {coverage_drop_ratio:.3f} exceeds {thermo_coverage_drop_limit:.3f}"
                    )

        extrapolation = [
            v["thermo_field_extrapolation_ratio"]
            for v in values
            if isinstance(v.get("thermo_field_extrapolation_ratio"), (int, float))
        ]
        if len(extrapolation) >= 2:
            baseline_extrapolation = (
                statistics.median(extrapolation[:-1]) if len(extrapolation) > 1 else extrapolation[0]
            )
            latest_extrapolation = extrapolation[-1]
            if baseline_extrapolation > 0:
                extrapolation_ratio = latest_extrapolation / baseline_extrapolation
                if extrapolation_ratio > thermo_extrapolation_trend_limit:
                    warnings.append(
                        f"THERMO_FIELD_EXTRAPOLATION_WORSENING:{fixture}: ratio {extrapolation_ratio:.3f} exceeds {thermo_extrapolation_trend_limit:.3f}"
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
