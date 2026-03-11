#!/usr/bin/env python3
import json
import sys
from pathlib import Path


NONLINEAR_FIXTURES = (
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_proxy_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
    "nonlinear_plasticity_proxy_gpu_provider",
)

THERMO_FIXTURE = "thermo_mech_kickoff_gpu_provider"


def threshold_map(record):
    return {
        item.get("name"): item
        for item in record.get("threshold_assertions", [])
        if isinstance(item, dict) and item.get("name")
    }


def format_num(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def build_summary(report):
    lines = []
    records_by_id = {
        record.get("fixture_id"): record
        for record in report.get("records", [])
        if isinstance(record, dict)
    }
    lines.append("## Nonlinear Conformance Summary")
    lines.append("")
    lines.append(
        f"Overall benchmark/conformance pass: **{bool(report.get('passed', False))}**"
    )
    lines.append("")
    lines.append(
        "| Fixture | Publishable | GPU ms | Speedup | Failed increments | Backtracks | Tangent rebuilds | Calibration profile | Acceptance score | Thermo enabled | Thermo modulus scale | Thermo spread ratio | Thermo heterogeneity | Thermo transient sev | Thermo nonlinear sev | Electro enabled | Electro joule scale | Electro spread ratio | Electro transient sev | Electro nonlinear sev | Plastic nonlinear sev |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    )

    for fixture_id in NONLINEAR_FIXTURES:
        record = records_by_id.get(fixture_id)
        if record is None:
            lines.append(f"| {fixture_id} | missing | - | - | - | - | - |")
            continue
        tmap = threshold_map(record)

        failed = tmap.get("nonlinear_failed_increments", {}).get("observed")
        if failed is None:
            failed = tmap.get("nonlinear_stress_failed_increments", {}).get("observed")

        backtracks = tmap.get("nonlinear_line_search_backtracks", {}).get("observed")
        if backtracks is None:
            backtracks = tmap.get("nonlinear_stress_line_search_backtracks", {}).get(
                "observed"
            )

        tangent = tmap.get("nonlinear_stress_tangent_rebuild_count", {}).get("observed")
        if tangent is None:
            tangent = tmap.get("nonlinear_tangent_rebuild_count", {}).get("observed")

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                fixture_id,
                record.get("publishable", "-"),
                format_num(record.get("gpu_run_ms")),
                format_num(record.get("gpu_speedup_ratio")),
                format_num(failed),
                format_num(backtracks),
                format_num(tangent),
                record.get("prep_calibration_profile", "-"),
                format_num(record.get("prep_acceptance_score")),
                record.get("thermo_coupling_enabled", "-"),
                format_num(record.get("thermo_effective_modulus_scale")),
                format_num(record.get("thermo_constitutive_material_spread_ratio")),
                format_num(record.get("thermo_assignment_heterogeneity_index")),
                format_num(record.get("thermo_transient_severity")),
                format_num(record.get("thermo_nonlinear_severity")),
                record.get("electro_thermal_coupling_enabled", "-"),
                format_num(record.get("electro_joule_heating_scale")),
                format_num(record.get("electro_conductivity_spread_ratio")),
                format_num(record.get("electro_transient_severity")),
                format_num(record.get("electro_nonlinear_severity")),
                format_num(record.get("plastic_nonlinear_severity")),
            )
        )

    lines.append("")
    lines.append("### Thermo Kickoff")
    thermo_record = records_by_id.get(THERMO_FIXTURE)
    if thermo_record is None:
        lines.append(f"- `{THERMO_FIXTURE}`: missing")
    else:
        lines.append(
            "- enabled={} modulus_scale={} spread_ratio={} heterogeneity_index={} transient_severity={} nonlinear_severity={}".format(
                thermo_record.get("thermo_coupling_enabled", "-"),
                format_num(thermo_record.get("thermo_effective_modulus_scale")),
                format_num(thermo_record.get("thermo_constitutive_material_spread_ratio")),
                format_num(thermo_record.get("thermo_assignment_heterogeneity_index")),
                format_num(thermo_record.get("thermo_transient_severity")),
                format_num(thermo_record.get("thermo_nonlinear_severity")),
            )
        )

    lines.append("")
    lines.append("### Failures")
    failure_lines = []
    for fixture_id in NONLINEAR_FIXTURES:
        record = records_by_id.get(fixture_id)
        if not record:
            continue
        for failure in record.get("failures", []):
            failure_lines.append(f"- `{fixture_id}`: {failure}")
    if failure_lines:
        lines.extend(failure_lines)
    else:
        lines.append("- None")
    lines.append("")
    lines.append(
        "Raw report: `target/runmat-analysis-artifacts/analysis_benchmark_report.json`"
    )
    return "\n".join(lines)


def main():
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("target/runmat-analysis-artifacts/analysis_benchmark_report.json")
    )
    if not path.exists():
        print(f"analysis report not found: {path}", file=sys.stderr)
        return 1
    report = json.loads(path.read_text())
    print(build_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
