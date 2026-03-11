#!/usr/bin/env python3
import json
import os
from pathlib import Path


RATCHET_HISTORY = {
    "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO": {
        "release": {"old": 1.1, "new": 1.1},
        "development": {"old": 1.2, "new": 1.16},
        "feature": {"old": 1.35, "new": 1.28},
    },
    "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO": {
        "release": {"old": 1.1, "new": 1.1},
        "development": {"old": 1.2, "new": 1.15},
        "feature": {"old": 1.35, "new": 1.26},
    },
    "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO": {
        "release": {"old": 1.1, "new": 1.1},
        "development": {"old": 1.2, "new": 1.16},
        "feature": {"old": 1.35, "new": 1.28},
    },
    "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO": {
        "release": {"old": 1.1, "new": 1.1},
        "development": {"old": 1.2, "new": 1.15},
        "feature": {"old": 1.35, "new": 1.26},
    },
}

OBSERVED_FIELDS = {
    "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO": "plastic_trend_ratio",
    "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO": "plastic_reference_trend_ratio",
    "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO": "contact_trend_ratio",
    "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO": "contact_reference_trend_ratio",
}


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    in_path = Path(
        os.getenv(
            "RUNMAT_THRESHOLD_RATCHET_INPUT",
            "target/runmat-analysis-artifacts/nonlinear_release_readiness.json",
        )
    )
    out_json = Path(
        os.getenv(
            "RUNMAT_THRESHOLD_RATCHET_OUTPUT_JSON",
            "target/runmat-analysis-artifacts/threshold_ratchet_report.json",
        )
    )
    out_md = Path(
        os.getenv(
            "RUNMAT_THRESHOLD_RATCHET_OUTPUT_MD",
            "target/runmat-analysis-artifacts/threshold_ratchet_report.md",
        )
    )

    if not in_path.exists():
        print(f"readiness report missing: {in_path}")
        return 1

    readiness = json.loads(in_path.read_text())
    profile = readiness.get("governance_profile", "unknown")
    rationale = readiness.get("reference_trend_rationale", "unknown")
    rolling_count = readiness.get("rolling_report_count", 0)

    entries = []
    for key, profile_values in RATCHET_HISTORY.items():
        values = profile_values.get(profile)
        if not values:
            continue
        old = float(values["old"])
        new = float(values["new"])
        observed = readiness.get(OBSERVED_FIELDS[key])
        status = "ratcheted" if new < old else "unchanged"
        entries.append(
            {
                "threshold_key": key,
                "profile": profile,
                "old": old,
                "new": new,
                "delta": new - old,
                "observed": observed,
                "status": status,
            }
        )

    report = {
        "schema_version": "threshold-ratchet-report/v1",
        "governance_profile": profile,
        "rationale": rationale,
        "rolling_report_count": rolling_count,
        "entries": entries,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "## Threshold Ratchet Report",
        "",
        f"- Governance profile: `{profile}`",
        f"- Rationale: `{rationale}`",
        f"- Rolling reports: `{rolling_count}`",
        "",
        "| Threshold Key | Old | New | Delta | Observed | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                entry["threshold_key"],
                fmt(entry["old"]),
                fmt(entry["new"]),
                fmt(entry["delta"]),
                fmt(entry["observed"]),
                entry["status"],
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
