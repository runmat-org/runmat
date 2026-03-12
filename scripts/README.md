# Scripts Layout

Top-level `scripts/` is intentionally minimal and directory-oriented.

Primary organization:

- `scripts/release/`: release utility scripts.
- `scripts/analysis/governance/`: readiness, ratchet, calibration, and external-reference gates.
- `scripts/analysis/reporting/`: analysis summaries/trend reports.
- `scripts/analysis/prep_calibration/`: prep calibration drift/recommendation/promotion flow.
- `scripts/analysis/thermo_artifacts/`: thermo artifact generation/validation/promotion flow.
- `scripts/analysis/reference_data/`: benchmark/reference baseline data files.
- `scripts/metadata/`: metadata tooling assets.
- `scripts/runtime/`: runtime/testing helper scripts (wasm/headless verification, etc.).
