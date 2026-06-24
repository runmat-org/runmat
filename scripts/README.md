# Scripts Layout

Top-level `scripts/` is intentionally minimal. Keep stable human and CI
entrypoints at the root; keep implementation scripts under their owning domain.

Primary organization:

- `scripts/fea/governance/`: readiness, ratchet, calibration, and external-reference gates.
- `scripts/fea/reporting/`: FEA summaries and trend reports.
- `scripts/fea/prep_calibration/`: prep calibration drift/recommendation/promotion flow.
- `scripts/fea/thermo_artifacts/`: thermo artifact generation/validation/promotion flow.
- `scripts/fea/reference_data/`: benchmark/reference baseline data files.
- `scripts/metadata/`: metadata tooling assets.
- `scripts/runtime/`: runtime/testing helper scripts (wasm/headless verification, etc.).

Stable entrypoints:

- `scripts/test-wasm-headless.sh`: full local/CI WASM headless verification.
- `scripts/test-fea-scripts.sh`: FEA governance and reporting script unit tests.
