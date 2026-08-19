# Scripts Layout

Top-level `scripts/` is intentionally minimal. Keep stable human and CI entrypoints at the root; keep implementation scripts under their owning domain.

Primary organization:

- `scripts/fea/governance/`: readiness, ratchet, calibration, and external-reference gates.
- `scripts/fea/reporting/`: FEA summaries and trend reports.
- `scripts/fea/prep_calibration/`: prep calibration drift/recommendation/promotion flow.
- `scripts/fea/thermo_artifacts/`: thermo artifact generation/validation/promotion flow.
- `scripts/fea/reference_data/`: benchmark/reference baseline data files.
- `scripts/metadata/`: metadata tooling assets.
- `scripts/runtime/`: runtime/testing helper scripts (wasm/headless verification, etc.).
- `scripts/development/`: reproducible development audits and migration inventories.

Stable entrypoints:

- `scripts/test-wasm-headless.sh`: full local/CI WASM headless verification.
- `scripts/test-fea-scripts.sh`: FEA governance and reporting script unit tests.
- `scripts/check-aot-runtime-dependencies.sh`: verifies that the standalone runtime uses the shared native executor without pulling the adaptive JIT or bytecode VM into its normal dependency graph.
- `scripts/check-closed-world-binary.sh <executable> <link-plan.json>`: verifies that a linked closed-world executable's defined builtin symbols exactly match its plan and that compiler, VM, and JIT symbols were omitted. Qualification hosts need `nm`, `jq`, and `rg`.
- `scripts/development/integer-storage-census.sh`: stable lexical baseline for the authoritative numeric-storage migration. Its output is a discovery frontier, not a defect count.
- `scripts/development/check-architecture-boundaries.sh`: validates durable dependency-direction and ownership boundaries between the value, type, builtin catalog, frontend, and runtime crates.
