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
- `scripts/runtime/verify-builtin-examples.mjs`: inventories, plans, executes, and strictly reconciles catalog and legacy builtin examples across typed native, browser, graphics, and WGPU lanes; the protocol and legacy compatibility interface are documented in `docs/development/builtin-example-sharding.md`.
- `scripts/runtime/stage-builtin-example-product.mjs`: stages one native executable and its adjacent Windows runtime libraries without accepting conflicting dependency bytes.
- `scripts/runtime/run-builtin-example-product.mjs`: executes every shard in one product-scoped verifier plan with separate result and work paths, using the exact artifact manifest declared on the command line. Distributed CI invokes the single-shard `run` command directly from its frozen matrix.
- `scripts/development/builtin-migration-factory.mjs`: emits the development-only C00-C07 identity inventory and complexity-weighted work queue. See `docs/development/builtin-migration-factory.md`; generated output is evidence, never an editable production authority.
- Its `verify` command reconciles an exact migration audit with pinned combined/sharded example reports; verification output remains development evidence only.
