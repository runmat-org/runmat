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
- `scripts/development/integer-storage-census.sh`: stable lexical baseline for the authoritative numeric-storage migration. Its output is a discovery frontier, not a defect count.
- `scripts/development/check-rm1064-inventory.sh`: deterministic RM-1064 cross-domain authority and migration-inventory guard. The generated report is evidence for migration/zero-state closure, not a runtime semantic authority.
- `scripts/development/check-rm1064-value-cutover.sh`: validates the exact I03C live-value declaration owner, structural baseline ancestry, non-value relocation classifications, and representative pre-I04 pilot set. R03 updates the manifest atomically to `extracted`; R04/R06 update it to `catalog-separated` when the bounded `runmat-builtins -> runmat-value` dependency is removed.
