# Analysis Architecture Rules

Last updated: 2026-05-17

## Layer Boundaries (Canonical)

- `runmat-analysis/core`
  - Solver-agnostic model and validation schema.
- `runmat-analysis/fea`
  - Physics assembly, solve strategies, diagnostics.
- `runmat-runtime/src/analysis`
  - Operation validation/orchestration, artifact lineage, summaries/trends.
- `scripts/analysis/*`
  - Governance/reporting/evidence lifecycle tooling.

Rule:

- Runtime must not embed physics-equation implementation details.
- Dependency direction remains runtime -> analysis crates (never reverse).

## Contract Rules

1. Version every operation (`.../vN`).
2. Additive evolution only unless explicitly version-bumped.
3. Typed errors and reason codes are stable machine contracts.
4. Run provenance is required (`backend`, solver method, precision, deterministic flag, fallback events).

## Study Workflow Target

Canonical high-level sequence:

1. `analysis.validate_study`
2. `analysis.plan_study`
3. `analysis.run_study`
4. internal expansion to geometry/analysis operations
5. evidence emitted under `.artifacts/analysis/by-id/<analysis_id>/runs/<run_id>/...`

## Determinism and Reproducibility

Required:

- deterministic replay path for repeated inputs,
- immutable run roots,
- stable study fingerprint canonicalization,
- explicit fallback telemetry for backend deviations.

## Prep-Aware Trust Model

- Prep references are artifact-backed (`prep_artifact_id`), not ad-hoc payloads.
- Typed prep mismatch/staleness/schema failures are required.
- Prep fidelity advancement is additive and diagnostics-first.
