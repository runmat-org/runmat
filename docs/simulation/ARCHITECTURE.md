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

Canonical prep-aware solve flow:

1. `geometry.load/v1`
2. `geometry.prep_for_analysis/v1`
3. `analysis.create_model/v1`
4. `analysis.run_*/v1` with `prep_artifact_id`

Required prep-reference failure families:

- `ANALYSIS_RUN_PREP_UNTRUSTED_CONTEXT`
- `ANALYSIS_RUN_PREP_NOT_FOUND`
- `ANALYSIS_RUN_PREP_SCHEMA_UNSUPPORTED`
- `ANALYSIS_RUN_PREP_MISMATCH`
- `ANALYSIS_RUN_PREP_STALE`

## Prep Fidelity Ladder (Condensed)

Prep-aware fidelity is tracked as an additive ladder:

- Tier 1: metadata/surrogate influence.
- Tier 2-2.5: topology-informed assembly and operator shaping.
- Tier 3-3.5: region/element contribution synthesis with connectivity scatter.
- Tier 4-4.5: prep-graph backbone assembly and solver co-design.
- Tier 5-5.5: calibration and acceptance harness with policy surfaces.
- Tier 6-7.5: evidence-driven drift control, recommendation artifacts, and CI/runbook hardening.

## Geometry Prep Artifact Lifecycle

`geometry.prep_for_analysis/v1` returns stable prep references and typed prep payloads used by analysis runs.

Primary store knob:

- `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT`

Retention knobs:

- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`
- `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`

Health/observability operation:

- `geometry.prep_artifact_health/v1` for count, age, and lifecycle counters.
