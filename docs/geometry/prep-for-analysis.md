# Geometry Prep For Analysis

`geometry.prep_for_analysis/v1` transforms imported geometry into deterministic analysis-prep artifacts.

## What it returns

- `prep_artifact_id`: stable reference used by analysis run operations.
- `prep`: typed prep payload (`geometry-prep-for-analysis/v1`) with:
  - prepared mesh descriptors,
  - region mapping metadata,
  - quality diagnostics (`min_scaled_jacobian`, `mean_aspect_ratio`, `inverted_element_count`),
  - topology hints per prepared mesh (`connectivity_class`, `element_family_hint`, `region_span_hint`).

Analysis runtime derives a deterministic topology profile from prep descriptors (surface/volume connectivity mix,
element-family mix, and mean region span) to shape assembly and operator coefficients.

Analysis runtime also derives deterministic region-block topology summaries from `region_mappings`
(block count and region mesh participation distribution) to drive region-local assembly partitioning
before full remeshed element assembly is available.

In Tier 3 core runs, prep descriptors also drive deterministic element-family contribution synthesis
for stiffness/mass/damping/right-hand-side accumulation, with replay-stable assembly fingerprints.

In Tier 3.5 runs, prep descriptors additionally drive deterministic element connectivity scatter
that shapes off-diagonal coupling structure and emits replay-stable connectivity fingerprints.

In Tier 4 runs, prep descriptors drive deterministic sparse graph backbone construction for
connectivity traversal, with replay-stable graph fingerprints and graph-structure diagnostics.

In Tier 4.5 runs, graph ordering and graph-conditioned preconditioner tuning are applied in
prep-aware solve paths and surfaced via graph-solver diagnostics.

In Tier 5 runs, deterministic calibration profiles and acceptance checks are applied to prep-aware
assembly/solve behavior, with replay-stable calibration and acceptance fingerprints.

In Tier 5.5 runs, calibration profile overrides can be selected from analysis run options and
prep acceptance-rate governance can be enforced in release-readiness checks.

In Tier 6 runs, calibration evidence artifacts define fixture/profile acceptance-score envelopes
used for drift detection and release-readiness policy checks.

In Tier 6.5 runs, evidence lifecycle validation and staleness controls are enforced, and
deterministic profile-shift recommendations are produced from drift behavior.

In Tier 7 runs, recommendation artifacts are promoted to first-class governance outputs and
evidence promotion from candidate to approved state is controlled by deterministic checks.

In Tier 7.5 runs, governance profiles and workflow-level artifact requirements are enforced in CI
and release preflight to prevent silent bypass of calibration governance.

## Prep artifact lifecycle

Prep artifacts are persisted and can be filesystem-backed using:

- `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT`

Retention policy knobs:

- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`
- `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`

## Health and observability

Use `geometry.prep_artifact_health/v1` for prep store telemetry:

- current artifact count,
- age p50/p95,
- lifecycle counters (`created`, `loaded`, `pruned`, stale/mismatch rejects),
- optional per-geometry distribution.

## Compatibility

- Prep artifact schema: `geometry_prep_artifact/v1`
- Prep payload schema: `geometry-prep-for-analysis/v1`

Clients should treat unknown additive fields as forward-compatible.
