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
