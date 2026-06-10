---
title: "Analysis Geometry"
category: "Analysis & Simulation"
section: "13.2"
last_updated: "June 9, 2026"
---

# Geometry

Geometry is the starting point for analysis. The first questions are usually simple:

- Can RunMat recognize this file?
- What regions or entities does it contain?
- Is the shape usable for numerical analysis?
- Should I create a prep artifact before solving?

RunMat answers those questions through geometry operations and optional prep artifacts.

## Bring Geometry In

Use `geometry.inspect/v1` when a host needs to identify a payload before loading it. Use `geometry.load/v1` to import bytes into a `GeometryAsset`.

From RunMat code, use the registered geometry builtins:

```matlab
info = geometry_inspect("geometry/bracket.stl");
geometry = geometry_load("geometry/bracket.stl");
```

`geometry_inspect` returns the detected format and byte-level inspection data. `geometry_load` returns the structured geometry asset that study files and analysis model creation use.

Supported importer paths currently include:

| Format family | Use |
| --- | --- |
| STL | Surface mesh input. |
| STEP | CAD exchange input. |
| OBJ | Mesh-style geometry input. |
| PLY | Mesh and point-oriented geometry input. |
| glTF | Scene or mesh-oriented geometry input. |

Format support covers ingestion. Region names, CAD semantics, and topology detail still depend on the importer and the source file. Check [Current Status](/docs/runtime/analysis/status) before assuming a specific file carries the detail your workflow needs.

## Inspect Geometry

After loading, hosts can ask:

| Question | Operation |
| --- | --- |
| What are the basic counts, extents, and mesh quality signals? | `geometry.compute_stats/v1` |
| What regions are known? | `geometry.list_regions/v1` |
| Which entities match a bounded query? | `geometry.query_entities/v1` |
| Can I capture a view for review? | `geometry.capture_view/v1` |

Inspection is useful before modeling because physics setup depends on region identity. Materials, loads, constraints, interfaces, and domain-specific inputs need stable places to attach.

## Prepare For Analysis

Use `geometry.prep_for_analysis/v1` when the next step needs analysis-ready geometry data. Prep produces a `prep_artifact_id` and a `MeshingPrepResult`.

Prep profiles:

| Profile | Use |
| --- | --- |
| `surface_only` | Keep surface-oriented data for surface workflows. |
| `analysis_ready` | Create deterministic baseline prep for analysis runs. |
| `adaptive_refine` | Add deterministic refinement-oriented quality and topology data. |

Prep artifacts can seed model creation and can be passed into run options. A prep-aware run validates that the artifact matches the model geometry id and revision before using it.

## What Prep Artifacts Protect

Prep-aware analysis runs check:

- the artifact exists,
- the schema is supported,
- the artifact came from the same geometry id,
- the artifact came from the same geometry revision,
- the artifact is not stale relative to stored prep artifacts,
- inline prep context is complete enough for the run.

Typed operation failures include `RM.ANALYSIS.RUN_PREP.NOT_FOUND`, `RM.ANALYSIS.RUN_PREP.SCHEMA_UNSUPPORTED`, `RM.ANALYSIS.RUN_PREP.MISMATCH`, `RM.ANALYSIS.RUN_PREP.STALE`, and `RM.ANALYSIS.RUN_PREP.UNTRUSTED_CONTEXT`.

## Artifact Lifecycle

Project runtime config is the preferred way to configure prep artifact storage and retention:

```toml
[runtime.analysis]
geometry_prep_artifact_root = "target/runmat-analysis-artifacts/geometry-prep"
geometry_prep_max_artifacts = 500
geometry_prep_max_artifacts_per_geometry = 20
geometry_prep_max_age_seconds = 2592000
geometry_prep_require_latest_revision = true
```

Environment variables remain supported as fallbacks:

| Environment variable | Purpose |
| --- | --- |
| `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT` | Geometry prep artifact root. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS` | Maximum total retained prep artifacts. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY` | Maximum retained prep artifacts per geometry id. |
| `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS` | Maximum retained artifact age. |
| `RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION` | Whether prep-aware runs reject stale revisions. |

Use `geometry.prep_artifact_health/v1` to inspect artifact counts, ages, lifecycle counters, stale or mismatch rejections, and optional per-geometry entries.

## Current Boundaries

- Geometry import paths exist for the listed format families, but format depth varies.
- CAD semantic richness, region naming, and topology detail depend on the importer and source file.
- Prep profiles are deterministic and saved as artifacts; meshing and adaptivity depth continue to expand.
- View capture uses the configured capture adapter and has fallback behavior.

For stage-by-stage support details, see [Current Status](/docs/runtime/analysis/status).
