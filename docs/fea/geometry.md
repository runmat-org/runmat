---
title: "Geometry"
category: "FEA"
section: "13.2"
last_updated: "June 10, 2026"
---

# Geometry

Geometry is the first object in the FEA workflow. Before a model can have materials, loads, constraints, or domains, RunMat needs a structured `geometry.Asset`.

## Load Or Inspect

From `.m` code:

```matlab
info = geometry.inspect("geometry/bracket.step");
geom = geometry.load("geometry/bracket.step");
```

From a `.fea` file:

```yaml
geometry:
  path: ../geometry/bracket.step
  units: millimeter
  import:
    max_triangles: 16000000
```

From host code, use `geometry.inspect/v1` when you need format and byte-level inspection before import. Use `geometry.load/v1` when you are ready to create a `GeometryAsset`.

## Supported Inputs

| Format family | Typical use | Notes |
| --- | --- | --- |
| STL | Surface mesh input | Good for mesh-only workflows. Region semantics are limited. |
| STEP | CAD exchange input | Can carry product labels and material evidence when present. |
| OBJ | Mesh-style geometry input | Useful for polygonal assets. |
| PLY | Mesh or point-oriented input | Useful for mesh/scan style data. |
| glTF | Scene or mesh input | Useful when geometry arrives as a scene asset. |

Format support means RunMat can ingest the file family. It does not guarantee that every file contains the region names, topology, or material evidence your study needs.

## Inspect Before Modeling

Use inspection when you need to answer:

| Question | Operation |
| --- | --- |
| What format is this file? | `geometry.inspect/v1` |
| What meshes and regions were imported? | `geometry.list_regions/v1` |
| What are the basic counts and statistics? | `geometry.compute_stats/v1` |
| Which entities match this bounded query? | `geometry.query_entities/v1` |
| Can I save visual evidence for review? | `geometry.capture_view/v1` |

This matters because model setup attaches physics data to geometry regions. A study is easier to review when the selected regions are visible and stable.

## Regions And Selectors

FEA materials, boundary conditions, loads, and interfaces attach to regions in the imported `geometry.Asset`. Region selectors are resolved against `geometry.regions`; they are not a separate FEA-only naming system.

Supported selectors are:

| Selector | Meaning |
| --- | --- |
| `region_1` | Direct region id. |
| `id:region_1` | Explicit region id. |
| `region:region_1` | Explicit region id. |
| `name:Base_Mount` | Region whose imported name is `Base_Mount`. |
| `tag:step_part` | First region with the imported tag `step_part`. |

Always inspect the loaded geometry before writing selectors:

```matlab
geom = geometry.load("geometry/bracket.step");
regions = geometry.listRegions(geom);
```

For STEP today, RunMat derives regions from STEP `PRODUCT` labels when they are present. If a STEP file does not contain explicit product labels, the importer creates a synthetic `region_1` for the model. That means CAD setup matters: use stable part, body, or product names in the CAD tool and preserve them during STEP export. Mesh-only formats often have weaker region metadata and may need a workflow-specific annotation step before they are useful for boundary conditions.

Face-level named selections are not the same thing as part-level product labels. If a study needs “fixed face” or “loaded face” semantics, verify that those names appear as imported regions before using them in `.fea` or `fea.boundaryCondition(...)`.

## Prepare Geometry

Use `geometry.prep_for_analysis/v1` when you need deterministic analysis-prep data before solving. Prep returns a `prep_artifact_id` and a `MeshingPrepResult`.

| Prep profile | Use |
| --- | --- |
| `surface_only` | Preserve surface-oriented data for surface workflows. |
| `analysis_ready` | Create deterministic baseline prep for analysis runs. |
| `adaptive_refine` | Create refinement-oriented quality and topology data. |

Prep artifacts are useful when later steps must prove which geometry revision, mappings, and prep data were used.

## Prep Artifact Checks

Prep-aware runs check that:

- the artifact exists,
- the schema is supported,
- the artifact belongs to the same geometry id,
- the artifact belongs to the same geometry revision,
- the artifact is not stale when latest-revision enforcement is enabled,
- inline prep context has enough data for the selected run.

Typed FEA prep failures use `RM.FEA.RUN_PREP.*` codes.

## Artifact Config

Configure prep artifact storage through `[runtime.fea]`:

```toml
[runtime.fea]
artifact_root = "artifacts"
geometry_prep_max_artifacts = 500
geometry_prep_max_artifacts_per_geometry = 20
geometry_prep_max_age_seconds = 2592000
geometry_prep_require_latest_revision = true
```

Use `geometry.prep_artifact_health/v1` to inspect retained artifact counts, ages, lifecycle counters, stale or mismatch rejections, and optional per-geometry entries.

## Boundaries

- Importer depth varies by file format and source data.
- STEP can expose useful product labels, but CAD semantic depth is not uniform.
- Mesh-only formats usually need additional region or boundary data from the workflow.
- Prep artifacts improve reproducibility; they do not prove that a physics model is well posed.

See [Current Status](/docs/fea/status) for current support by workflow stage.
