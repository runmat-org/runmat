---
title: "Geometry"
category: "FEA"
section: "13.2"
last_updated: "June 22, 2026"
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

## Supported Inputs

| Format family | Typical use | Notes |
| --- | --- | --- |
| STL | Surface mesh input | Good for mesh-only workflows. Region semantics are limited. |
| STEP | CAD exchange input | Source builds without OCCT read STEP metadata. OCCT-enabled builds import topology, tessellate faces, and map XCAF assembly, label, layer, color, and material ownership to selectable regions when the file carries that data. |
| IGES | CAD exchange input | Requires the OCCT CAD backend. Imports topology as tessellated face regions and maps XCAF names, layers, and colors when available. |
| BREP | Native OCCT B-rep input | Requires the OCCT CAD backend. Imports topology as tessellated face regions. BREP files do not normally carry STEP-style exchange metadata. |
| OBJ | Mesh-style geometry input | Useful for polygonal assets. |
| PLY | Mesh or point-oriented input | Useful for mesh/scan style data. |
| glTF | Scene or mesh input | Useful when geometry arrives as a scene asset. |

Format support means RunMat can ingest the file family. It does not guarantee that every file contains the region names, topology, or material evidence your study needs.

Source builds that do not enable the optional OCCT feature do not link OCCT. In those builds, STEP remains useful for assembly, product-label, and material-evidence metadata, but it does not produce CAD face topology. Official RunMat CLI and desktop binaries are built with OCCT enabled, so they can import STEP, IGES, and BREP topology through OCCT. Browser/WASM builds use the `occt-wasm-host` feature with a configured browser sidecar.

Native packagers can build the OCCT path from the bundled OCCT source or point RunMat at an existing OCCT installation with `RUNMAT_OCCT_ROOT`, or with `RUNMAT_OCCT_INCLUDE_DIR` and `RUNMAT_OCCT_LIB_DIR`. `RUNMAT_OCCT_ROOT` accepts both `include/` and `include/opencascade/` header layouts. Use `RUNMAT_OCCT_LINK_MODE=static` for static CLI-style linking. Use `RUNMAT_OCCT_LINK_MODE=dylib` for dynamic desktop packaging; dynamic mode requires an existing dynamic OCCT installation and does not use the bundled static OCCT builder.

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
| `tag:occt_face` | First face region produced by the OCCT CAD backend. |
| `tag:cad_body` | First body-level semantic region mapped from CAD ownership metadata. |
| `tag:cad_layer` | First layer-level semantic region mapped from CAD ownership metadata. |
| `tag:cad_material` | First material-level semantic region mapped from CAD ownership metadata. |

Always inspect the loaded geometry before writing selectors:

```matlab
geom = geometry.load("geometry/bracket.step");
regions = geometry.listRegions(geom);
```

Without OCCT, STEP regions are derived from STEP `PRODUCT` labels when they are present. If a STEP file does not contain explicit product labels, the metadata importer creates a synthetic `region_1` for the model.

With the OCCT CAD backend, STEP, IGES, and BREP imports produce tessellated face regions such as `face_000001` with the tag `occt_face`. Those regions map directly to imported mesh faces, so they can be picked, highlighted, and used by `.fea` selectors.

STEP and IGES imports also use the OCCT XCAF document model. When the file contains usable ownership metadata, RunMat creates additional semantic regions from CAD labels, bodies, components, assemblies, layers, colors, and materials. These regions use ids such as `cad_label_0_1_1`, `cad_layer_boundary_faces`, or `cad_material_aluminum_6061`, and tags such as `cad_body`, `cad_component`, `cad_assembly`, `cad_layer`, `cad_color`, and `cad_material`.

Semantic CAD regions are evidence from the imported exchange file. If a CAD file does not carry layer names, material assignments, color assignments, or meaningful product/body labels, RunMat still imports topology and face regions, but it cannot invent authored boundary names.

Mesh-only formats often have weaker region metadata and may need a workflow-specific annotation step before they are useful for boundary conditions.

## Prepare Geometry

Use `geometry.prep_for_analysis/v1` when you need deterministic analysis-prep data before solving. Prep returns a `prep_artifact_id` and a `MeshingPrepResult`.

| Prep profile | Use |
| --- | --- |
| `surface_only` | Preserve surface-oriented data for surface workflows. |
| `analysis_ready` | Create deterministic baseline prep for analysis runs. |
| `adaptive_refine` | Create refinement-oriented quality and topology data. |

Prep artifacts are useful when later steps must prove which geometry revision, mappings, and prep data were used.

Prepared analysis artifacts carry both bounded compatibility samples and full element topology vectors. The full vectors include node coordinates, edge-node incidence, element-edge incidence/orientation, and element areas. Prep-aware EM, electro-thermal, CHT, and FSI paths use those vectors for edge or interface graph construction where available, and conformance gates assert that governed prepared fixtures are not satisfied by an implicit line or diagnostic-only topology fallback.

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

- STEP and IGES can expose useful product, body, layer, color, and material ownership, but the quality of that ownership depends on the exported CAD data.
- OCCT-enabled CAD import provides face-level topology for STEP, IGES, and BREP, plus semantic regions for STEP and IGES when XCAF ownership metadata is present.
- Builds without OCCT enabled can load STEP metadata but not face-level topology.
- Mesh-only formats usually need additional region or boundary data from the workflow.
- Prep artifacts improve reproducibility; they do not prove that a physics model is well posed.

See [Current Status](/docs/fea/status) for current support by workflow stage.
