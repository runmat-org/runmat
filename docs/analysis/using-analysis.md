---
title: "Using Analysis"
category: "Analysis & Simulation"
section: "13.1"
last_updated: "June 9, 2026"
---

# Using Analysis

RunMat analysis is driven through runtime operations. The current implementation supports two setup styles:

1. Direct operation calls from a host or runtime integration.
2. Serializable study and sweep specs that a host can validate, plan, and run.

The same runtime operations execute both paths. A study spec packages the direct operation sequence into a repeatable unit with a fingerprint and evidence artifacts.

## Entry Points

| Entry point | Current state | Use when |
| --- | --- | --- |
| Direct runtime operations | Implemented | A host wants explicit control over geometry loading, model creation, validation, run options, and result queries. |
| Study and sweep specs | Implemented as `serde` contracts plus runtime operations | A host wants a reusable analysis definition, deterministic planning, repeatable execution, and evidence artifacts. |
| RunMat-language wrappers | Not registered as builtins by the current runtime | A host can add wrappers over the runtime operations when analysis should be called directly from `.m` code. |
| Dedicated study-file command | The current CLI does not include a dedicated analysis or study command | A host command can load a serialized study spec and call the study operations. |

## Direct Operation Flow

Use direct operations when the caller is orchestrating each step:

1. Read geometry bytes from the host filesystem, project store, upload, or remote source.
2. Call `geometry.inspect/v1` to detect the format and byte count.
3. Call `geometry.load/v1` to produce a `GeometryAsset`.
4. Call geometry inspection operations as needed: stats, regions, entity queries, view capture.
5. Optionally call `geometry.prep_for_analysis/v1` and keep the returned `prep_artifact_id`.
6. Call `analysis.create_model/v1` with an `AnalysisCreateModelIntentSpec`.
7. Call `analysis.validate/v1`.
8. Call the matching `analysis.run_*/v1` operation.
9. Call `analysis.results/v1`, `analysis.results_compare/v1`, or `analysis.trends/v1`.

The operation-envelope API is the integration contract. It returns stable `operation`, `op_version`, optional trace/request ids, typed data, and typed errors.

Rust hosts call the public runtime functions directly:

```rust
use runmat_analysis_fea::ComputeBackend;
use runmat_runtime::analysis::{
    analysis_create_model_op, analysis_run_linear_static_op, analysis_validate,
    AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
};
use runmat_runtime::geometry::geometry_load_op;
use runmat_runtime::operations::OperationContext;

let context = OperationContext::new(Some("trace-001".to_string()), None);
let geometry = geometry_load_op("bracket.step", &bytes, context.clone())?.data;

let model = analysis_create_model_op(
    &geometry,
    AnalysisCreateModelIntentSpec {
        model_id: "bracket_static_001".to_string(),
        profile: AnalysisCreateModelProfile::LinearStaticStructural,
        prep_context: None,
    },
    context.clone(),
)?.data;

analysis_validate(
    &model,
    geometry.units,
    &runmat_analysis_core::ReferenceFrame::Global,
    context.clone(),
)?;

let run = analysis_run_linear_static_op(&model, ComputeBackend::Cpu, context.clone())?.data;
```

Geometry also has convenience helpers such as `geometry_load`, `geometry_compute_stats`, `geometry_list_regions`, `geometry_query_entities`, `geometry_capture_view`, and `geometry_prep_for_analysis`. Those helpers return data or runtime errors without exposing the full operation envelope.

## Study Specs

Use a study when a single analysis should be reusable and evidence-backed. A study contains:

| Field | Meaning |
| --- | --- |
| `study_id` | Stable caller-provided id for the study. |
| `geometry` | A normalized `GeometryAsset`. The current contract embeds geometry data, not a raw file path. |
| `create_model_intent` | Model id, profile, and optional prep context. |
| `run_kind` | Physics run family such as `linear_static`, `modal`, `thermal`, `nonlinear`, or `electromagnetic`. |
| `backend` | Requested compute backend: `cpu` or `gpu`. |
| `electromagnetic_run_options` | Optional EM-specific run controls; valid only when `run_kind` is `electromagnetic`. |

A host that wants author-friendly study files can accept file paths in its own format, call `geometry.load/v1`, and write the resulting `GeometryAsset` into the runtime study spec before validation or execution. The runtime contract starts at the normalized study spec; path-based files are a host-level layer over that contract.

Minimal normalized study shape:

```json
{
  "study_id": "bracket_static_001",
  "geometry": {
    "geometry_id": "bracket_geo_001",
    "source": {
      "path": "bracket.step",
      "sha256": "sha256:...",
      "importer_version": "runmat-geometry-io/v1"
    },
    "source_geometry": {
      "kind": "cad",
      "assembly": null,
      "material_evidence": []
    },
    "tessellation_profile": {
      "profile_id": "default-v1",
      "chord_tolerance": null,
      "angle_tolerance_deg": null,
      "healing_mode": "safe"
    },
    "units": "millimeter",
    "revision": 1,
    "meshes": [
      {
        "mesh_id": "mesh_root",
        "kind": "surface",
        "vertex_count": 1024,
        "element_count": 2048
      }
    ],
    "regions": [
      {
        "region_id": "region_root",
        "name": "Bracket",
        "tag": null
      }
    ],
    "diagnostics": []
  },
  "create_model_intent": {
    "model_id": "bracket_model_static_001",
    "profile": "linear_static_structural"
  },
  "run_kind": "linear_static",
  "backend": "cpu"
}
```

The runtime study operations are:

| Operation | Result |
| --- | --- |
| `analysis.validate_study/v1` | Validity, issue codes, structured issues, and validation evidence path. |
| `analysis.plan_study/v1` | Operation sequence, run operation/version, study fingerprint, and plan evidence path. |
| `analysis.run_study/v1` | Created model, selected run operation, persisted run id, quality, provenance, and run evidence path. |

Study evidence is written under `RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT`. If that environment variable is unset, the runtime uses `target/runmat-analysis-artifacts/studies`.

## Sweeps

Use a sweep when several studies should be validated, planned, or run as one deterministic sequence.

```json
{
  "sweep_id": "bracket_material_sweep_001",
  "fail_fast": true,
  "studies": [
    { "...": "first AnalysisStudySpec" },
    { "...": "second AnalysisStudySpec" }
  ]
}
```

Sweep operations are:

| Operation | Result |
| --- | --- |
| `analysis.validate_study_sweep/v1` | Aggregate validity plus per-study issue entries. |
| `analysis.plan_study_sweep/v1` | Per-study plan entries and failure entries. |
| `analysis.run_study_sweep/v1` | Per-study run entries, failure entries, and sweep evidence path. |

Sweeps execute sequentially in `v1`. Hosts that need parallel execution should run separate sessions or processes while preserving artifact roots and trace ids.

## Profile And Run-Kind Pairing

Study validation rejects mismatched profiles and run kinds:

| `create_model_intent.profile` | Compatible `run_kind` |
| --- | --- |
| `linear_static_structural` | `linear_static` |
| `modal_structural` | `modal` |
| `acoustic_harmonic` | `acoustic` |
| `thermal_standalone` | `thermal` |
| `transient_structural` | `transient` |
| `thermo_mechanical_coupled` | `transient` |
| `nonlinear_structural` | `nonlinear` |
| `electromagnetic_static` | `electromagnetic` |
| `cfd_steady_state`, `cfd_transient` | `cfd` |
| `cht_coupled` | `thermal`, `cfd`, or `cht` |
| `fsi_coupled` | `transient`, `cfd`, or `fsi` |

For the complete operation inventory, see [Operation Reference](/docs/runtime/analysis/operation-reference). For result interpretation, see [Results & Trust](/docs/runtime/analysis/trust).
