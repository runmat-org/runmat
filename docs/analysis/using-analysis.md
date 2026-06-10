---
title: "Using Analysis"
category: "Analysis & Simulation"
section: "13.1"
last_updated: "June 9, 2026"
---

# Using Analysis

Most analysis work starts with a geometry file and ends with a run result you can inspect, compare, or store.

A typical study workflow is:

1. Load or inspect geometry.
2. Build a study file that embeds the normalized geometry asset.
3. Validate or plan the study.
4. Run the study from the CLI, a named project entrypoint, or RunMat code.
5. Inspect the result payload, saved artifacts, diagnostics, and provenance.

## Choose An Entry Point

| Entry point | Use when |
| --- | --- |
| `runmat run <study-file>` | You already have a study or sweep file and want to run it from a terminal or CI job. |
| Named project entrypoint | You want a stable project target such as `runmat run bracket_static`. |
| RunMat code builtins | You want `.m` code to inspect geometry or run an existing study file. |
| Runtime operations | A Rust or host integration needs direct control over geometry loading, model construction, run options, and result queries. |

Study files use `.study.json`, `.study.yaml`, `.study.yml`, or `.study`. JSON and YAML study files can contain either one `AnalysisStudySpec` or one `AnalysisStudySweepSpec`.

## Start With Geometry

From RunMat code, inspect or load a geometry file:

```matlab
info = geometry_inspect("geometry/bracket.stl");
geometry = geometry_load("geometry/bracket.stl");
```

`geometry_inspect` reports the detected format and basic file details. `geometry_load` returns the normalized geometry asset that study specs use.

Host integrations can call `geometry.inspect/v1` and `geometry.load/v1` directly when geometry comes from an upload, remote source, project store, or preprocessing pipeline.

## Create A Study File

A study file describes one repeatable analysis. It contains:

| Field | Meaning |
| --- | --- |
| `study_id` | Stable caller-provided id for the study. |
| `geometry` | A normalized `GeometryAsset`. |
| `create_model_intent` | Model id, profile, and optional prep context. |
| `run_kind` | Physics run family such as `linear_static`, `modal`, `thermal`, `nonlinear`, or `electromagnetic`. |
| `backend` | Requested compute backend: `cpu` or `gpu`. |
| `electromagnetic_run_options` | Optional electromagnetic controls; valid only when `run_kind` is `electromagnetic`. |

The runtime study contract contains the normalized geometry asset. Pipelines that start with raw CAD or mesh files should load geometry first, then write the resulting asset into the study file.

Compact YAML example:

```yaml
study_id: bracket_static_001
geometry:
  geometry_id: bracket_geo_001
  source:
    path: geometry/bracket.step
    sha256: "sha256:..."
    importer_version: runmat-geometry-io/v1
  source_geometry: { kind: cad, assembly: null, material_evidence: [] }
  tessellation_profile:
    profile_id: default-v1
    chord_tolerance: null
    angle_tolerance_deg: null
    healing_mode: safe
  units: millimeter
  revision: 1
  meshes:
    - mesh_id: mesh_root
      kind: surface
      vertex_count: 1024
      element_count: 2048
  regions:
    - region_id: region_root
      name: Bracket
      tag: null
  diagnostics: []
create_model_intent:
  model_id: bracket_model_static_001
  profile: linear_static_structural
run_kind: linear_static
backend: cpu
```

## Validate, Plan, And Run

Study operations write artifacts at each stage:

| Operation | Result |
| --- | --- |
| `analysis.validate_study/v1` | Validity, issue codes, structured issues, and study-validation artifact path. |
| `analysis.plan_study/v1` | Operation sequence, run operation/version, study fingerprint, and plan artifact path. |
| `analysis.run_study/v1` | Created model, selected run operation, saved run id, quality, provenance, and run artifact path. |

For how issue codes, typed operation errors, diagnostics, quality reasons, and artifacts fit together, see [Evidence & Artifacts](/docs/runtime/analysis/evidence).

Run a study file directly:

```sh
runmat run studies/bracket.study.yaml
```

The CLI prints operation data as JSON. With `runtime.verbose = true`, it prints the full operation envelope. Study files do not accept positional script arguments. `--emit-bytecode` is only for `.m` files.

## Add A Project Entrypoint

Define a named entrypoint in `runmat.toml`:

```toml
[sources]
roots = ["src"]

[entrypoints.bracket_static]
path = "studies/bracket.study.yaml"
```

Then run it like any other project target:

```sh
runmat run bracket_static
```

Named entrypoints can target either `.m` files or study files.

## Run From RunMat Code

RunMat code can call the same study file:

```matlab
validation = analysis_validate_study("studies/bracket.study.yaml");
plan = analysis_plan_study("studies/bracket.study.yaml");
run = analysis_run_study("studies/bracket.study.yaml");
```

The analysis study builtins accept either a single-study file or a sweep file and return the validate, plan, or run payload as a RunMat struct.

Use these builtins when `.m` code needs a simple analysis workflow. Use runtime operations when a host needs lower-level model construction, domain-specific options, result queries, or custom orchestration.

## Run A Sweep

Use a sweep when several studies should be validated, planned, or run as one deterministic sequence.

```yaml
sweep_id: bracket_material_sweep_001
fail_fast: true
studies:
  - study_id: bracket_static_aluminum
    geometry: { ... }
    create_model_intent: { ... }
    run_kind: linear_static
    backend: cpu
  - study_id: bracket_static_steel
    geometry: { ... }
    create_model_intent: { ... }
    run_kind: linear_static
    backend: cpu
```

Sweep operations are:

| Operation | Result |
| --- | --- |
| `analysis.validate_study_sweep/v1` | Aggregate validity plus per-study issue entries. |
| `analysis.plan_study_sweep/v1` | Per-study plan entries and failure entries. |
| `analysis.run_study_sweep/v1` | Per-study run entries, failure entries, and sweep artifact path. |

Sweeps execute sequentially in `v1`.

## Configure Artifacts

Project runtime config is the preferred way to configure analysis artifacts:

```toml
[runtime.analysis]
artifact_store = "filesystem"
artifact_root = "target/runmat-analysis-store"
study_artifact_root = "target/runmat-analysis-artifacts/studies"
geometry_prep_artifact_root = "target/runmat-analysis-artifacts/geometry-prep"
thermo_field_artifact_root = "target/runmat-analysis-artifacts/thermo-fields"
geometry_prep_require_latest_revision = true
```

Environment variables remain supported as fallbacks for existing integrations. See [Configuration Reference](/docs/runtime/getting-started/config) and [Operation Reference](/docs/runtime/analysis/operation-reference) for the full list.

## Match Profile To Run Kind

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

## Use Runtime Operations From Rust

Rust hosts call the public runtime functions directly when they need the full operation envelope:

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

For the complete operation inventory, see [Operation Reference](/docs/runtime/analysis/operation-reference). For result interpretation, see [Results & Trust](/docs/runtime/analysis/trust).
