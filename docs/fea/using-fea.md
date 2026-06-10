---
title: "Using FEA"
category: "FEA"
section: "13.1"
last_updated: "June 10, 2026"
---

# Using FEA

FEA workflows in RunMat start with a geometry file and end with a run result you can inspect or store.

Use one of these entry points:

| Entry point | Best for |
| --- | --- |
| `runmat check <file>` | Checking a `.m` script or `.fea` study before running it. |
| `runmat run <file>` | Running either a `.m` script or a `.fea` study/sweep. |
| Named project entrypoint | Giving a stable name to a `.m` file or `.fea` file in `runmat.toml`. |
| RunMat code | Loading geometry, building typed FEA studies, or loading `.fea` files from `.m`. |
| Rust/runtime operations | Host integrations that need low-level control over models, operations, artifacts, or result queries. |

## Run A `.fea` File

A `.fea` file is a YAML file that defines a study or parametric sweep.

```sh
runmat check studies/bracket_static.fea
runmat run studies/bracket_static.fea
```

`runmat check` loads geometry, resolves the study, validates it, and plans it. It does not solve. `runmat run` executes the study or sweep and prints a run summary by default.

`.fea` files use the same `runmat run` command as `.m` files. Positional script arguments and `--emit-bytecode` apply only to `.m` files.

By default, FEA run output is human-readable and includes the `run_id`, status, quality gates, evidence path, and a `fea.results("<run_id>")` hint for post-processing. Use `runmat run --json studies/bracket_static.fea` when a script or CI job needs structured JSON. Use `runmat check --json studies/bracket_static.fea` for structured validation and plan output.

## Minimal Study File

This is the smallest useful `.fea` study. The geometry path is resolved relative to the `.fea` file.

```yaml
version: 1
kind: study
id: bracket_static

geometry:
  path: ../geometry/bracket.stl
  units: meter

model:
  profile: linear_static_structural

run:
  kind: linear_static
  backend: cpu
```

The default model mode is `profile_scaffold`. RunMat loads the geometry and creates a scaffold model from the selected profile during validation, planning, and running.

## Full Study Shape

Use explicit fields when the study file should carry model data directly:

```yaml
version: 1
kind: study
id: bracket_static

geometry:
  path: ../geometry/bracket.step
  units: millimeter
  import:
    max_triangles: 16000000

model:
  id: bracket_static_model
  profile: linear_static_structural
  defaults: none
  frame: global

regions:
  fixed_base:
    selector: "name:Base_Mount"
  load_face:
    selector: "name:Tip_Load"

materials:
  aluminum_6061:
    name: Aluminum 6061
    mechanical:
      youngs_modulus_pa: 69000000000.0
      poisson_ratio: 0.33

material_assignments:
  - region: fixed_base
    material: aluminum_6061
  - region: load_face
    material: aluminum_6061

boundary_conditions:
  - id: fixed_base
    region: fixed_base
    kind: fixed

loads:
  - id: tip_force
    region: load_face
    type: force
    vector: [0.0, -1000.0, 0.0]

steps:
  - id: static_step
    kind: static

run:
  kind: linear_static
  backend: cpu
  options:
    deterministic_mode: true
    precision_mode: fp64
    preconditioner_mode: auto
    quality_policy: balanced
```

Supported region selectors are direct region ids, `id:<region-id>`, `region:<region-id>`, `name:<region-name>`, and `tag:<tag>`.

## Sweep File

Use a sweep when several studies should be checked or run as one deterministic sequence. Nested studies use the same study fields but do not include `kind`.

```yaml
version: 1
kind: sweep
id: bracket_material_sweep
fail_fast: true

studies:
  - version: 1
    id: bracket_aluminum
    geometry:
      path: ../geometry/bracket.stl
      units: meter
    model:
      profile: linear_static_structural
    run:
      kind: linear_static
      backend: cpu

  - version: 1
    id: bracket_steel
    geometry:
      path: ../geometry/bracket.stl
      units: meter
    model:
      profile: linear_static_structural
    run:
      kind: linear_static
      backend: cpu
```

Sweeps execute sequentially in `v1`. `fail_fast: false` lets planning or running continue after a study-level failure and records failure entries in the sweep result.

## Run From `.m` Code

The RunMat-code API can express the same study content as `.fea`. Use typed FEA objects when the study is generated, parameterized, or mixed with normal numeric code.

```matlab
geom = geometry.load("geometry/bracket.step");

steel = fea.material("steel", ...
    "YoungsModulusPa", 200e9, ...
    "PoissonRatio", 0.30);

fixed = fea.boundaryCondition("fixed_base", "name:Base_Mount", "fixed");
tipLoad = fea.loadCase("tip_force", "name:Tip_Load", "force", ...
    "Vector", [0 -1000 0]);

model = fea.model("bracket_static_model", geom, ...
    "Profile", "linear_static_structural", ...
    "Materials", {steel}, ...
    "BoundaryConditions", {fixed}, ...
    "Loads", {tipLoad});

opts = fea.runOptions("linear_static", ...
    "DeterministicMode", true, ...
    "PrecisionMode", "fp64", ...
    "PreconditionerMode", "auto", ...
    "QualityPolicy", "balanced");

study = fea.study("bracket_static", geom, ...
    "RunKind", "linear_static", ...
    "Backend", "cpu", ...
    "Model", model, ...
    "RunOptions", opts);

validation = fea.validate(study);
plan = fea.plan(study);
result = fea.run(study);
results = fea.results(result);
stress = fea.field(results, "von_mises");
```

Use `.fea` when the study definition should be checked in as a portable declarative artifact:

```matlab
study = fea.load("studies/bracket_static.fea");
validation = fea.validate(study);
result = fea.run(study);
results = result.results();
```

`fea.load(...)` returns either a `fea.Study` or `fea.Sweep` object. `geometry.load(...)` returns a `geometry.Asset` object. The workflow builtins accept `.fea` paths, `fea.Study` objects, or `fea.Sweep` objects. `fea.RunResult` objects expose `results()` and `field(fieldId)` methods.

The typed constructors are:

| Builtin | Use |
| --- | --- |
| `fea.material(...)` | Mechanical, thermal, electrical, and plastic material data. |
| `fea.materialAssignment(...)` | Region-to-material assignment. |
| `fea.boundaryCondition(...)` | Constraints such as `fixed` or EM boundary kinds. |
| `fea.loadCase(...)` | Force, pressure, body force, current density, or coil current. |
| `fea.step(...)` | Analysis step. |
| `fea.domain(...)` | Thermo-mechanical, electro-thermal, electromagnetic, or CFD domain data. |
| `fea.interface(...)` | Contact interfaces. |
| `fea.runOptions(...)` | Family-specific solver and quality options. |
| `fea.model(...)` | Explicit model assembled from typed components. |
| `fea.study(...)` | Study assembled from geometry, model, backend, run kind, and run options. |
| `fea.sweep(...)` | Study sweep assembled from `fea.Study` objects. |

Region strings in these constructors use the same selector grammar as `.fea`: direct region id, `id:<region-id>`, `region:<region-id>`, `name:<region-name>`, or `tag:<tag>`.

## Project Entrypoints

Named entrypoints can target `.m` or `.fea` files:

```toml
[sources]
roots = ["src"]

[entrypoints.bracket_static]
path = "studies/bracket_static.fea"
```

Run it with the same command:

```sh
runmat run bracket_static
```

## Runtime Config

Use `[runtime.fea]` for FEA artifacts:

```toml
[runtime.fea]
artifact_store = "filesystem"
artifact_root = "artifacts"
geometry_prep_require_latest_revision = true
```

When `artifact_root` is omitted, RunMat uses `./artifacts`. Run artifacts are written under `runs`, study evidence under `studies`, geometry prep under `geometry-prep`, and thermo-field artifacts under `thermo-fields`. Override the specific roots only when a host needs a different layout.

## Rust Host Usage

Rust hosts call runtime operations when they need the operation envelope, trace ids, direct model construction, or result queries:

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
        model_id: "bracket_static_model".to_string(),
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

let run = analysis_run_linear_static_op(&model, ComputeBackend::Cpu, context)?.data;
```

The public operation identifiers in envelopes are `fea.*`, such as `fea.create_model/v1` and `fea.run_linear_static/v1`.
