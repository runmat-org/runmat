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
  backend: cpu
  options:
    deterministic_mode: true
    precision_mode: fp64
    preconditioner_mode: auto
    quality_policy: balanced
```

Supported region selectors are direct region ids, `id:<region-id>`, `region:<region-id>`, `name:<region-name>`, and `tag:<tag>`. CAD imports commonly expose tags such as `occt_face`, `cad_body`, `cad_layer`, and `cad_material`; inspect the loaded geometry before relying on a selector.

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
      backend: cpu

  - version: 1
    id: bracket_steel
    geometry:
      path: ../geometry/bracket.stl
      units: meter
    model:
      profile: linear_static_structural
    run:
      backend: cpu
```

Sweeps execute sequentially in `v1`. `fail_fast: false` lets planning or running continue after a study-level failure and records failure entries in the sweep result.

## Run From `.m` Code

The RunMat-code API can express the same study content as `.fea`. Use typed FEA objects when the study is generated, parameterized, or mixed with normal numeric code.

```matlab
geom = geometry.load("geometry/bracket.step", "Units", "millimeter");
solverMesh = meshing.generate(geom, ...
    "TargetSize", 0.005, ...
    "MaximumDeviation", 0.00005, ...
    "ElementOrder", "tet10");

steel = fea.material("steel", ...
    "YoungsModulusPa", 200e9, ...
    "PoissonRatio", 0.30);

base = geometry.region(geom, "name:Base_Mount");
fixed = fea.StructuralConstraint.fixed("fixed_base", base);
loadPad = geometry.region(geom, "name:Tip_Load");
forceSignal = fea.ForceSignal.constantNewtons("tip_force_signal", [0 -1000 0]);
tipLoad = fea.StructuralExcitation.force( ...
    "tip_force", "service_loads", loadPad, forceSignal);
serviceLoads = fea.ExcitationCase.always("service_loads", {tipLoad});

model = fea.model("bracket_static_model", geom, ...
    "Profile", "linear_static_structural", ...
    "Materials", {steel}, ...
    "Constraints", {fixed}, ...
    "Signals", {forceSignal}, ...
    "Excitations", {tipLoad}, ...
    "ExcitationCases", {serviceLoads});

opts = fea.runOptions("linear_static", ...
    "DeterministicMode", true, ...
    "PrecisionMode", "fp64", ...
    "PreconditionerMode", "auto", ...
    "QualityPolicy", "balanced");

study = fea.study("bracket_static", geom, ...
    "Profile", "linear_static_structural", ...
    "Mesh", solverMesh, ...
    "Backend", "cpu", ...
    "Model", model, ...
    "RunOptions", opts);

validation = fea.validate(study);
plan = fea.plan(study);
result = fea.run(study);
results = fea.results(result);
stress = fea.field(results, "structural.von_mises");
mesh(results);
plot(stress);
print("bracket_von_mises.png", "-dpng");
```

Constraints and natural boundaries are separate typed values. Select a `geometry.Region`, then use the class for the relevant physics family. For example, `fea.StructuralConstraint.fixed(...)` constrains structural degrees of freedom, while `fea.StructuralBoundary.pressure(...)` applies a pressure operator. Thermal, electrical, magnetic, acoustic, and fluid families expose the same distinction where it is physically meaningful. Physical values must be created with dimensioned constructors such as `fea.Pressure.pascals(...)`, `fea.Temperature.kelvin(...)`, and `fea.HeatFlux.wattsPerSquareMeter(...)`; an untyped number cannot silently acquire a unit.

This thermal example builds a Robin boundary without embedding a unit or condition kind in text:

```matlab
wall = geometry.region(geom, "name:Cooling_Wall");
ambient = fea.Temperature.celsius(22);
h = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(18);
cooling = fea.ThermalBoundary.convection("ambient_cooling", wall, ambient, h);

thermalModel = fea.model("cooled_bracket", geom, ...
    "Template", "thermal_transient", ...
    "Boundaries", {cooling});
```

Use `Constraints` for prescribed values and kinematic relations, and `Boundaries` for natural or Robin operators. For fluid models, inlet velocity and wall motion are constraints; outlet pressure is a boundary. For thermal models, prescribed temperature is a constraint; heat flux and convection are boundaries.

The same host-only physical-value boundary applies to `fea.material`, the quantity-specific signal constructors, and the dimensioned quantity constructors. Domain constructors accept only the quantity classes required by their physics: for example, `fea.ElectromagneticDomain.atFrequency` requires `fea.Frequency`, and `fea.FluidDomain.newtonian` requires `fea.MassDensity`, `fea.DynamicViscosity`, and `fea.TurbulenceIntensity`. Bare numbers cannot silently acquire units. Numeric quantity inputs are checked before one binary64 model-storage conversion, so sufficiently wide integers can round. Numerical enforcement controls such as contact penalties and coupling relaxation belong to typed step or solver policy, not to interface physics. Geometry/model revisions, plan counts, field shape/count metadata, and comparison count deltas are structural integers; they remain exact and do not enter the physical-value conversion. The conventional plotting handle remains a double at the scripting boundary. Resident numeric constructor fields are rejected because typed FEA object construction performs no provider execution.

`fea.runOptions` distinguishes structural controls from physical solver settings. Mode, step, iteration, retry, backtrack, and tangent-refresh counts accept all eight integer classes and ordinary integral doubles exactly; tolerances, time steps, scales, ratios, frequencies, and residual thresholds cross one checked binary64 boundary. Logical settings require logical scalars. The complete per-family option/default table is in the [operation reference](./operation-reference).

`fea.results` uses positive one-based exact vectors for `ModeIndices` and `TransientSnapshotIndices`, matching normal RunMat indexing. Its inclusion flags accept logical scalars or exact numeric zero and one. Selector shape, positivity, integrality, and range are validated before artifact lookup; resident selectors reject without provider access. Structural result metadata such as shapes, counts, indexes, byte sizes, iteration counts, and fingerprints remains exact, while field values and physical/timing/residual quantities remain double. `fea.trends("WindowSize", n)` applies the same exact structural rule to a positive scalar and defaults to 16.

`fea.materialAssignment`, `fea.model`, `fea.plan`, `fea.field`, and `fea.compare` have no integer-array input role. Their inputs are exact text identifiers, typed FEA or geometry objects, or document paths, while any integer-valued metadata they return follows the structural-output rule. The ordinary `plot(...)` and `mesh(...)` functions dispatch FEA objects through their visualization methods without changing the numeric MATLAB-compatible forms. `fea.compare` accepts exactly the baseline and candidate run identifiers and has no Name, Value options.

Use `.fea` when the study definition should be checked in as a portable declarative artifact:

```matlab
study = fea.load("studies/bracket_static.fea");
validation = fea.validate(study);
result = fea.run(study);
results = result.results();
stress = result.field("structural.von_mises");
peakStressPa = max(stress.values);
```

`fea.load(...)` returns either a `fea.Study` or `fea.Sweep` object. `geometry.load(...)` returns a `geometry.Asset` object, and `meshing.generate(...)` returns a validated `meshing.Mesh`. The workflow builtins accept `.fea` paths, `fea.Study` objects, or `fea.Sweep` objects. Study operations return `fea.Validation`, `fea.Plan`, and `fea.RunResult`; sweep operations return the distinct aggregate classes `fea.SweepValidation`, `fea.SweepPlan`, and `fea.SweepRunResult`. A `fea.RunResult` exposes `results()`, `field(fieldId)`, and `mesh()` methods, while a sweep result exposes its `run_entries` for explicit selection. `fea.Field` objects expose script-friendly `values`, `shape`, `unit`, `location`, `kind`, `family`, and `quantity` properties plus `plot()` and `mesh()` visualization methods. Field selection belongs to `fea.field`; it is not a plotting option.

The typed constructors and post-processing helpers do not intrinsically require native OCCT. They can consume an existing `geometry.Asset`, STL-backed or synthetic geometry, and persisted mesh/result artifacts without OCCT; only an earlier import of topology-bearing CAD such as STEP, IGES, or BREP may require the configured OCCT geometry backend.

Argument decoding for `fea.run`, and artifact filtering for `fea.results` and `fea.trends`, is host-side. The solver invoked by `fea.run` is different: it may execute through the CPU or configured GPU backend selected by the study. Result queries return stored device references as metadata and do not gather or execute them.

The ordinary `mesh` function displays geometry and solver-mesh objects directly:

```matlab
asset = geometry.load("geometry/bracket.step", "Units", "millimeter");
mesh(asset);

solverMesh = meshing.generate(asset, "TargetSize", 0.005);
mesh(solverMesh, "Edges", true);
```

For low-level custom visualization, `geometry.meshes(asset)` returns patch-ready surface topology:

```matlab
asset = geometry.load("geometry/bracket.step", "Units", "millimeter");
meshes = geometry.meshes(asset);
m = meshes{1};
patch("Faces", m.faces, "Vertices", m.vertices);
```

The typed constructors are:

| Builtin | Use |
| --- | --- |
| `fea.material(...)` | Mechanical, thermal, electrical, and plastic material data. |
| `fea.materialAssignment(...)` | Region-to-material assignment. |
| `fea.<Family>Constraint.<kind>(...)` | Essential, kinematic, gauge, pairing, and reference constraints bound to a `geometry.Region`. |
| `fea.<Family>Boundary.<kind>(...)` | Natural and Robin boundary operators such as pressure, heat flux, convection, radiation, impedance, and outlet pressure. |
| `fea.<Quantity>Signal.constant<Unit>(...)` | A constant, quantity-typed physical signal in the unit named by the constructor. |
| `fea.<Family>Excitation.<action>(...)` | A typed spatial action binding a compatible signal to a `geometry.Region`. |
| `fea.ExcitationCase.always(...)`, `.duringSteps(...)`, `.mutuallyExclusive(...)` | Group typed actions and declare their activation without changing either the signal or spatial action. |
| `fea.<Family>Step.<procedure>(id)` | A concrete direct-solution procedure step; family and procedure are selected by the class method rather than text. |
| `fea.ThermoMechanicalDomain.atReferenceTemperature(...)` | Thermo-mechanical constitutive reference state from a typed `fea.Temperature`. |
| `fea.ElectroThermalDomain.atReferenceTemperature(...)` | Electro-thermal constitutive reference state from a typed `fea.Temperature`. |
| `fea.ElectromagneticDomain.atFrequency(...)` | Electromagnetic frequency-domain definition from a typed `fea.Frequency`. |
| `fea.FluidDomain.newtonian(...)` | Newtonian fluid properties from typed density, viscosity, and turbulence intensity values. |
| `fea.ContactInterface.frictionless(...)`, `fea.ContactInterface.coulomb(...)` | Contact between two typed regions, optionally with typed Coulomb friction. |
| `fea.FluidStructureInterface.noSlip(...)` | No-slip fluid-to-structure transfer across two typed regions. |
| `fea.ConjugateHeatTransferInterface.perfect(...)`, `.conductance(...)`, `.resistance(...)` | Fluid-to-solid heat transfer with exactly one physical transfer law. |
| `fea.runOptions(...)` | Family-specific solver and quality options. |
| `fea.model(...)` | Explicit model assembled from typed components. |
| `fea.study(...)` | Study assembled from geometry, physics profile/model data, backend, and run options. |
| `fea.sweep(...)` | Study sweep assembled from `fea.Study` objects. |
| `plot(field, ...)` | Plot an `fea.Field` through the ordinary plotting function. |
| `mesh(runOrResultsOrField, ...)` | Display solver topology, CAD geometry, or both through the ordinary mesh function. |

Resolve typed region arguments with `geometry.region`. Its selector accepts a direct region id, `id:<region-id>`, `region:<region-id>`, `name:<region-name>`, or `tag:<tag>`. Constraints, boundaries, and interfaces then retain the resolved geometry identity and revision instead of an unqualified selector string.

Moment excitations are structural load resultants in N*m. Construct the vector with
`fea.MomentSignal.constantNewtonMeters`, then bind it to a region with
`fea.StructuralExcitation.moment`. Direct moments require structural elements with rotational
degrees of freedom, such as beams or shells. Solid-only displacement regions reject them because
continuum solids do not own nodal rotations.

Signals, spatial actions, and activation cases are deliberately separate objects. This keeps an
ordinary RunMat-generated history reusable across regions and cases, while a spatial action keeps
its geometry revision and projection policy. A constant structural force is authored as follows:

```matlab
loadPad = geometry.region(geom, "name:load_pad");
force = fea.ForceSignal.constantNewtons("tip_force_signal", [0 -1250 0]);
tipLoad = fea.StructuralExcitation.force( ...
    "tip_force", "service_loads", loadPad, force);
serviceLoads = fea.ExcitationCase.always("service_loads", {tipLoad});

model = fea.model("bracket", geom, ...
    "Template", "linear_static_structural", ...
    "Signals", {force}, ...
    "Excitations", {tipLoad}, ...
    "ExcitationCases", {serviceLoads});
```

```matlab
tipNode = geometry.region(geom, "id:node:2");
momentSignal = fea.MomentSignal.constantNewtonMeters( ...
    "tip_moment_signal", [0 0 125]);
tipMoment = fea.StructuralExcitation.moment( ...
    "tip_moment", "torsion_case", tipNode, momentSignal);
torsionCase = fea.ExcitationCase.always("torsion_case", {tipMoment});
```

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
```

When `artifact_root` is omitted, RunMat uses `./artifacts`. Run artifacts are written under `runs`, study evidence under `studies`, and thermo-field artifacts under `thermo-fields`. Override the specific roots only when a host needs a different layout.

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
