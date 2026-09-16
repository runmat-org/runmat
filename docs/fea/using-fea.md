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

`.fea` files use the same `runmat run` command as `.m` files, but neither file form accepts
positional script arguments. RunMat does not invent non-MATLAB `argv` semantics for scripts.
Remote typed function entrypoints accept declared string inputs through
`runmat job submit --function NAME -- ...`. `--emit-bytecode` applies only to `.m` files.

By default, FEA run output is human-readable and includes each procedure step's `run_id`, status,
quality gates, evidence path, and a `fea.results("<run_id>")` hint for post-processing. Use
`runmat run --json studies/bracket_static.fea` when a script or CI job needs structured JSON. Use
`runmat check --json studies/bracket_static.fea` for structured validation and plan output.

## Smallest Runnable Study

The repository includes a complete typed static study and its geometry fixture:

```sh
runmat check docs/fea/examples/structural-static-experimental.fea
runmat run docs/fea/examples/structural-static-experimental.fea
```

The file is intentionally explicit. A study contains a typed model template, authored physical
objects, one semantic `sequence`, and an `execution` entry for each step that will run. The
sequence owns physics and dataflow; execution owns backend, formulation requirement, and solver
options. Neither is inferred from a generic run kind.

The top-level shape is:

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
  template: linear_static_structural
  defaults: none
  frame: global

regions:
  fixed_base:
    selector: { name: Base_Mount }
  load_face:
    selector: { name: Tip_Load }

# Typed materials, assignments, constraints, signals, excitation cases,
# excitations, initial conditions, response spectra, domains, and interfaces.

sequence:
  sequence_id: bracket_sequence
  steps: []       # typed procedure-step records
  dependencies: []
  transitions: []
  branches: []

execution:
  - step: static
    backend: cpu
    requirement: production
```

The abbreviated sequence above documents structure; use the checked-in example for runnable step,
activation, state, observation, and accuracy records. `model.defaults: template_scaffold` asks
RunMat to provide template-owned model defaults; `none` requires the document to provide the
physical model explicitly.

Region selectors are closed objects containing exactly one of `id`, `name`, or `tag`. CAD imports
commonly expose tags such as `occt_face`, `cad_body`, `cad_layer`, and `cad_material`; inspect the
loaded geometry before relying on a selector. Prefix strings such as `name:Tip_Load` are not part
of the `.fea` schema.

## Sweep File

Use a sweep when several complete studies should be checked or run as one deterministic collection.
Nested studies use the same typed study fields but omit `kind`.

```yaml
version: 1
kind: sweep
id: bracket_material_sweep
fail_fast: true

studies:
  - version: 1
    id: bracket_aluminum
    geometry: { path: ../geometry/bracket.stl, units: meter }
    model: { template: linear_static_structural }
    sequence: # complete typed sequence, as in the runnable example
    execution: # one typed entry per executable step

  - version: 1
    id: bracket_steel
    geometry: { path: ../geometry/bracket.stl, units: meter }
    model: { template: linear_static_structural }
    sequence: # complete typed sequence
    execution: # one typed entry per executable step
```

The fragment documents nesting and is not a complete runnable sweep because its sequence records
are elided. `fail_fast: false` lets planning or running continue after a study-level failure and
records failure entries in the sweep result. Scheduling policy is not encoded into the document
schema; the execution service may run independent studies concurrently when placement permits.

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

step = fea.StructuralStep.static("static");
sequence = fea.AnalysisSequence.new( ...
    "bracket_sequence", {step}, cell(0, 0), cell(0, 0), cell(0, 0));
backend = fea.ComputeBackend.cpu();
requirement = fea.FormulationRequirement.experimental();
execution = fea.ExecutionPlan.defaults(sequence, backend, requirement);

model = fea.model("bracket_static_model", geom, ...
    "Template", "linear_static_structural", ...
    "Materials", {steel}, ...
    "Constraints", {fixed}, ...
    "Signals", {forceSignal}, ...
    "Excitations", {tipLoad}, ...
    "ExcitationCases", {serviceLoads}, ...
    "Sequence", sequence);

study = fea.study("bracket_static", geom, ...
    "Template", "linear_static_structural", ...
    "Mesh", solverMesh, ...
    "Execution", execution, ...
    "Model", model);

validation = fea.validate(study);
plan = fea.plan(study);
result = fea.run(study);
results = fea.results(result);
stress = fea.field(results, fea.StructuralField.vonMises());
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
step = fea.ThermalStep.steady("thermal_equilibrium");

thermalModel = fea.model("cooled_bracket", geom, ...
    "Template", "thermal_steady", ...
    "Boundaries", {cooling}, ...
    "Steps", {step});
```

Use `fea.ThermalStep.transient(...)` and the `thermal_transient` template when temperature must
evolve over time. Transient procedures sample typed temperature, heat-flux, and
ambient-temperature histories at each physical solution time. The steady procedure accepts typed
constants (including constant signals) and solves the equilibrium operator without a heat-capacity
term; it does not silently select a point from a time history.

Use `Constraints` for prescribed values and kinematic relations, and `Boundaries` for natural or Robin operators. For fluid models, inlet velocity and wall motion are constraints; outlet pressure is a boundary. For thermal models, prescribed temperature is a constraint; heat flux and convection are boundaries.

Structural support motion is also a constraint. Author one vector acceleration history, select the
component with a typed translational DOF, and supply the support state at the beginning of the
transient step:

```matlab
base = geometry.region(geom, "name:Base");
times = [0; 0.25; 0.5; 1.0];
accelerationSI = [0 0 0; 12 0 0; -6 0 0; 0 0 0];
acceleration = fea.AccelerationSignal.fromTimeSamples( ...
    "base-acceleration", times, accelerationSI, ...
    fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());
motionX = fea.StructuralMotion.acceleration( ...
    fea.StructuralDof.ux(), acceleration, ...
    fea.Displacement.meters(0), fea.Velocity.metersPerSecond(0));
movingBase = fea.StructuralConstraint.prescribedMotion( ...
    "moving-base", base, motionX);
step = fea.StructuralStep.transient("support-response");

model = fea.model("moving-support-model", geom, ...
    "Template", "transient_structural", ...
    "Constraints", {movingBase}, ...
    "Signals", {acceleration}, ...
    "Steps", {step});
```

The acceleration matrix has one row per time and global X/Y/Z columns in meters per second squared.
The selected DOF chooses the column used by that motion component. Multiple components may share
the same signal and are passed as a cell array to `prescribedMotion`. RunMat reconstructs their
displacement and velocity exactly for piecewise-linear acceleration and reports support work
separately from work done by applied loads. The acceleration signal remains a first-class model
input: pass it through `Signals` as shown above, while the constraint retains its stable signal
identity.

The same host-only physical-value boundary applies to `fea.material`, the quantity-specific signal constructors, and the dimensioned quantity constructors. Domain constructors accept only the quantity classes required by their physics: for example, `fea.ElectromagneticDomain.atFrequency` requires `fea.Frequency`, and `fea.FluidDomain.newtonian` requires `fea.MassDensity`, `fea.DynamicViscosity`, and `fea.TurbulenceIntensity`. Bare numbers cannot silently acquire units. Numeric quantity inputs are checked before one binary64 model-storage conversion, so sufficiently wide integers can round. Numerical enforcement controls such as contact penalties and coupling relaxation belong to typed step or solver policy, not to interface physics. Geometry/model revisions, plan counts, field shape/count metadata, and comparison count deltas are structural integers; they remain exact and do not enter the physical-value conversion. The conventional plotting handle remains a double at the scripting boundary. Resident numeric constructor fields are rejected because typed FEA object construction performs no provider execution.

Execution is authored separately from the semantic sequence. `fea.ComputeBackend` and
`fea.FormulationRequirement` are closed typed choices. `fea.SolverOptions.defaults(step)` selects
the family-compatible default options for one procedure step;
`fea.ExecutionPlan.defaults(sequence, backend, requirement)` does that for every step. Use
`fea.ExecutionStep.new(...)` and `fea.ExecutionPlan.new(...)` when different steps require
different backends or formulation maturity. Plan construction rejects missing, duplicate,
out-of-order, and solver-incompatible step entries before a study can run.

`fea.results` uses positive one-based exact vectors for `ModeIndices` and `TransientSnapshotIndices`, matching normal RunMat indexing. Its inclusion flags accept logical scalars or exact numeric zero and one. Selector shape, positivity, integrality, and range are validated before artifact lookup; resident selectors reject without provider access. Structural result metadata such as shapes, counts, indexes, byte sizes, iteration counts, and fingerprints remains exact, while field values and physical/timing/residual quantities remain double. `fea.trends("WindowSize", n)` applies the same exact structural rule to a positive scalar and defaults to 16.

`fea.materialAssignment`, `fea.model`, `fea.plan`, `fea.field`, and `fea.compare` have no integer-array input role. Their inputs are exact text identifiers, typed FEA or geometry objects, or document paths, while any integer-valued metadata they return follows the structural-output rule. The ordinary `plot(...)` and `mesh(...)` functions dispatch FEA objects through their visualization methods without changing the numeric MATLAB-compatible forms. `fea.compare` accepts exactly the baseline and candidate run identifiers and has no Name, Value options.

Use `.fea` when the study definition should be checked in as a portable declarative artifact:

```matlab
study = fea.load("studies/bracket_static.fea");
validation = fea.validate(study);
result = fea.run(study);
results = result.results();
stress = result.field(fea.StructuralField.vonMises());
stressPage = stress.page();
peakStressPa = max(stressPage.values);
```

`fea.load(...)` returns either a `fea.Study` or `fea.Sweep` object. `geometry.load(...)` returns a `geometry.Asset` object, and `meshing.generate(...)` returns a validated `meshing.Mesh`. The workflow builtins accept `.fea` paths, `fea.Study` objects, or `fea.Sweep` objects. Study operations return `fea.Validation`, `fea.Plan`, and an aggregate `fea.RunResult`; sweep operations return the distinct aggregate classes `fea.SweepValidation`, `fea.SweepPlan`, and `fea.SweepRunResult`. Select one persisted procedure-step run with `stepRun = result.step("step_id")`; the resulting `fea.StepRunResult` owns the step's `run_id` and exposes `results()`, `field(selector, sample)`, and `mesh()`. Those methods also work directly on a `fea.RunResult` when the study has exactly one step. A multi-step aggregate requires explicit step selection.

`fea.Field` is a typed field handle. Its `shape`, unit, location, kind, family, quantity, sample coordinate, numerical representation, and run identity are available without loading the numerical payload. Small conventional results may also carry an already-materialized `values` property; code that must work for large or distributed results should use `page = field.page(start, count)`. `start` is a positive one-based logical-value index, `count` defaults to 256, and each request is capped at 4096 values. `page.values` is a real tensor for real fields and a native complex tensor for harmonic fields. `page.next_start` is empty at the end; otherwise pass it to the next call. `page.count`, `page.total_count`, and `page.start` remain exact integers.

Frequency selection is typed and remains separate from rendering:

```matlab
sample = fea.Frequency.hertz(120);
displacement = fea.field( ...
    harmonicResults, fea.StructuralField.displacement(), sample);
page = displacement.page(1, 1024);
amplitude = abs(page.values);
plot(displacement);
```

`plot(field)` streams the selected artifact through the visualization boundary and does not require a full driver-side tensor. It renders complex harmonic values by magnitude. Field selection belongs to `fea.field`; it is not a plotting option.

The typed constructors and post-processing helpers do not intrinsically require native OCCT. They can consume an existing `geometry.Asset`, STL-backed or synthetic geometry, and persisted mesh/result artifacts without OCCT; only an earlier import of topology-bearing CAD such as STEP, IGES, or BREP may require the configured OCCT geometry backend.

Argument decoding for `fea.run`, and artifact filtering for `fea.results` and `fea.trends`, is host-side. The solver invoked by `fea.run` is different: it may execute through the CPU or configured GPU backend selected by the study. Published result fields use canonical host `double` storage; solver backend, device-operator use, host synchronization, and fallback provenance report where the solve actually executed. Reading results does not trigger device work or an implicit transfer.

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
| `fea.<Quantity>Signal.fromTimeSamples(...)` | A quantity-typed sampled history with typed interpolation and extrapolation policy. |
| `fea.<Family>Excitation.<action>(...)` | A typed spatial action binding a compatible signal to a `geometry.Region`. |
| `fea.ExcitationCase.always(...)`, `.duringSteps(...)`, `.mutuallyExclusive(...)` | Group typed actions and declare their activation without changing either the signal or spatial action. |
| `fea.<Family>InitialCondition.<state>(...)` | Sample a compatible typed signal into the model state before a named procedure step. |
| `fea.ResponseSpectrum.new(...)` | Assemble response-spectrum data, evaluation policy, direction, modal rules, evidence, and provenance from typed components. |
| `fea.<Family>Step.<procedure>(id)` | A concrete direct-solution procedure step; family and procedure are selected by the class method rather than text. |
| `fea.ProcedureStep.configure(...)`, `fea.AnalysisSequence.new(...)` | Add typed domains, activations, state transfer, observations, accuracy, restart, dependencies, transitions, and branches to procedure steps. |
| `fea.ThermoMechanicalDomain.atReferenceTemperature(...)` | Thermo-mechanical constitutive reference state from a typed `fea.Temperature`. |
| `fea.ElectroThermalDomain.atReferenceTemperature(...)` | Electro-thermal constitutive reference state from a typed `fea.Temperature`. |
| `fea.ElectromagneticDomain.atFrequency(...)` | Electromagnetic frequency-domain definition from a typed `fea.Frequency`. |
| `fea.FluidDomain.newtonian(...)` | Newtonian fluid properties from typed density, viscosity, and turbulence intensity values. |
| `fea.ContactInterface.frictionless(...)`, `fea.ContactInterface.coulomb(...)` | Contact between two typed regions, optionally with typed Coulomb friction. |
| `fea.FluidStructureInterface.noSlip(...)` | No-slip fluid-to-structure transfer across two typed regions. |
| `fea.ConjugateHeatTransferInterface.perfect(...)`, `.conductance(...)`, `.resistance(...)` | Fluid-to-solid heat transfer with exactly one physical transfer law. |
| `fea.SolverOptions.defaults(step)` | Create typed solver options compatible with one procedure step. |
| `fea.ExecutionStep.new(...)`, `fea.ExecutionPlan.new(...)`, `fea.ExecutionPlan.defaults(...)` | Bind each semantic procedure step to typed backend, formulation maturity, and solver options. |
| `fea.model(...)` | Explicit model assembled from typed components. |
| `fea.study(...)` | Study assembled from geometry, a typed model template/model, sequence, backend, and formulation requirement. |
| `fea.sweep(...)` | Study sweep assembled from `fea.Study` objects. |
| `fea.stepResult(run, stepId)` | Select a typed `fea.StepRunResult` from an aggregate study result. |
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

### Histories, initial state, and response spectra

A sampled signal owns its physical quantity, time axis, values, interpolation, extrapolation,
precision, evidence, and authored provenance. Closed choices are objects, not spelling-sensitive
strings:

```matlab
interpolation = fea.SignalInterpolation.linear();
extrapolation = fea.SignalExtrapolation.zero();

launchAcceleration = fea.AccelerationSignal.fromTimeSamples( ...
    "launch.acceleration", ...
    [0 0.01 0.02], ...
    [0 0 0; 2 0 0; 0 0 0], ...
    interpolation, extrapolation);
```

The time axis must contain at least two finite, strictly increasing samples. Scalar signals accept
one value per time. Vector signals accept either an N-by-3 or 3-by-N matrix and retain the sample
ordering. Interpolation is `leftEndpoint`, `rightEndpoint`, or `linear`; extrapolation is `reject`,
`holdEndpoint`, or `zero`.

Represent a physical jump with its two limits and say which limit owns the value at the jump.
This keeps the function value distinct from either one-sided query and lets transient solvers land
on the event instead of stepping across it:

```matlab
leftContinuous = fea.SignalPointContinuity.left();
knots = {
    fea.SignalKnot.continuous(0, 0)
    fea.SignalKnot.jump(0.5, 0, 100, leftContinuous)
    fea.SignalKnot.continuous(1, 100)
};
temperature = fea.TemperatureSignal.fromTimeKnots( ...
    "boundary.temperature", knots, interpolation, extrapolation);
```

Initial conditions do not copy anonymous numbers into a solver option. They identify the typed
signal sample that establishes state and the step that consumes it:

```matlab
zeroDisplacement = fea.DisplacementSignal.constantMeters( ...
    "initial.displacement", [0 0 0]);
wholeModel = fea.InitialConditionTarget.wholeModel();
initialDisplacement = fea.StructuralInitialCondition.displacement( ...
    "initial", wholeModel, zeroDisplacement, 0, ...
    fea.SignalSampleApproach.at(), "transient");
```

Use a `geometry.Region` instead of `wholeModel` to bind an initial condition to a volume region.
The condition retains the geometry id and revision; model composition rejects a region from another
asset or revision.
Structural displacement, velocity, and angular velocity; thermal temperature; electrical
potential; magnetic vector potential; acoustic pressure; and fluid pressure and velocity each
have separate constructors and accept only a signal of the required quantity and shape.

A response spectrum is equally explicit. The spectrum ordinates remain numeric data; every
scientific choice around those data is a typed value:

```matlab
axis = fea.SpectrumAxis.frequencyHertz([5 10 20]);
damping = fea.SpectrumDamping.ratio(0.05);
response = fea.SpectrumResponse.absoluteAcceleration();
envelope = fea.SpectrumEnvelope.maximax();
spectrumInterpolation = fea.SpectrumInterpolation.linearFrequency();
spectrumExtrapolation = fea.SpectrumExtrapolation.reject();
direction = fea.SpectrumDirection.new( ...
    fea.CoordinateFrame.globalFrame(), [1 0 0]);
modalCombination = fea.ModalCombination.completeQuadratic();
directionalCombination = fea.DirectionalCombination.srss();
truncation = fea.ModalTruncation.new(0.90, true, true, true);
construction = fea.SpectrumConstruction.authoredDesign();
evidence = fea.SpectrumEvidence.bounded(0.95, 0.01);

launchSpectrum = fea.ResponseSpectrum.new( ...
    "launch", axis, [12 18 9], damping, response, envelope, ...
    spectrumInterpolation, spectrumExtrapolation, direction, ...
    modalCombination, directionalCombination, truncation, ...
    construction, evidence, "sha256:launch-spectrum");
```

Frequency axes pair with linear-frequency interpolation; period axes pair with linear-period
interpolation; either can use log-log interpolation when every ordinate is positive. Axis values
must be positive and strictly increasing. Direction vectors are normalized in their typed
coordinate frame. Damping, directional percentages, modal truncation, sampling adequacy, and
interpolation-error evidence are range checked at construction and again with the complete model.
`SpectrumConstruction.fromAccelerationHistoryExact(signal)` and
`fromAccelerationHistoryRecursive(signal)` retain the source acceleration-signal identity; model
validation then requires that exact time-domain signal to be present in `Signals`.

Supply these objects through `InitialConditions` and `ResponseSpectra` alongside the signals they
reference:

```matlab
model = fea.model("launch_model", geom, ...
    "Template", "modal_structural", ...
    "Signals", {launchAcceleration, zeroDisplacement}, ...
    "InitialConditions", {initialDisplacement}, ...
    "ResponseSpectra", {launchSpectrum});
```

Modal response spectrum is a two-step study. Modal extraction publishes the exact mass-normalized
basis; the response-spectrum step consumes that state and names one or more spectrum ids in its
independent domain. The dependency and state transition are explicit, so a spectrum cannot be
silently left unused or paired with an unrelated basis. Direction solves may run on separate
workers, while canonical fan-in always follows authored direction order. The terminal result
retains participation and effective mass by mode, captured-mass evidence, combination and
correction policy, the component-magnitude envelope, and its accuracy artifact.

The complete runnable study is
[`modal-response-spectrum-experimental.fea`](examples/modal-response-spectrum-experimental.fea):

```sh
runmat check docs/fea/examples/modal-response-spectrum-experimental.fea
runmat run docs/fea/examples/modal-response-spectrum-experimental.fea
```

### Random vibration from cross-power spectra

Random response is a two-step structural study: modal extraction publishes a mass-normalized modal
basis, then the random-response step integrates an authored cross-power spectral-density (CPSD)
matrix over frequency. The excitation owns the physical input definition. Solver options own only
modal damping, modal-mass truncation, a dimensionless relative quadrature tolerance, and how many
canonical frequency intervals may be grouped into one portable task.

Each channel has a quantity, coordinate frame, and direction. At every frequency, `values` is the
complete row-major complex CPSD matrix. Diagonal entries are auto spectra; off-diagonal entries
retain phase and correlation between channels. RunMat requires a Hermitian positive-semidefinite
matrix, one-sided density per hertz, strictly increasing nonnegative frequencies, and units of the
channel quantity squared per hertz. For one base-acceleration channel, a sample is a one-entry
matrix:

```yaml
random_excitations:
  - excitation_id: broadband_base
    frequencies_hz: [0.1, 10.0, 100.0]
    channels:
      - channel_id: base_x
        quantity: base_acceleration
        direction: { frame: global, vector: [1.0, 0.0, 0.0] }
    spectral_density:
      - values: [{ real: 0.0001, imaginary: 0.0 }]
      - values: [{ real: 0.0002, imaginary: 0.0 }]
      - values: [{ real: 0.0001, imaginary: 0.0 }]
    convention: one_sided_per_hertz
    interpolation: linear_frequency
    extrapolation: reject
    evidence:
      max_absolute_density_error: 0.0
      equivalent_degrees_of_freedom: null
    provenance_digest: sha256:documented-broadband-base
```

The consumer step names that excitation and consumes the exact modal-basis state through a typed
state transition. RunMat partitions the spectrum into resonance-aware frequency intervals, admits
the resources for each interval, executes the same canonical tasks locally or on remote workers,
and reduces interval contributions in ordinal order. Completion publishes variances and RMS fields
plus a numerical receipt bound to the model, basis, excitation, integration policy, task set, and
result. CPU execution is available as an experimental formulation; unsupported accelerator
placement is rejected rather than silently falling back.

The complete runnable study is
[`modal-random-response-experimental.fea`](examples/modal-random-response-experimental.fea):

```sh
runmat check docs/fea/examples/modal-random-response-experimental.fea
runmat run docs/fea/examples/modal-random-response-experimental.fea
```

Use the ordinary result surface after completion: select a published field with `fea.field(...)`,
then call `plot(field)`; call `mesh(run)` to inspect the solver topology. Random-excitation
authoring is also available through the typed document operations `add_random_excitation`,
`update_random_excitation`, and `remove_random_excitation`.

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
    AnalysisCreateModelIntentSpec, AnalysisModelTemplate,
};
use runmat_runtime::geometry::geometry_load_op;
use runmat_runtime::operations::OperationContext;

let context = OperationContext::new(Some("trace-001".to_string()), None);
let geometry = geometry_load_op("bracket.step", &bytes, context.clone())?.data;

let model = analysis_create_model_op(
    &geometry,
    AnalysisCreateModelIntentSpec {
        model_id: "bracket_static_model".to_string(),
        template: AnalysisModelTemplate::LinearStaticStructural,
    },
    context.clone(),
)?.data;

analysis_validate(
    &model,
    geometry.units,
    &runmat_analysis_core::CoordinateFrame::Global,
    context.clone(),
)?;

let run = analysis_run_linear_static_op(&model, ComputeBackend::Cpu, context)?.data;
```

The public operation identifiers in envelopes are `fea.*`, such as `fea.create_model/v1` and `fea.run_linear_static/v1`.
