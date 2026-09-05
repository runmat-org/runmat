# Operation Reference

This page is for host integrations and tooling that care about operation names, envelopes, error codes, artifact roots, and builtin names.

## Envelopes

Successful operations return `OperationEnvelope<T>`:

| Field | Meaning |
| --- | --- |
| `operation` | Stable operation name, such as `fea.run_modal`. |
| `op_version` | Versioned operation identifier, such as `fea.run_modal/v1`. |
| `trace_id` | Optional host trace id. |
| `request_id` | Optional host request id. |
| `data` | Typed operation payload. |

Failures return `OperationErrorEnvelope`:

| Field | Meaning |
| --- | --- |
| `error_code` | Stable machine-readable code. |
| `error_type` | `input`, `validation`, `capacity`, `backend`, `internal`, or `contract`. |
| `message` | Human-readable message. |
| `operation`, `op_version` | Operation identity that failed. |
| `retryable` | Whether retrying may be reasonable. |
| `severity` | `warning`, `error`, or `fatal`. |
| `context` | String key/value details for automation. |
| `trace_id`, `request_id`, `timestamp` | Correlation and timing metadata. |

Error codes use `RM.<DOMAIN>.<OPERATION>.<REASON>`, for example `RM.FEA.RUN_STUDY.INVALID_SPEC` or `RM.GEOMETRY.LOAD.UNSUPPORTED_FORMAT`.

## Geometry Operations

| Operation version | Use |
| --- | --- |
| `geometry.inspect/v1` | Detect supported input format and byte count before loading. |
| `geometry.load/v1` | Import geometry bytes into a `GeometryAsset`. |
| `geometry.compute_stats/v1` | Return geometry statistics for an asset. |
| `geometry.list_regions/v1` | Return known geometry regions. |
| `geometry.query_entities/v1` | Return region, mesh, or entity references with a bounded query. |
| `geometry.capture_view/v1` | Capture a geometry view through the installed adapter. |

## FEA Operations

| Operation version | Use |
| --- | --- |
| `fea.create_model/v1` | Build an `AnalysisModel` from geometry and a model profile intent. |
| `fea.validate/v1` | Validate model units, frame, materials, loads, boundary conditions, domains, and geometry compatibility. |
| `fea.run_linear_static/v1` | Run the linear static structural path. |
| `fea.run_modal/v1` | Run modal analysis. |
| `fea.run_acoustic/v1` | Run acoustic harmonic analysis. |
| `fea.run_thermal/v1` | Run standalone thermal analysis. |
| `fea.run_transient/v1` | Run structural transient analysis. |
| `fea.run_nonlinear/v1` | Run nonlinear structural analysis. |
| `fea.run_electromagnetic/v1` | Run electromagnetic analysis. |
| `fea.run_cfd/v1` | Run finite-volume incompressible CFD analysis. |
| `fea.run_cht/v1` | Run coupled CFD plus thermal conjugate heat-transfer analysis. |
| `fea.run_fsi/v1` | Run partitioned fluid-structure interaction analysis. |
| `fea.results/v1` | Query fields, diagnostics, payload subsets, quality reasons, provenance, and summaries. |
| `fea.results_compare/v1` | Compare selected fields between two persisted runs. |
| `fea.trends/v1` | Summarize persisted runs by family. |

## Study Operations

| Operation version | Use |
| --- | --- |
| `fea.validate_study/v1` | Validate one study and write a validation artifact. |
| `fea.plan_study/v1` | Produce operation sequence, run operation, fingerprint, and plan artifact. |
| `fea.run_study/v1` | Execute one study and write run evidence. |
| `fea.validate_study_sweep/v1` | Validate a sweep with aggregate and per-study issues. |
| `fea.plan_study_sweep/v1` | Plan a sweep with plan entries and failure entries. |
| `fea.run_study_sweep/v1` | Execute a deterministic sequential sweep. |

## RunMat Builtins

| Builtin | Use |
| --- | --- |
| `geometry.inspect(path)` | Read a geometry file and return a `geometry.InspectResult` object. |
| `geometry.load(path, "Units", units)` | Read geometry with explicit source units and return a `geometry.Asset` object. |
| `geometry.listRegions(asset)` | Return imported regions from a `geometry.Asset`. |
| `geometry.meshes(asset)` | Return patch-ready surface mesh topology with vertices, faces/triangles, and region mappings. |
| `mesh(asset, Name, Value, ...)` | Display a `geometry.Asset`; `Edges` and `XRay` are visual options. |
| `meshing.generate(asset, Name, Value, ...)` | Generate a validated `meshing.Mesh` through the installed execution provider. |
| `meshing.load(meshPath, evidencePath)` | Decode and jointly validate canonical solver-mesh and evidence artifacts. |
| `meshing.save(mesh, meshPath, evidencePath, Name, Value, ...)` | Stage and transactionally publish the canonical pair; existing paths require `Force`, true. |
| `meshing.nodes(mesh)`, `meshing.elements(mesh)`, `meshing.boundaryFaces(mesh)` | Return coordinates and exact typed topology identifiers/connectivity. |
| `meshing.regions(mesh)` | Return persistent physical-region ids and element counts. |
| `meshing.quality(mesh)` | Return per-element and summary Jacobian, aspect-ratio, and volume statistics. |
| `meshing.evidence(mesh)` | Return the evidence identity and resource summary bound to the mesh. |
| `mesh(mesh, Name, Value, ...)` | Display solver topology with `Edges`, `Regions`, and `Section` options. |
| `fea.load(path)` | Load a `.fea` study or sweep file and return `fea.Study` or `fea.Sweep`. |
| `fea.material(...)`, `fea.materialAssignment(...)` | Create typed material data and region assignments. |
| `fea.<Family>Constraint.<kind>(...)` | Create typed essential, kinematic, gauge, pairing, and reference constraints. |
| `fea.<Family>Boundary.<kind>(...)` | Create typed natural and Robin boundary operators. |
| `fea.<Quantity>Signal.constant<Unit>(...)` | Create a quantity-typed constant physical signal. |
| `fea.SignalInterpolation.<policy>()`, `fea.SignalExtrapolation.<policy>()` | Create closed typed sampling policies. |
| `fea.<Quantity>Signal.fromTimeSamples(id, times, values, interpolation, extrapolation)` | Create a quantity-typed sampled time history. |
| `fea.<Family>Excitation.<action>(...)`, `fea.ExcitationCase.<activation>(...)` | Bind signals to geometry, then group actions under a typed activation policy. |
| `fea.InitialConditionTarget.wholeModel()`, `fea.<Family>InitialCondition.<state>(...)` | Create whole-model or region-bound initial state from a typed signal sample. |
| `fea.ResponseSpectrum.new(...)` and `fea.Spectrum*` policy classes | Create a typed modal response spectrum with explicit axis, direction, combination, evidence, and provenance. |
| `fea.<Family>Step.<procedure>(id)`, `fea.<Physics>Domain.<constructor>(...)` | Create concrete typed analysis steps and physics domains. |
| `fea.ContactInterface.frictionless(...)`, `.coulomb(...)`; `fea.FluidStructureInterface.noSlip(...)`; `fea.ConjugateHeatTransferInterface.perfect(...)`, `.conductance(...)`, `.resistance(...)` | Create a concrete typed interface between two `geometry.Region` values. |
| `fea.runOptions(kind, Name, Value, ...)` | Create family-specific run options. |
| `fea.model(id, geometry, Name, Value, ...)` | Assemble an explicit model from geometry and typed components. |
| `fea.study(id, geometry, Name, Value, ...)` | Create a `fea.Study` from geometry, optional validated `Mesh`, physics profile/model data, backend, and run options. |
| `fea.sweep(id, studies, Name, Value, ...)` | Create a deterministic sweep from `fea.Study` objects. |
| `fea.validate(study)` | Validate a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.plan(study)` | Plan a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.run(study)` | Run a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.results(runOrId, Name, Value, ...)` | Load result data from a run result object or persisted run id. |
| `fea.field(resultsOrRun, fieldId)` | Extract one field from a result query. |
| `plot(field, Name, Value, ...)` | Plot an `fea.Field` using its retained study geometry and solver topology context. Field selection remains in `fea.field`. |
| `mesh(runOrResultsOrField, Name, Value, ...)` | Display the solver mesh by default; `Source` accepts `solver`, `cad`, `cad_reference`, or `auto`, and `Edges` controls the solver-edge overlay. |
| `fea.compare(baselineRunId, candidateRunId)` | Compare two persisted runs; no Name, Value options are accepted. |
| `fea.trends(Name, Value, ...)` | Summarize recent persisted runs. |

Family-specific constraints and boundaries require immutable `geometry.Region` values, so every authored condition carries the source geometry identity and revision. Model composition rejects regions from another geometry or an earlier revision. Numeric physical inputs are dimensioned values rather than bare scalars: pressure, heat flux, ambient temperature, heat-transfer coefficient, and acoustic impedance each use their corresponding `fea.*` quantity class. Constraints are supplied through the `Constraints` model/study option; natural and Robin operators are supplied through `Boundaries`.

| Boundary constructor | Typed value |
| --- | --- |
| `fea.StructuralBoundary.pressure(id, region, pressure)` | `fea.Pressure` |
| `fea.ThermalBoundary.heatFlux(id, region, heatFlux)` | `fea.HeatFlux` |
| `fea.ThermalBoundary.convection(id, region, ambientTemperature, coefficient)` | `fea.Temperature`, `fea.HeatTransferCoefficient` |
| `fea.MagneticBoundary.insulation(id, region)` | None |
| `fea.AcousticBoundary.rigidWall(id, region)` | None |
| `fea.AcousticBoundary.radiation(id, region)` | None |
| `fea.AcousticBoundary.impedance(id, region, impedance)` | `fea.SpecificAcousticImpedance` |
| `fea.FluidBoundary.outletPressure(id, region, pressure)` | `fea.Pressure` |

Fluid inlet velocity, moving/no-slip/slip walls, and symmetry are kinematic constraints and therefore live on `fea.FluidConstraint`. A fluid outlet pressure is a natural boundary. Thermal prescribed temperature is an essential constraint, while heat flux and convection are natural/Robin boundaries. This division follows the mathematical role of the condition rather than its placement on a geometric boundary.

| Interface constructor | Typed parameters |
| --- | --- |
| `fea.ContactInterface.frictionless(id, primary, secondary)` | None |
| `fea.ContactInterface.coulomb(id, primary, secondary, coefficient)` | `fea.FrictionCoefficient` |
| `fea.FluidStructureInterface.noSlip(id, fluid, structure)` | None |
| `fea.ConjugateHeatTransferInterface.perfect(id, fluid, solid)` | None |
| `fea.ConjugateHeatTransferInterface.conductance(id, fluid, solid, conductance)` | `fea.HeatTransferCoefficient` |
| `fea.ConjugateHeatTransferInterface.resistance(id, fluid, solid, resistance)` | `fea.ThermalContactResistance` |

Each interface joins two distinct `geometry.Region` values from one geometry revision. Contact does not choose an FSI or CHT transfer law, and coupled interfaces do not double as contact constraints. Their orientation is explicit: FSI and CHT use the fluid region as the primary side, with the structure or solid region as the secondary side.

`fea.material`, the quantity-specific signal constructors, and the dimensioned quantity constructors are RunMat-native host constructors. Finite real quantity inputs accept all eight built-in integer classes, preserve exact input through shape validation, and then cross one binary64 model-storage boundary; sufficiently wide integers can round, and resident numeric fields are rejected without provider dispatch. Physics domains accept only the corresponding typed quantities and cannot contain loads, signals, solver choices, or enable flags. Interface constructors accept only typed `geometry.Region` selections and, where required, a semantic physical parameter. CHT selects exactly one of perfect transfer, conductance, or resistance, so redundant or contradictory transfer laws cannot be authored. Contact penalty, stabilization, damping, and relaxation are numerical policy owned by the step and solver rather than interface physics.

`fea.materialAssignment`, `fea.model`, `fea.plan`, `fea.field`, and `fea.compare` accept structural objects and text rather than integer arrays. FEA objects overload the ordinary `plot` and `mesh` functions through the runtime object protocol; numeric calls keep their MATLAB-compatible behavior. Discrete output metadata remains integer typed: model and geometry revisions, plan counts and indexes, field shape/component/element/size metadata, and comparison count deltas do not pass through a physical-value binary64 boundary. The conventional figure handle remains a double at the scripting boundary. Input arity and structural identity are validated before artifact lookup, planning, field projection, comparison, or figure creation.

These constructors, planning operations, field projections, comparisons, and plots do not intrinsically require native OCCT. Existing geometry assets, STL or synthetic geometry, and persisted mesh/result artifacts use their normal paths without OCCT; only document resolution that must import topology-bearing CAD may require an OCCT-enabled geometry backend.

### Workflow signatures and validation

| Builtin | Exact signature | Validation boundary |
| --- | --- | --- |
| `fea.StructuralStep.static/modal/transient/nonlinear` | `step = fea.StructuralStep.<procedure>(id)` | The id is a non-empty text scalar; family, procedure, direct solution space, independent domain, and state outputs come from the concrete method. |
| `fea.ThermalStep.transient`, `fea.ElectromagneticStep.harmonic`, `fea.AcousticStep.harmonic` | `step = fea.<Family>Step.<procedure>(id)` | Only currently supported family/procedure combinations are constructible. |
| `fea.FluidStep.steadyState/transient` | `step = fea.FluidStep.<procedure>(id)` | Steady-state produces a static domain; transient produces a time domain. |
| `fea.ThermoMechanicalStep.transient`, `fea.ElectroThermalStep.transient/nonlinear`, `fea.ConjugateHeatTransferStep.steadyState/transient`, `fea.FluidStructureStep.transient` | `step = fea.<CoupledFamily>Step.<procedure>(id)` | Coupled family and procedure are fixed by the concrete method and cannot be supplied as strings. |
| `fea.study` | `study = fea.study(path)` or `study = fea.study(id, geometry, Name, Value, ...)` | The path must resolve to a study. Constructor identity, geometry class, option spelling, component classes, and profile/run-kind agreement are checked before construction. |
| `fea.sweep` | `sweep = fea.sweep(id, studies, Name, Value, ...)` | The id and study object classes are checked before `FailFast`; `FailFast` is a logical scalar and defaults to true. |
| `fea.validate` | `result = fea.validate(studyOrSweepOrPath)` | Object/path identity is checked before document validation; no solver executes. |
| `fea.run` | `run = fea.run(studyOrSweepOrPath)` | Object/path identity and document validity are checked before the selected solver executes. |
| `fea.results` | `results = fea.results(runOrRunId, Name, Value, ...)` | Run identity, option pairing, option values, and selector shape/range are checked before artifact lookup. An existing `fea.Results` object is accepted without options. |
| `fea.trends` | `trends = fea.trends(Name, Value, ...)` | `WindowSize` is checked before artifact-store access. |

`fea.study` defaults to profile `linear_static_structural`, backend `cpu`, defaults mode `profile_scaffold`, and a sanitized `<study-id>_model` model id when none is supplied. Its constructor options include `Template`, `Backend`, `Requirement`, `ModelId`, `Model`, `Frame`, `Defaults`, `Materials`, `MaterialAssignments`/`Assignments`, `Constraints`, `Boundaries`, `Signals`, `Excitations`, `ExcitationCases`/`Cases`, `InitialConditions`, `ResponseSpectra`, `Steps`, `Domains`, `Interfaces`, and `Mesh`. `fea.model` accepts the same model-component options. Direct integer values do not substitute for any identity or typed-object option.

### Run-option numeric contracts

`fea.runOptions(solver, Name, Value, ...)` accepts solver families `linear_static`, `modal`, `acoustic`, `thermal`, `transient`, `cfd`, `cht`, `fsi`, `nonlinear`, and `electromagnetic`. Every family defaults `DeterministicMode=false`, `PrecisionMode="fp64"`, and `QualityPolicy="balanced"`; linear static also defaults `PreconditionerMode="auto"`.

| Solver | Exact structural options and defaults | Binary64 options and defaults |
| --- | --- | --- |
| `modal` | `ModeCount=3` | `ResidualWarnThreshold=1e-3` |
| `acoustic` | `ModeCount=3` | `ResidualWarnThreshold=1e-3` |
| `thermal` | `StepCount=10` | `TimeStepS=1e-2`, `ResidualWarnThreshold=1e-4` |
| `transient` | `StepCount=10`, `MaxLinearIters=128`, `MaxStepRetries=4` | `TimeStepS=1e-3`, `MinTimeStepS=1e-6`, `MaxTimeStepS=2e-2`, `Tolerance=1e-8`, `ResidualTarget=1e-6`, `AdaptMinScale=0.8`, `AdaptMaxScale=1.25`, `AdaptGrowthExponent=0.35`, `AdaptRetryGrowthCap=1.05`, `AdaptNonconvergedShrink=0.75`, `DtBucketRelTolerance=0` |
| `cfd` | `StepCount=12`, `MaxLinearIters=128` | `TimeStepS=1e-3`, `Tolerance=1e-8`, `ResidualWarnThreshold=1e-5` |
| `cht` | `StepCount=12`, `MaxLinearIters=128` | `TimeStepS=1e-3`, `Tolerance=1e-8`, `ResidualWarnThreshold=1e-4` |
| `fsi` | `StepCount=12`, `MaxLinearIters=128` | `TimeStepS=1e-3`, `Tolerance=1e-8`, `ResidualWarnThreshold=1e-4` |
| `nonlinear` | `IncrementCount=12`, `MaxNewtonIters=24`, `MaxLineSearchBacktracks=6`, `TangentRefreshInterval=2` | `Tolerance=1e-6`, `ResidualConvergenceFactor=5`, `IncrementNormTolerance=1e-7`, `LineSearchReduction=0.5` |
| `electromagnetic` | `HarmonicMaxIterations=96` | `ResidualTarget=1e-6`, `HarmonicTolerance=1e-7`, `SweepFrequencyHz=[]` |

The table defines 18 integer-capability forms: one exact structural and one binary64 physical form for each of the nine numeric solver families. Exact controls accept all eight integer classes and ordinary integral doubles and retain their exact values through validation. Physical controls accept real scalars and perform one explicit binary64 conversion after field, solver, and scalar-shape validation; solver-specific finiteness and range checks run before execution. Logical options require logical scalars. Structural output properties remain exact; physical properties remain double.

### Results query options

| Option | Accepted value | Default |
| --- | --- | --- |
| `Fields` / `IncludeFields` | Field-id text list | Empty; do not filter by field id |
| `IncludeFieldValues` / `FieldValues` | Logical or exact numeric 0/1 | `true` |
| `IncludeDiagnostics` | Logical or exact numeric 0/1 | `true` |
| `DiagnosticCodes` | Diagnostic-code text list | Empty; do not filter by code |
| `IncludeModalResults` | Logical or exact numeric 0/1 | `true` |
| `ModeIndices` | Positive one-based exact structural vector | Empty; include every mode |
| `IncludeTransientResults` | Logical or exact numeric 0/1 | `true` |
| `TransientSnapshotIndices` | Positive one-based exact structural vector | Empty; include every snapshot |
| `IncludeNonlinearResults` | Logical or exact numeric 0/1 | `true` |
| `IncludeElectromagneticResults` | Logical or exact numeric 0/1 | `true` |

Mode and transient snapshot selectors use the normal one-based RunMat scripting convention. All eight integer classes and ordinary integral doubles are accepted from authoritative host storage; zero, negative, fractional, nonfinite, out-of-range, matrix-shaped, and resident selectors reject before artifact lookup. Public field shapes, element/component counts, byte sizes, mode/snapshot indexes, iteration counts, fingerprints, study/sweep counts, failure indexes, and solver synchronization counts remain exact typed integers. Field values and true physical, timing, residual, rate, and ratio quantities remain double.

`fea.trends` accepts only `WindowSize`, a positive exact structural scalar defaulting to 16. All eight integer classes and ordinary integral doubles are accepted; invalid and resident values reject before artifact-store access.

The `fea.results` and `fea.trends` query paths are host-side and do not gather or execute stored device fields. `fea.run` may execute through the study's configured GPU solver. None of these operations intrinsically requires native OCCT when consuming existing typed objects, STL/synthetic geometry, or persisted artifacts; only path-time import of topology-bearing CAD may require the configured OCCT backend.

`fea.Study` and `fea.Sweep` objects expose `validate`, `plan`, and `run` methods through the class system. A study produces `fea.Validation`, `fea.Plan`, and `fea.RunResult`; a sweep produces the distinct aggregate classes `fea.SweepValidation`, `fea.SweepPlan`, and `fea.SweepRunResult`. Only `fea.RunResult` represents one run and therefore exposes `results`, `field`, and `plot`. `fea.Results` exposes `field` and `plot`; `fea.Field` exposes `plot`.

## `.fea` Document Fields

Top-level study fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `version` | Yes | Document version. Current value is `1`. |
| `kind` | Yes | `study`. |
| `id` | Yes | Stable study id. |
| `geometry` | Yes | Geometry file path, units, and import options. |
| `model` | Yes | Model id, profile, defaults mode, and frame. |
| `run` | Yes | Run family, backend, and family-specific options. |
| `regions` | No | Named aliases for geometry region selectors. |
| `materials` | No | Material definitions keyed by material id. |
| `material_assignments` | No | Region-to-material assignments. |
| `boundary_conditions` | No | Constraints and boundary data. |
| `loads` | No | Forces, moments/torques, pressures, body forces, current densities, or coil currents. |
| `steps` | No | Analysis steps. |
| `domains` | No | Thermo-mechanical, electro-thermal, electromagnetic, or CFD domain data. |
| `interfaces` | No | Contact or coupling interfaces. |

Top-level sweep fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `version` | Yes | Document version. Current value is `1`. |
| `kind` | Yes | `sweep`. |
| `id` | Yes | Stable sweep id. |
| `fail_fast` | No | Defaults to `true`. |
| `studies` | Yes | List of nested study documents. Nested studies omit `kind`. |

Load documents use `type` or `kind`. Structural moment loads use `type: moment` with `vector: [mx, my, mz]` in N*m. Direct moment loads are valid only for structural regions whose elements provide rotational DOFs; non-structural run families and solid-only displacement regions reject them during validation or assembly.

Rotational structural runs may return `structural.rotation` and `structural.reaction_moment` fields. Beam-specific moment resultants use `structural.beam_torsion_moment` and `structural.beam_bending_moment`; shell-specific resultants use `structural.shell_bending_moment`.

## Runtime Config

Use `[runtime.fea]`:

| Key | Purpose |
| --- | --- |
| `artifact_store` | `in_memory` or `filesystem` run artifact store. Defaults to `filesystem` when omitted by the CLI/runtime bootstrap. |
| `artifact_root` | Filesystem root for persisted run artifacts. Defaults to `artifacts`. |
| `artifact_max_runs` | Optional global retained run limit. |
| `artifact_max_runs_per_kind` | Optional retained run limit per family. |
| `study_artifact_root` | Study validate, plan, run, and sweep artifact root. |

Preferred environment variables:

| Environment variable | Purpose |
| --- | --- |
| `RUNMAT_FEA_ARTIFACT_STORE` | `in_memory` or `filesystem` run artifact store fallback. |
| `RUNMAT_FEA_ARTIFACT_ROOT` | Run artifact root. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS` | Optional global retained run limit. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND` | Optional retained run limit per family. |
| `RUNMAT_FEA_STUDY_ARTIFACT_ROOT` | Study validate, plan, run, and sweep artifact root. |

## Evolution Rules

1. Add fields instead of changing field meaning.
2. Preserve stable error codes and quality reason codes.
3. Keep operation-specific validation failures typed as operation errors.
4. Persist enough provenance to explain backend and solver policy choices.
5. Update status, V&V docs, and tests when a family changes support level.
6. Version-bump an operation if a payload or semantic break is unavoidable.
