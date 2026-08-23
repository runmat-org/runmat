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
| `geometry.load(path)` | Read a geometry file and return a `geometry.Asset` object. |
| `geometry.listRegions(asset)` | Return imported regions from a `geometry.Asset`. |
| `geometry.meshes(asset)` | Return patch-ready surface mesh topology with vertices, faces/triangles, and region mappings. |
| `fea.load(path)` | Load a `.fea` study or sweep file and return `fea.Study` or `fea.Sweep`. |
| `fea.material(...)`, `fea.materialAssignment(...)` | Create typed material data and region assignments. |
| `fea.boundaryCondition(...)`, `fea.loadCase(...)` | Create typed constraints, loads, and sources. |
| `fea.step(...)`, `fea.domain(...)`, `fea.interface(...)` | Create typed analysis steps, physics domains, and interfaces. |
| `fea.runOptions(kind, Name, Value, ...)` | Create family-specific run options. |
| `fea.model(id, geometry, Name, Value, ...)` | Assemble an explicit model from geometry and typed components. |
| `fea.study(id, geometry, Name, Value, ...)` | Create a `fea.Study` from geometry, physics profile/model data, backend, and run options. |
| `fea.sweep(id, studies, Name, Value, ...)` | Create a deterministic sweep from `fea.Study` objects. |
| `fea.validate(study)` | Validate a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.plan(study)` | Plan a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.run(study)` | Run a `.fea` path, `fea.Study`, or `fea.Sweep`. |
| `fea.results(runOrId, Name, Value, ...)` | Load result data from a run result object or persisted run id. |
| `fea.field(resultsOrRun, fieldId)` | Extract one field from a result query. |
| `fea.plot(runOrResultsOrField, fieldId)` | Create a RunMat figure for a result field using the study geometry context. |
| `fea.compare(baselineRunId, candidateRunId)` | Compare two persisted runs; no Name, Value options are accepted. |
| `fea.trends(Name, Value, ...)` | Summarize recent persisted runs. |

`fea.boundaryCondition` is a RunMat-native constructor. Its seven numeric boundary forms accept finite real numeric scalars from all eight built-in integer classes and convert them once to the model's binary64 storage fields; scalar shape, finiteness, required fields, and unknown fields are validated before object construction, and invalid input is attributed to `fea.boundaryCondition` with `RunMat:fea:InvalidInput`.

`fea.material`, `fea.loadCase`, `fea.domain`, and `fea.interface` are also RunMat-native host constructors. Their documented finite real physical fields accept all eight built-in integer classes, preserve exact input through kind/field/shape validation, and then cross one binary64 model-storage boundary; sufficiently wide integers can round, and resident numeric fields are rejected without provider dispatch. The nested thermo-field-source revision is a structural nonnegative integer and remains exact.

`fea.materialAssignment`, `fea.model`, `fea.plan`, `fea.field`, `fea.plot`, and `fea.compare` accept structural objects and text rather than integer arrays. Discrete output metadata remains integer typed: model and geometry revisions, plan counts and indexes, field shape/component/element/size metadata, and comparison count deltas do not pass through a physical-value binary64 boundary. The conventional figure handle remains a double at the scripting boundary. Input arity and structural identity are validated before artifact lookup, planning, field projection, comparison, or figure creation.

These constructors, planning operations, field projections, comparisons, and plots do not intrinsically require native OCCT. Existing geometry assets, STL or synthetic geometry, and persisted mesh/result artifacts use their normal paths without OCCT; only document resolution that must import topology-bearing CAD may require an OCCT-enabled geometry backend.

### Workflow signatures and validation

| Builtin | Exact signature | Validation boundary |
| --- | --- | --- |
| `fea.step` | `step = fea.step(id, kind)` | Both inputs are text scalars; numeric inputs reject before construction. |
| `fea.study` | `study = fea.study(path)` or `study = fea.study(id, geometry, Name, Value, ...)` | The path must resolve to a study. Constructor identity, geometry class, option spelling, component classes, and profile/run-kind agreement are checked before construction. |
| `fea.sweep` | `sweep = fea.sweep(id, studies, Name, Value, ...)` | The id and study object classes are checked before `FailFast`; `FailFast` is a logical scalar and defaults to true. |
| `fea.validate` | `result = fea.validate(studyOrSweepOrPath)` | Object/path identity is checked before document validation; no solver executes. |
| `fea.run` | `run = fea.run(studyOrSweepOrPath)` | Object/path identity and document validity are checked before the selected solver executes. |
| `fea.results` | `results = fea.results(runOrRunId, Name, Value, ...)` | Run identity, option pairing, option values, and selector shape/range are checked before artifact lookup. An existing `fea.Results` object is accepted without options. |
| `fea.trends` | `trends = fea.trends(Name, Value, ...)` | `WindowSize` is checked before artifact-store access. |

`fea.study` defaults to profile `linear_static_structural`, backend `cpu`, defaults mode `profile_scaffold`, and a sanitized `<study-id>_model` model id when none is supplied. Its constructor options are `RunKind`/`Kind`, `Profile`, `Backend`, `ModelId`, `Model`, `Frame`, `Defaults`, `Materials`, `MaterialAssignments`/`Assignments`, `BoundaryConditions`/`BCs`, `Loads`/`LoadCases`, `Steps`, `Domains`, `Interfaces`, and `RunOptions`/`Options`. Direct integer values do not substitute for any identity or typed-object option.

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

`fea.Study` and `fea.Sweep` objects expose `validate`, `plan`, and `run` methods through the class system. `fea.RunResult` exposes `results`, `field`, and `plot`; `fea.Results` exposes `field` and `plot`; `fea.Field` exposes `plot`.

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

Load documents use `type` or `kind`. Structural moment loads use `type: moment` with `vector: [mx, my, mz]` in N*m. `type: torque` is accepted as an alias and resolves to the canonical moment load. Direct moment loads are valid only for structural regions whose elements provide rotational DOFs; non-structural run families and solid-only displacement regions reject them during validation or assembly.

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
| `thermo_field_artifact_root` | Thermo-field artifact root for coupled thermal paths. |

Preferred environment variables:

| Environment variable | Purpose |
| --- | --- |
| `RUNMAT_FEA_ARTIFACT_STORE` | `in_memory` or `filesystem` run artifact store fallback. |
| `RUNMAT_FEA_ARTIFACT_ROOT` | Run artifact root. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS` | Optional global retained run limit. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND` | Optional retained run limit per family. |
| `RUNMAT_FEA_STUDY_ARTIFACT_ROOT` | Study validate, plan, run, and sweep artifact root. |
| `RUNMAT_THERMO_FIELD_ARTIFACT_ROOT` | Thermo-field artifact root. |

## Evolution Rules

1. Add fields instead of changing field meaning.
2. Preserve stable error codes and quality reason codes.
3. Keep operation-specific validation failures typed as operation errors.
4. Persist enough provenance to explain backend and solver policy choices.
5. Update status, V&V docs, and tests when a family changes support level.
6. Version-bump an operation if a payload or semantic break is unavoidable.
