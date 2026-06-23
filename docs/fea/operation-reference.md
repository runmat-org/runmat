---
title: "Operation Reference"
category: "FEA"
section: "13.10"
last_updated: "June 22, 2026"
---

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
| `geometry.prep_for_analysis/v1` | Produce a prep artifact and `MeshingPrepResult`. |
| `geometry.prep_artifact_health/v1` | Report prep artifact counts, ages, metrics, and lifecycle data. |

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
| `fea.compare(baselineRunId, candidateRunId, Name, Value, ...)` | Compare two persisted runs. |
| `fea.trends(Name, Value, ...)` | Summarize recent persisted runs. |

`fea.Study` and `fea.Sweep` objects expose `validate`, `plan`, and `run` methods through the class system. `fea.RunResult` exposes `results` and `field`; `fea.Results` exposes `field`.

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
| `geometry_prep_artifact_root` | Geometry prep artifact root. |
| `geometry_prep_max_artifacts` | Optional global prep artifact retention limit. |
| `geometry_prep_max_artifacts_per_geometry` | Optional per-geometry prep artifact retention limit. |
| `geometry_prep_max_age_seconds` | Optional prep artifact age retention limit. |
| `geometry_prep_require_latest_revision` | Whether prep-aware runs reject stale geometry revisions. |
| `thermo_field_artifact_root` | Thermo-field artifact root for coupled thermal paths. |

Preferred environment variables:

| Environment variable | Purpose |
| --- | --- |
| `RUNMAT_FEA_ARTIFACT_STORE` | `in_memory` or `filesystem` run artifact store fallback. |
| `RUNMAT_FEA_ARTIFACT_ROOT` | Run artifact root. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS` | Optional global retained run limit. |
| `RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND` | Optional retained run limit per family. |
| `RUNMAT_FEA_STUDY_ARTIFACT_ROOT` | Study validate, plan, run, and sweep artifact root. |
| `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT` | Geometry prep artifact root. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS` | Optional global prep artifact retention limit. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY` | Optional per-geometry prep artifact retention limit. |
| `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS` | Optional prep artifact age retention limit. |
| `RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION` | Whether prep-aware runs reject stale revisions. |
| `RUNMAT_THERMO_FIELD_ARTIFACT_ROOT` | Thermo-field artifact root. |

Legacy `RUNMAT_ANALYSIS_*` variables remain compatibility fallbacks where supported.

## Evolution Rules

1. Add fields instead of changing field meaning.
2. Preserve stable error codes and quality reason codes.
3. Keep operation-specific validation failures typed as operation errors.
4. Persist enough provenance to explain backend and solver policy choices.
5. Update status, V&V docs, and tests when a family changes support level.
6. Version-bump an operation if a payload or semantic break is unavoidable.
