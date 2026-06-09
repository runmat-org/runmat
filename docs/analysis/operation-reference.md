---
title: "Analysis Operation Reference"
category: "Analysis & Simulation"
section: "13.7"
last_updated: "June 9, 2026"
---

# Operation Reference

Hosts call geometry and analysis through versioned runtime operation envelopes. This reference lists the envelope shape, operation families, artifact roots, and evolution rules.

For the end-to-end analysis flow, start with [FEA and Math on Geometry](/docs/runtime/analysis). For setup options, see [Using Analysis](/docs/runtime/analysis/using-analysis). For interpretation and support status, see [Results & Trust](/docs/runtime/analysis/trust) and [Current Status](/docs/runtime/analysis/status).

## Envelopes

Successful operations return `OperationEnvelope<T>`:

| Field | Meaning |
| --- | --- |
| `operation` | Stable operation family name, such as `analysis.run_modal`. |
| `op_version` | Versioned operation identifier, such as `analysis.run_modal/v1`. |
| `trace_id` | Optional host trace identifier. |
| `request_id` | Optional host request identifier. |
| `data` | Typed operation payload. |

Operation failures return `OperationErrorEnvelope`:

| Field | Meaning |
| --- | --- |
| `error_code` | Stable machine-readable code. |
| `error_type` | `input`, `validation`, `capacity`, `backend`, `internal`, or `contract`. |
| `message` | Human-readable message. |
| `operation`, `op_version` | Operation identity that failed. |
| `retryable` | Whether retrying may be reasonable. |
| `severity` | `warning`, `error`, or `fatal`. |
| `context` | String key/value details for automation and diagnostics. |
| `trace_id`, `request_id`, `timestamp` | Correlation and timing metadata. |

The shared implementation lives in `crates/runmat-runtime/src/operations.rs`.

## Geometry Operations

| Operation version | Payload role |
| --- | --- |
| `geometry.inspect/v1` | Detect supported input format and byte count before loading. |
| `geometry.load/v1` | Import geometry bytes into a `GeometryAsset`. |
| `geometry.compute_stats/v1` | Return geometry statistics for an asset. |
| `geometry.list_regions/v1` | Return known geometry regions. |
| `geometry.query_entities/v1` | Return region, mesh, or entity references with a bounded query limit. |
| `geometry.capture_view/v1` | Capture a geometry view through the installed capture adapter, with fallback behavior. |
| `geometry.prep_for_analysis/v1` | Produce a prep artifact and `MeshingPrepResult`. |
| `geometry.prep_artifact_health/v1` | Report prep artifact counts, ages, metrics, and optional per-geometry entries. |

## Analysis Operations

| Operation version | Payload role |
| --- | --- |
| `analysis.create_model/v1` | Build a solver-agnostic `AnalysisModel` from geometry and a profile intent. |
| `analysis.validate/v1` | Validate model units, frame, materials, loads, boundary conditions, domains, and geometry compatibility. |
| `analysis.run_linear_static/v1` | Run the structural linear-static path. |
| `analysis.run_modal/v1` | Run modal analysis. |
| `analysis.run_acoustic/v1` | Run the acoustic harmonic path. |
| `analysis.run_thermal/v1` | Run standalone thermal analysis. |
| `analysis.run_transient/v1` | Run structural transient analysis. |
| `analysis.run_nonlinear/v1` | Run nonlinear structural analysis. |
| `analysis.run_electromagnetic/v1` | Run electromagnetic analysis. |
| `analysis.run_cfd/v1` | Run the CFD baseline path. |
| `analysis.run_cht/v1` | Run the coupled CFD plus thermal path. |
| `analysis.run_fsi/v1` | Run the coupled structural transient plus CFD path. |
| `analysis.results/v1` | Query fields, diagnostics, payload subsets, quality reasons, provenance, and summaries. |
| `analysis.results_compare/v1` | Compare selected fields between two persisted runs. |
| `analysis.trends/v1` | Summarize persisted runs by `AnalysisRunKind`. |

## Study Operations

| Operation version | Payload role |
| --- | --- |
| `analysis.validate_study/v1` | Validate one `AnalysisStudySpec` and persist evidence. |
| `analysis.plan_study/v1` | Produce the operation sequence, run operation identity, fingerprint, and evidence path. |
| `analysis.run_study/v1` | Execute the planned study and return run identity, quality, provenance, and evidence path. |
| `analysis.validate_study_sweep/v1` | Validate a set of studies with aggregate and per-study issues. |
| `analysis.plan_study_sweep/v1` | Plan a sweep with plan entries and failure entries. |
| `analysis.run_study_sweep/v1` | Execute a deterministic sequential sweep. |

## Artifact Roots

| Environment variable | Purpose |
| --- | --- |
| `RUNMAT_ANALYSIS_ARTIFACT_ROOT` | Analysis run artifact root. |
| `RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT` | Study validate, plan, run, and sweep evidence root. |
| `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT` | Geometry prep artifact root. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS` | Optional global prep artifact retention limit. |
| `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY` | Optional per-geometry prep artifact retention limit. |
| `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS` | Optional prep artifact age retention limit. |

## Evolution Rules

1. Add fields instead of changing field meaning.
2. Preserve stable error codes and quality reason codes.
3. Keep operation-specific validation failures typed as operation errors.
4. Persist enough provenance to explain backend and solver policy choices.
5. Update status, governance, and tests when a domain changes support level.
6. Version-bump an operation if a payload or semantic break is unavoidable.
