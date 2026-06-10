---
title: "Results & Trust"
category: "Analysis & Simulation"
section: "13.8"
last_updated: "June 9, 2026"
---

# Results & Trust

A solve is useful only if the result can be inspected, explained, reproduced, and judged against the workflow's standard.

RunMat records those details in the run result, saved artifacts, and governance reports. For the stage-by-stage artifact flow, see [Evidence & Artifacts](/docs/runtime/analysis/evidence). For solver correctness and production-grade criteria, see [FEA Verification & Validation](/docs/runtime/analysis/validation).

## What A Run Returns

Every `analysis.run_*` operation returns `AnalysisRunResult`.

The common result includes:

- `run_id`,
- the underlying FEA run payload,
- optional domain payloads,
- `model_validity`, `solver_convergence`, and `result_quality`,
- `run_status`,
- `publishable`,
- `quality_reasons`,
- provenance.

Domain payloads carry family-specific fields such as modal frequencies, thermal fields, transient summaries, nonlinear convergence data, or electromagnetic field proxies and sweep metrics.

## Quality Gates

Quality gates tell callers how the run behaved:

| Field | Meaning |
| --- | --- |
| `model_validity` | Whether the model passed the relevant input and model validation checks. |
| `solver_convergence` | Whether the solver path converged under the selected policy. |
| `result_quality` | Whether domain-specific quality checks passed. |
| `run_status` | Final status such as publishable, degraded, or rejected. |
| `publishable` | Boolean summary for hosts that need a direct accept/reject signal. |
| `quality_reasons` | Stable machine-readable explanations for warnings, degradation, or rejection. |

Quality policy controls how strict the publishability decision is. A run can still produce useful diagnostics even when it is not publishable.

## Diagnostics

Diagnostics are domain-specific records emitted by the execution path. They are intended for both humans and automation:

- humans can inspect messages to understand weak setup or solver behavior,
- automation can track diagnostic codes and metrics over time,
- governance scripts can consume diagnostics as readiness signals.

Diagnostics should be read together with quality reasons. A diagnostic explains what was observed; a quality reason explains why that observation affected trust.

## Provenance

Provenance records how the run executed:

- requested backend,
- selected solver backend,
- precision mode,
- deterministic mode,
- solver method,
- preconditioner,
- fallback events.

This matters when comparing runs. Two results may use the same operation but different backend or fallback behavior.

## Compare And Trend

Use persisted run ids to inspect later:

| Operation | Use |
| --- | --- |
| `analysis.results/v1` | Query one run with optional payload filtering. |
| `analysis.results_compare/v1` | Compare two runs directly. |
| `analysis.trends/v1` | Summarize recent runs by domain family. |

Comparisons and trends help turn individual runs into workflow decisions.

## Governance Context

Governance uses run records for release and promotion decisions. It checks benchmark reports, reference comparisons, readiness, threshold ratchets, calibration reports, and missing records.

For the governance script map, see [Evidence & Artifacts](/docs/runtime/analysis/evidence). For production-readiness criteria by physics family, see [FEA Verification & Validation](/docs/runtime/analysis/validation).
