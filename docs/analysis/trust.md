---
title: "Results & Trust"
category: "Analysis & Simulation"
section: "13.5"
last_updated: "June 9, 2026"
---

# Results & Trust

A solve is useful only if the result can be inspected, explained, reproduced, and judged against the standard for the workflow.

RunMat keeps those concerns in the run result, persisted artifacts, and governance reports.

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
| `model_validity` | Whether the model passed the relevant validation checks. |
| `solver_convergence` | Whether the solver path converged under the selected policy. |
| `result_quality` | Whether domain-specific quality checks passed. |
| `run_status` | Final status such as publishable, degraded, or rejected. |
| `publishable` | Boolean summary for hosts that need a direct accept/reject signal. |
| `quality_reasons` | Stable machine-readable explanations for warnings, degradation, or rejection. |

Quality policy controls how strict the publishability decision is. A run can still produce useful diagnostic evidence even when it is not publishable.

## Diagnostics

Diagnostics are domain-specific evidence emitted by the execution path. They are intended for both humans and automation:

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

Comparisons and trends are the bridge between one run and a workflow-level trust decision.

## Governance Evidence

Governance turns tests, benchmark artifacts, reference comparisons, readiness checks, threshold ratchets, and calibration reports into promotion signals.

Gate families include:

| Gate family | Purpose |
| --- | --- |
| Contract conformance | Keep operation envelopes, typed errors, payloads, and artifacts stable. |
| Domain benchmark acceptance | Check fixture metrics against domain thresholds. |
| Benchmark schema validation | Reject malformed or incomplete reports before readiness evaluation. |
| External-reference comparison | Compare observed metrics against machine-checkable reference baselines. |
| Release readiness | Apply branch policy over reports, trends, missing fields, quality posture, and performance SLOs. |
| Threshold ratchet | Govern proposed threshold updates. |
| Promotion calibration | Check whether promotion policy has enough trusted history and coherent budgets. |
| Prep calibration | Track prep artifact quality, drift, recommendations, and promoted calibration evidence. |

## Script Map

| Script area | Role |
| --- | --- |
| `scripts/analysis/governance/validate_analysis_report_nonlinear.py` | Validate nonlinear and multiphysics benchmark reports. |
| `scripts/analysis/governance/generate_external_reference_benchmark.py` | Generate comparator artifacts from benchmark reports and references. |
| `scripts/analysis/governance/validate_external_reference_benchmark.py` | Enforce comparator coverage and pass/fail envelopes. |
| `scripts/analysis/governance/release_readiness_nonlinear.py` | Produce branch readiness verdicts and posture/trend summaries. |
| `scripts/analysis/governance/generate_threshold_ratchet_report.py` | Generate threshold ratchet proposals. |
| `scripts/analysis/governance/validate_threshold_ratchet_report.py` | Validate threshold ratchet reports. |
| `scripts/analysis/governance/generate_promotion_threshold_calibration.py` | Generate promotion calibration artifacts. |
| `scripts/analysis/governance/validate_promotion_threshold_calibration.py` | Validate promotion calibration artifacts. |
| `scripts/analysis/prep_calibration/*` | Evaluate prep drift, summarize prep artifacts, and promote calibration evidence. |
| `scripts/analysis/thermo_artifacts/*` | Generate, promote, and validate thermo field artifacts. |
| `scripts/analysis/reporting/*` | Summarize analysis reports and trend artifacts. |

## Release-Ready Means Evidence-Ready

A release-ready domain needs the operation path plus the evidence that reviewers and automation use to trust it:

1. stable versioned contracts,
2. model-owned schema for the relevant physics,
3. deterministic replay and provenance for representative runs,
4. domain-native diagnostics and quality reasons,
5. conformance fixtures with acceptance thresholds,
6. external-reference coverage where required,
7. readiness checks for posture, missing evidence, trends, and performance,
8. tests that prevent generator, validator, baseline, and readiness drift.
