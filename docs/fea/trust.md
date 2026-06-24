---
title: "Results & Trust"
category: "FEA"
section: "13.8"
last_updated: "June 10, 2026"
---

# Results & Trust

A run result is useful when you can inspect it, explain it, reproduce it, and judge it against the workflow's standard.

RunMat records those signals in the result payload, saved artifacts, diagnostics, quality reasons, and provenance.

## What A Run Returns

Every run returns common result data:

- `run_id`,
- fields from the underlying FEA run payload,
- optional domain payloads,
- `model_validity`,
- `solver_convergence`,
- `result_quality`,
- `run_status`,
- `publishable`,
- `quality_reasons`,
- provenance.

Domain payloads can include modal frequencies, thermal summaries, transient snapshots, nonlinear convergence data, electromagnetic metrics, or coupled-family diagnostics.

## Quality Gates

| Field | Meaning |
| --- | --- |
| `model_validity` | Whether model validation passed for the run. |
| `solver_convergence` | Whether the solver path converged under the selected policy. |
| `result_quality` | Whether domain-specific quality checks passed. |
| `run_status` | Final status such as publishable, degraded, or rejected. |
| `publishable` | Boolean accept/reject signal for callers. |
| `quality_reasons` | Stable machine-readable reasons for warnings, degradation, or rejection. |

Quality policy controls how strict the publishability decision is. A run can return useful diagnostics even when it is not publishable.

## Diagnostics

Diagnostics explain observed behavior. They can describe solver fallback, modal residuals, nonlinear increment failures, electromagnetic source or boundary quality, thermal spread, CFD flow signals, and similar family-specific conditions.

Read diagnostics with quality reasons:

- diagnostics say what was observed,
- quality reasons say why that observation affected trust.

## Provenance

Provenance records how the run executed:

- requested backend,
- selected solver backend,
- precision mode,
- deterministic mode,
- solver method,
- preconditioner,
- fallback events.

This matters for comparisons. Two runs can use the same model and operation but differ in backend, precision, fallback behavior, or solver policy.

## Compare And Trend

Use persisted run ids to inspect later:

| Operation | Purpose |
| --- | --- |
| `fea.results/v1` | Query one run with optional payload filtering. |
| `fea.results_compare/v1` | Compare two persisted runs. |
| `fea.trends/v1` | Summarize recent runs by family. |

Comparisons and trends help turn individual solves into workflow decisions.

## Trust Decision

Before treating a result as reliable, ask:

1. Did the `.fea` file or model validate?
2. Did the planned operation match the intended family?
3. Did the selected backend and solver policy match expectations?
4. Did the result pass quality gates?
5. Are there quality reasons or diagnostics that affect use?
6. Is the family mature enough for this use class?
7. Is there V&V evidence for this model class, not just a successful run?
8. Are artifacts available for review and reproduction?

Use [Evidence & Artifacts](/docs/fea/evidence) for the record flow, [Verification & Validation](/docs/fea/validation) for correctness criteria, and [Current Status](/docs/fea/status) for current family boundaries.
