---
title: "Verification & Validation"
category: "FEA"
section: "13.7"
last_updated: "June 22, 2026"
---

# Verification & Validation

RunMat uses validation, tests, solver diagnostics, benchmark fixtures, and governance records to decide how much trust a FEA workflow has earned.

The purpose of validation is to determine whether the implementation, numerical method, and physics model produce defensible results for a documented use.

## Correctness Layers

| Layer | Question | Evidence |
| --- | --- | --- |
| Contract correctness | Does the public operation behave as specified? | Versioned envelopes, typed errors, payload snapshots, builtin tests. |
| Input validation | Is the model or `.fea` document well formed? | `runmat check`, `fea.validate_study/v1`, model validation, issue codes. |
| Solver implementation verification | Does the implementation solve the intended discrete problem? | Residual checks, deterministic replay, convergence diagnostics, backend provenance. |
| Numerical verification | Does the discretization converge toward known answers? | Analytic cases, manufactured solutions, patch tests, mesh convergence, time-step convergence. |
| Physics validation | Does the model match trusted external behavior? | Literature cases, independent solver comparisons, standard benchmarks, experimental references. |
| Production readiness | Are the checks repeatable and governed for a defined use class? | CI gates, readiness reports, threshold ratchets, owners, documented boundaries. |

No single test makes a FEA family production-grade. Readiness is earned per family and per intended use.

## Maturity Levels

Use these labels in [Current Status](/docs/fea/status):

| Level | Meaning | Minimum bar |
| --- | --- | --- |
| L0 | Contracted operation | Versioned operation, typed errors, stable payload shape. |
| L1 | Regression tested | Deterministic fixtures, valid/invalid cases, replay stability. |
| L2 | Solver behavior checked | Residuals, convergence, quality reasons, backend provenance, CPU/GPU parity or documented fallback. |
| L3 | Numerically verified | Known-answer, manufactured-solution, patch, mesh-convergence, or time-step-convergence evidence. |
| L4 | Externally validated | Independent solver, literature, standard benchmark, or experimental comparison with accepted tolerances. |
| L5 | Production ready | L4 evidence plus governed thresholds, CI enforcement, owner-approved limits, and documented use boundaries. |

## Current Test Sources

| Source | What it covers |
| --- | --- |
| `crates/runmat-analysis/fea` tests | Core FEA solver behavior, invalid fixtures, diagnostics, deterministic replay, backend parity/fallback behavior where covered. |
| `crates/runmat-runtime/src/analysis/tests.rs` | Runtime study, sweep, run, result, artifact, prep, option, and `.fea` parsing behavior. |
| `crates/runmat-runtime/tests/operation_contracts.rs` | Public operation contracts, error codes, result queries, quality propagation, provenance, and study artifacts. |
| `crates/runmat-runtime/tests/analysis` | Conformance manifest and benchmark report harness across the supported families. |
| `crates/runmat-runtime/tests/geometry_prep_conformance.rs` | Geometry prep determinism and region mapping expectations. |
| `crates/runmat-runtime/tests/prep_solve_conformance.rs` | Prep-aware solve behavior and documented quality limits. |
| `scripts/fea/governance` | Benchmark report validation, external references, readiness, thresholds, and promotion calibration. |
| `scripts/fea/prep_calibration` and `scripts/fea/thermo_artifacts` | Prep and thermo-field artifact drift, promotion, and validation workflows. |

These checks provide strong L0-L2 evidence across many paths. They do not by themselves declare a family full L2. Full-family L2 requires every exposed mode, material, boundary/source type, result field, invalid-case class, and backend parity/fallback path for that family to meet the L2 bar. L3-L5 require deeper family-specific known-answer, convergence, external-reference, and governance evidence.

## Current Gaps

The main gaps for production-grade claims are:

- broader analytic and manufactured-solution tests,
- structural patch tests,
- mesh-convergence studies,
- time-step-convergence studies for transient families,
- independent solver or literature comparisons for each promoted family,
- calibrated experimental or industry benchmark references where appropriate,
- clearer accepted tolerances by model class, unit system, and family,
- status labels that separate operation support from physical validation.

## Family Maturity Summary

| Family | Current maturity | Main missing evidence |
| --- | --- | --- |
| Linear static structural | L2 evidence in progress | Broader analytic and patch-test families, mesh convergence, and independent references. |
| Modal structural | L2 evidence in progress | Dedicated repeated/near-repeated mode fixtures, known eigenfrequency references, modal convergence studies, external solver comparisons. |
| Thermal standalone | L2 evidence in progress | Broader sampled element-gradient fixtures, mesh convergence, and external thermal references. |
| Structural transient | L2 evidence in progress | Time integration known answers, conservation checks, time-step convergence. |
| Nonlinear structural | L2 evidence in progress | True contact-surface maps, broader nonlinear-law coverage, and independent nonlinear, plasticity, and contact references with tolerance envelopes. |
| Thermo-mechanical | L2 evidence in progress | Broader coupled known-answer cases and independent thermo-mechanical references. |
| Electro-thermal | L2 evidence in progress | Broader coupled electrical/thermal references, conservation studies across more authored cases, mesh-convergence evidence, and larger workload studies. |
| Electromagnetic | L2 evidence in progress | Broader Maxwell field validation, source/boundary realization references, convergence studies, independent references, and larger workload studies. |
| Acoustic harmonic | L2 evidence in progress | Broader impedance/radiation validation, mesh convergence, and external acoustic references. |
| CFD | L2 evidence in progress | Canonical CFD benchmarks, conservation checks, independent fluid references, and GPU parity beyond explicit fallback. |
| CHT | L2 evidence in progress | Broader coupled fluid/thermal benchmarks, mesh-convergence evidence, and independent CHT references. |
| FSI | L2 evidence in progress | Broader two-way FSI benchmarks, mesh-convergence evidence, and independent FSI references. |

## Adding A V&V Case

When adding a new validation case:

1. Define the engineering question and physics family.
2. Choose the reference type: analytic, manufactured, patch, literature, independent solver, standard benchmark, or experiment.
3. Record geometry, units, materials, loads, constraints, assumptions, and accepted tolerance.
4. Add the fixture in the solver crate or runtime conformance harness.
5. Assert physical metrics, not only payload shape.
6. Add convergence checks when the case depends on mesh density, time step, mode count, or nonlinear increments.
7. Emit governed metrics into benchmark reports when the result should be tracked over time.
8. Add or update governance checks if promotion depends on the metric.
9. Update [Current Status](/docs/fea/status) with the new level and remaining boundary.

## Naming Rule

Use these terms precisely:

| Term | Use for |
| --- | --- |
| Input validation | Schema, units, missing materials, missing loads, incompatible options, invalid `.fea` documents. |
| Evidence | Artifacts, diagnostics, quality reasons, provenance, benchmark reports, readiness records. |
| Verification | Showing the implementation solves the intended discrete equations correctly. |
| Validation | Showing the implemented model matches known physical or trusted external behavior for a defined use. |
| Production ready | A documented use class with validation records, release gates, accepted tolerances, and known boundaries. |
