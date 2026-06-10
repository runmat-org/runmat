---
title: "FEA Verification & Validation"
category: "Analysis & Simulation"
section: "13.7"
last_updated: "June 9, 2026"
---

# FEA Verification & Validation

RunMat uses several kinds of checks to decide whether an FEA path is correct enough to trust.

Input validation checks whether a model or study is well formed. Run records describe what happened during execution. FEA verification and validation answer the harder question: whether the math, discretization, solvers, backend behavior, and supported physics families produce defensible results within documented limits.

## What Correct Means

Correctness has several layers:

| Layer | Question | Examples |
| --- | --- | --- |
| Contract correctness | Did the public operation behave as specified? | Stable envelopes, typed errors, payload snapshots, builtin registration tests. |
| Input validation | Did the model contain the required units, materials, loads, constraints, domains, and steps? | `analysis.validate/v1`, operation-specific validation, study issue codes. |
| Solver implementation verification | Did the implementation solve the intended discrete problem? | Residual checks, deterministic replay, CPU/GPU parity, fallback provenance, solver diagnostics. |
| Numerical verification | Does the discretization converge toward known answers? | Analytic cases, manufactured solutions, patch tests, mesh and time-step convergence. |
| Physics validation | Does the model match trusted external reality for the intended domain? | Literature cases, independent solver comparisons, calibrated reference artifacts, experimental benchmarks where available. |
| Production readiness | Are the checks repeatable, governed, and documented enough to release? | CI gates, benchmark reports, threshold ratchets, readiness reports, status updates, documented boundaries. |

No single test proves production-grade FEA. A family becomes production-grade only when the relevant layers are covered for its intended use.

## Maturity Levels

Use these levels when updating [Current Status](/docs/runtime/analysis/status):

| Level | Meaning | Minimum bar |
| --- | --- | --- |
| L0 Contracted | The operation surface exists. | Versioned operation, typed input errors, stable payload shape. |
| L1 Regression-tested | The path is covered by deterministic fixtures. | Replay stability, fixture coverage, expected publishability, contract tests. |
| L2 Solver-behavior verified | Solver behavior is instrumented and checked. | Residuals, convergence, quality reasons, backend provenance, CPU/GPU parity or fallback checks where relevant. |
| L3 Numerically verified | The discrete method is checked against known answers. | Analytic, manufactured, patch, mesh-convergence, or time-step-convergence tests. |
| L4 Externally validated | Results are compared with trusted independent references. | External solver, literature, standard benchmark, or experimental comparison with accepted tolerances. |
| L5 Production-ready | The domain is releasable for documented use. | L4 evidence plus governed thresholds, CI enforcement, drift checks, owner-approved limits, and status documentation. |

The current analysis stack has meaningful L0 to L2 coverage across many paths. L3 to L5 must be earned per physics family and per use class.

## Current Checks

Current checks are spread across the Rust tests, conformance harness, and governance scripts:

| Source | What it checks |
| --- | --- |
| `crates/runmat-analysis/fea/src/tests.rs` | Core solver paths run, reject invalid fixtures, emit diagnostics, replay deterministically, and respect CPU/GPU parity policies for covered fixtures. |
| `crates/runmat-runtime/tests/operation_contracts.rs` | Runtime operations preserve contract shape, typed errors, artifacts, result queries, quality propagation, and backend provenance. |
| `crates/runmat-runtime/tests/analysis/manifest.rs` | The conformance manifest currently defines 60 analysis fixtures across structural, modal, transient, nonlinear, thermal, electromagnetic, acoustic, CFD, CHT, and FSI paths. |
| `crates/runmat-runtime/tests/analysis/mod.rs` | The conformance harness executes the manifest, writes `analysis-benchmark-report/v1`, and can compare against baseline and rolling reports. |
| `crates/runmat-runtime/tests/geometry_prep_conformance.rs` | Geometry prep is deterministic and preserves region mapping expectations for covered inputs. |
| `crates/runmat-runtime/tests/prep_solve_conformance.rs` | Prep artifacts can influence nonlinear solve profiles while staying within documented quality limits. |
| `scripts/analysis/governance/*` | Benchmark reports, external-reference artifacts, readiness, threshold ratchets, and promotion calibration are machine-checked. |
| `scripts/analysis/prep_calibration/*` and `scripts/analysis/thermo_artifacts/*` | Prep and thermo-field artifacts have drift, promotion, and validation workflows. |

These checks are valuable, but many of them cover contracts, regression behavior, diagnostics, and governance. Do not describe a family as physically validated unless it also has known-answer and independent-reference coverage.

## Current Gaps

The main production-grade gaps are:

- broader analytic and manufactured-solution tests,
- mesh-convergence and time-step-convergence studies,
- patch tests for structural element behavior,
- independent solver or literature comparisons for each promoted family,
- calibrated experimental or industry benchmark references where relevant,
- clearer acceptance tolerances tied to model class and unit system,
- status labels that distinguish operation support from numerical validation.

These gaps should be tracked as release blockers for any claim stronger than "contracted and regression-tested."

## Family Maturity

| Family | Current checks | Current V&V level | Missing for production |
| --- | --- | --- | --- |
| Linear static structural | Dedicated solver path, cantilever range checks, deterministic replay, CPU/GPU parity, quality gates, conformance fixtures. | L2, with limited known-answer checks. | Add analytic/patch benchmarks, mesh convergence, and independent reference comparisons. |
| Modal structural | Dedicated modal path, mode payloads, residuals, orthogonality and separation diagnostics, CPU/GPU fixture coverage. | L2. | Add known eigenfrequency references, modal convergence studies, and external solver comparisons. |
| Thermal standalone | Thermal operation path, thermal payloads, stability and constitutive diagnostics, fixture coverage. | L1 to L2. | Add heat-equation known-answer cases, transient/steady convergence, and external thermal references. |
| Structural transient | Dedicated transient path, residual and energy diagnostics, adaptive-step controls, CPU/GPU fixture coverage. | L2. | Add time-integration known-answer cases, conservation checks, and time-step convergence. |
| Nonlinear structural | Nonlinear path, convergence diagnostics, plastic/contact proxy and reference fixtures, readiness/governance scripts. | L2. | Add independent nonlinear/plastic/contact reference solutions and calibrated tolerance envelopes. |
| Thermo-mechanical | Coupled thermal/structural contexts, thermal field artifacts, coupling diagnostics, readiness signals. | L1 to L2. | Add coupled known-answer cases and independent thermo-mechanical references. |
| Electro-thermal | Coupled electro-thermal contexts and Joule-coupling diagnostics. | L1 to L2. | Add coupled electrical/thermal known-answer cases and independent references. |
| Electromagnetic | Electromagnetic operation path, field-proxy payloads, sweep/resonance metrics, governance thresholds. | L2 for proxy behavior. | Add Maxwell field validation, boundary/source realization references, and larger workload studies. |
| Acoustic harmonic | Baseline path backed by modal behavior and acoustic diagnostics. | L1 to L2 baseline. | Add acoustic-specific solver validation, analytic acoustic cases, and external acoustic references. |
| CFD | Baseline path backed by transient execution and CFD diagnostics. | L1 to L2 baseline. | Add fluid solver validation, canonical CFD benchmarks, conservation checks, and external comparisons. |
| CHT | Coupled baseline path combining CFD diagnostics with thermal/transient payloads. | L1 to L2 baseline. | Add coupled fluid/thermal validation and independent CHT references. |
| FSI | Coupled baseline path backed by transient execution and CFD/FSI diagnostics. | L1 to L2 baseline. | Add two-way FSI validation, canonical benchmarks, and independent references. |

A runtime path can be useful and governed without being production-grade physics.

## Required Production Gates

Before a family is marked production-ready for a defined use class, it needs:

1. Stable operation contracts and typed errors.
2. Deterministic fixture coverage for valid and invalid inputs.
3. Residual, convergence, conservation, or family-specific solver diagnostics.
4. CPU/GPU parity or documented fallback behavior where GPU execution is supported.
5. Known-answer or manufactured-solution coverage.
6. Mesh, time-step, or parameter convergence where applicable.
7. Independent reference comparison with accepted tolerances.
8. Benchmark report generation and validation.
9. Readiness gate and threshold-ratchet coverage.
10. Documentation of assumptions, unsupported cases, and current boundaries.

If a family lacks one of these, [Current Status](/docs/runtime/analysis/status) should name the gap.

## Adding A Validation Case

Use this workflow when adding a new V&V case:

1. Define the engineering question and the physics family.
2. Choose the reference type: analytic, manufactured, patch, literature, independent solver, or experiment.
3. Record the reference source, model assumptions, units, geometry, material data, loads, constraints, and accepted tolerance.
4. Add or extend a fixture in `crates/runmat-analysis/fea` or the runtime analysis harness.
5. Assert physical metrics, not only payload shape. Examples: displacement magnitude, stress range, modal frequency, residual norm, energy growth, temperature range, flux metric, conservation error.
6. Add convergence checks when the case depends on mesh density, time step, mode count, or nonlinear increment policy.
7. Emit the metric into the benchmark report if it should be governed over time.
8. Add or update governance validation when promotion depends on the metric.
9. Update [Current Status](/docs/runtime/analysis/status) with the new V&V level and remaining boundary.

## Naming Rule

Use these terms precisely:

| Term | Use for |
| --- | --- |
| Input validation | Schema, units, missing materials, missing loads, incompatible options, invalid study files. |
| Evidence | Artifacts, diagnostics, quality reasons, provenance, benchmark reports, readiness records. |
| Verification | Showing the implementation solves the intended discrete equations correctly. |
| Validation | Showing the implemented model matches known physical or trusted external behavior for a defined use. |
| Production-ready | A documented use class with validation records, release gates, and known boundaries. |

## Update Rule

When solver behavior, fixture coverage, reference baselines, or support claims change, update this page and [Current Status](/docs/runtime/analysis/status) in the same change set. The update should say which V&V level changed, what records support the change, and which production blockers remain.
