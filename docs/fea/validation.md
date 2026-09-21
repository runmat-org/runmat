# Verification & Validation

RunMat uses validation, tests, solver diagnostics, benchmark fixtures, and governance records to decide how much trust a FEA workflow has earned.

The purpose of validation is to determine whether the implementation, numerical method, and physics model produce defensible results for a documented use.

## Correctness Layers

| Layer | Question | Evidence |
| --- | --- | --- |
| Contract correctness | Does the public operation behave as specified? | Versioned envelopes, typed errors, payload snapshots, builtin tests. |
| Input validation | Is the model or typed `.fea` source well formed? | `runmat check`, `fea.validate_study/v1`, model validation, issue codes. |
| Solver implementation verification | Does the implementation solve the intended discrete problem? | Residual checks, deterministic replay, convergence diagnostics, backend provenance. |
| Numerical verification | Does the discretization converge toward known answers? | Analytic cases, manufactured solutions, patch tests, mesh convergence, time-step convergence. |
| Physics validation | Does the model match trusted external behavior? | Literature cases, independent solver comparisons, standard benchmarks, experimental references. |
| Production readiness | Are the checks repeatable and governed for a defined use class? | CI gates, readiness reports, threshold ratchets, owners, documented boundaries. |

No single test makes a FEA family production-grade. Readiness is earned per family and per intended use.

Model validation runs before formulation admission or solver progress. It rejects non-finite or
inadmissible constitutive properties, duplicate materials, and material-domain gaps, overlaps,
stale selectors, wrong formulation roles, or cross-geometry assignments. It also rejects invalid
typed domain reference states and mismatched signal, excitation, constraint, boundary, and
interface quantities. A deliberate expected-versus-assigned material mismatch is
valid when both materials exist; its confidence is retained as quality evidence. Invalid material
data and invalid assignments use distinct typed diagnostics, so a solver failure is not used as an
input-validation mechanism.

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
| `crates/runmat-runtime/src/analysis/tests.rs` | Runtime study, sweep, run, result, artifact, option, and `.fea` parsing behavior. |
| `crates/runmat-runtime/tests/operation_contracts.rs` | Public operation contracts, error codes, result queries, quality propagation, provenance, and study artifacts. |
| `crates/runmat-runtime/tests/analysis` | Conformance manifest and benchmark report harness across the supported families. |
| `scripts/fea/governance` | Benchmark report validation, external references, readiness, thresholds, and promotion calibration. |

These checks provide substantial contract and component evidence, but component tests do not promote a
whole-study workflow. The public typed `.fea` executor currently admits structural procedure graphs
only. Other physics families have checkable language contracts and varying amounts of solver-component
evidence; they remain unavailable through the native whole-study entry point until that complete path is
implemented and qualified. L3-L5 additionally require family-specific known-answer, convergence,
external-reference, and governance evidence.

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

| Family | Public whole-study state | Main open gate |
| --- | --- | --- |
| Structural | Executable with formulation-specific experimental or production admission | Close the recorded result/recovery defects, then qualify each formulation and execution envelope independently. |
| Thermal and thermo-mechanical | Checkable; native typed execution unavailable | Connect geometry, properties, procedures, recovery, publication, local/remote execution, and governed validation evidence end to end. |
| Electro-thermal and electromagnetic | Checkable; native typed execution unavailable | Complete the same whole-study path plus conservation, source/boundary realization, convergence, and independent references. |
| Acoustic | Checkable; native typed execution unavailable | Complete the whole-study path plus impedance/radiation, convergence, and independent acoustic references. |
| Fluid, conjugate heat transfer, and fluid-structure interaction | Checkable; native typed execution unavailable | Complete the whole-study path plus conservation, coupling/interface, canonical benchmark, scale, and independent-reference evidence. |

Within the structural family, static, modal, transient, harmonic, spectrum, random-response,
prestress-dependent, nonlinear, and buckling paths do not share one maturity label. Consult the
formulation requirement and emitted evidence for the exact run, and see [Current Status](./status)
for the current execution boundary.

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
| Input validation | Syntax, types, units, missing materials, missing loads, incompatible options, and invalid `.fea` sources. |
| Evidence | Artifacts, diagnostics, quality reasons, provenance, benchmark reports, readiness records. |
| Verification | Showing the implementation solves the intended discrete equations correctly. |
| Validation | Showing the implemented model matches known physical or trusted external behavior for a defined use. |
| Production ready | A documented use class with validation records, release gates, accepted tolerances, and known boundaries. |
