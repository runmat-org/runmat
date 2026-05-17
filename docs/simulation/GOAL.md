# Analysis System End-State Goal

Last updated: 2026-05-17

## Purpose

Define the single, consolidated end-state goal for RunMat's FEA/simulation system, including:

- target product shape,
- non-negotiable architecture rules,
- completion criteria,
- current progress with explicit checkmarks.

This is an internal engineering target document.

## Source Documents Reviewed

- `docs/simulation/ARCHIVE/legacy-2026-05-17/analysis-physics-domain-coverage-and-migration.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/fea-rival-gap-matrix.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/multi-physics-parity-roadmap.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/code-reviewed-fea-workflow.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/m6-industrial-credibility-benchmark-spec.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/geo-and-analysis.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/prep-aware-solves.md`
- `docs/simulation/ARCHIVE/legacy-2026-05-17/prep-for-analysis.md`

## North-Star Outcome

RunMat is a deterministic, programmatic, code-reviewed simulation platform where engineers and agents can author studies, run trusted multiphysics solves, and make release decisions from typed evidence artifacts and governance signals, with domain depth and verification quality strong enough to be credibly compared against industrial incumbents for prioritized workflows.

## End-State Product Definition

1. Programmatic-first study workflow is canonical.
2. Runtime operation contracts are versioned, stable, and additive.
3. Physics ownership is model/material/interface-owned, not run-option-owned.
4. Solvers are deterministic-capable and provenance-complete.
5. Results, diagnostics, trends, and readiness signals are first-class API surfaces.
6. Benchmark and external-reference governance blocks promotion when confidence degrades.
7. System scales from desktop workflows to larger GPU/HPC-oriented runs without contract churn.

## Non-Negotiable Rules

1. Determinism first: reproducible mode must be replay-stable for the same inputs.
2. Additive contract evolution only: no silent breaking payload changes.
3. One-way layering: runtime orchestrates; physics math lives in analysis/fea.
4. Typed errors and reason codes are stable machine interfaces.
5. Governance is mandatory on protected branches; no silent bypass.
6. Documentation and implementation must be co-updated.

## Canonical Architecture End State

| Layer | End-state responsibility |
| --- | --- |
| Authoring | `study.yaml`/JSON as source of truth, include/compose, validate/plan/run builtins |
| Runtime contracts | `analysis.*` and `geometry.*` typed envelopes, typed errors, run artifact persistence |
| Analysis core | Solver-agnostic model schema and validation |
| Analysis FEA | Assembly, solver policies, physics-family modules, diagnostics |
| Governance | Conformance, trends, release readiness, external-reference comparators |
| Evidence plane | Immutable run artifacts + `run.data` dataset + lineage indices |

## Capability Completion Matrix

Legend: `[x]` complete baseline, `[~]` in active deepening, `[ ]` not started.

| Capability family | End-state target | Status |
| --- | --- | --- |
| Contract spine (`analysis.create_model/validate/run/results/trends`) | Stable, versioned, additive, typed | [x] |
| Structural core (linear/modal/transient/nonlinear) | Production-grade robustness + diagnostics + policy gating | [x] |
| Standalone thermal | Constitutive-depth thermal solve with governance | [~] |
| Thermo-mechanical coupling | Credible coupled behavior with benchmarked acceptance | [~] |
| Electro-thermal coupling | Credible Joule-coupled behavior with benchmarked acceptance | [~] |
| Plasticity depth | Constitutive realism beyond baseline stress proxies | [~] |
| Contact depth | Robust contact formulations and nonlinear stability | [~] |
| Maxwell EM | Static/harmonic to transient/frequency-dependent, reference-benchmarked | [~] |
| CFD core | Steady/transient fluid solver family | [~] |
| CHT | Fluid-thermal coupled family | [~] |
| FSI | Coupled structural-fluid family | [ ] |
| Acoustics | Acoustic solver family | [ ] |
| Meshing/adaptivity | Production mesh generation/refinement pipeline | [ ] |
| CAD interop depth | Robust CAD-native ingestion + metadata fidelity | [~] |
| Performance/scale | Scale SLOs and regression-gated performance | [~] |
| External-reference credibility (M6) | Protected-branch enforced industrial reference comparators | [~] |
| Programmatic UX productization | Reusable study templates, sweep orchestration, result extraction ergonomics | [~] |

## EM Program Snapshot (Current)

| EM track | Definition | Status |
| --- | --- | --- |
| Phase 0-4 | Contract scaffolding through first weak-form-style assembled static path | [x] |
| Phase 5-15 | Material/source/boundary governance depth + coupled harmonic/block solves | [x] |
| Phase 16 | Frequency sweep and resonance readiness governance surfaces | [x] |
| Next EM slices | Frequency-dependent materials, deeper Maxwell fidelity, transient EM, stronger external references | [~] |

## Definition of Success (Binary)

The analysis system is considered at target for a domain only when all conditions below are true for that domain:

1. Versioned operation contracts are stable and backward-compatible.
2. Canonical schema ownership is enforced (model/material/interface).
3. Deterministic replay and provenance are demonstrably stable.
4. Domain-native diagnostics and quality reasons are surfaced in results/trends.
5. Reference benchmarks pass defined acceptance envelopes.
6. Trend and drift governance gates are active in CI/release policy.
7. External-reference comparators are enforced where required.

## Program-Level Exit Criteria

RunMat's FEA/simulation program reaches its target state when all are true:

1. Prioritized domain set is production-credible:
   Structural, thermal/thermo, electro-thermal, plastic/contact, Maxwell EM.
2. Major expansion families have baseline operational coverage:
   CFD core plus at least one additional coupled family (CHT or FSI).
3. Study workflow is first-class:
   `analysis.validate_study`, `analysis.plan_study`, `analysis.run_study` and canonical evidence outputs are standard paths.
4. Governance is release-blocking:
   protected branches fail on conformance/readiness/external-reference violations.
5. Scale posture is regression-protected:
   performance SLOs and slowdown gates are active for key workloads.

## Near-Term Execution Order

1. Complete EM realism slices (frequency-dependent constitutive behavior, stronger Maxwell-form fidelity, external-reference checks).
2. Close remaining in-scope depth gaps (thermal/thermo, electro-thermal, plastic/contact realism).
3. Land first missing major family (CFD core) under existing contract/governance discipline.
4. Extend coupled-family coverage (CHT, then FSI) with benchmark gates before parity claims.

## Change Control

Any change to this goal document must:

1. update status checkmarks with explicit rationale,
2. link to the implementing code/docs/tests in the same PR or commit series,
3. preserve binary completion criteria and avoid ambiguous language.
