---
title: "Current Status"
category: "FEA"
section: "13.9"
last_updated: "June 10, 2026"
---

# Current Status

This page states what the FEA system supports today, where support is limited, and what evidence backs each claim.

## Status Terms

| Term | Meaning |
| --- | --- |
| Works today | The operation exists, validates inputs, returns typed results or typed errors, and has automated coverage. |
| Evidence-backed | The path writes artifacts, diagnostics, provenance, quality reasons, or benchmark records. |
| Baseline path | The operation is contracted and tested but currently reuses a related solver path plus family diagnostics. |
| Dedicated path | The operation has a family-specific execution path. |
| Limited | The path is usable within stated boundaries, but fidelity, scale, or validation depth is not production-grade for every use. |

V&V levels are defined in [Verification & Validation](/docs/fea/validation).

## Workflow Status

| Question | Status | Boundary |
| --- | --- | --- |
| Can I run `.m` and `.fea` with the same command? | Yes. `runmat run` accepts `.m` scripts and `.fea` study/sweep files. | `.fea` files do not accept positional script arguments or bytecode emission. |
| Can I check before running? | Yes. `runmat check` supports `.m` scripts and `.fea` documents. | `.m` check compiles the source path; `.fea` check validates and plans but does not solve. |
| Can I use named entrypoints? | Yes. Project entrypoints can target `.m` or `.fea` paths. | Entrypoint resolution must resolve to an actual file. |
| Can I use RunMat code? | Yes. `geometry.inspect`, `geometry.load`, `geometry.listRegions`, typed `fea.*` constructors, workflow builtins, and result-query builtins are registered. | `.m` code and `.fea` files can express the same study content; use `.m` when generation or parameterization is the better shape. |
| Can I define studies in files? | Yes. `.fea` supports geometry loading, model scaffolds or explicit model data, run options, studies, and sweeps. | `.fea` is YAML with extension `.fea`; legacy `.study.yaml` contracts are no longer the user-facing format. |
| Can hosts call lower-level operations? | Yes. Runtime operations cover geometry, model creation, validation, run families, results, studies, and sweeps. | Rust function names still use internal `analysis_*` APIs, while public operation identifiers are `fea.*`. |
| Are artifacts configurable? | Yes. `[runtime.fea]` configures FEA run, study, prep, and thermo-field artifact roots. | Legacy environment variables remain as compatibility fallbacks but new docs use `RUNMAT_FEA_*`. |

## Geometry Status

| Capability | Status | Evidence | Boundary |
| --- | --- | --- | --- |
| Format inspection | Works today | `geometry.inspect/v1` tests | Detects supported families; deep CAD semantics depend on source data. |
| Geometry loading | Works today | `geometry.load/v1` tests | STL, STEP, OBJ, PLY, and glTF importer paths are covered; fidelity varies by format. |
| Regions and entities | Works today | list/query operation tests | Mesh-only inputs may not contain useful regions. |
| Statistics | Works today | geometry stats tests | Statistics are import-derived and do not prove model suitability. |
| View capture | Works today with adapter fallback | capture operation tests | Output depends on configured capture adapter. |
| Prep artifacts | Works today and evidence-backed | prep conformance and health tests | Meshing/adaptivity depth continues to expand. |
| Prep-aware runs | Works today and governed | stale/mismatch/untrusted-context tests | Requires matching geometry id and revision. |

## Physics Family Status

| Family | What works today | Evidence | V&V | Boundary |
| --- | --- | --- | --- | --- |
| Linear static structural | Dedicated structural linear-static run path, study support, result persistence, quality gates. | Runtime contracts, solver tests, persisted result tests, GPU fallback/parity fixtures. | L2, with limited known-answer checks. | Linear static assumptions only; needs broader analytic, patch, convergence, and independent-reference evidence for production claims. |
| Modal structural | Dedicated modal run path with modal payloads. | Modal residual, orthogonality, separation, contract, and fixture tests. | L2. | Needs known eigenfrequency references, modal convergence, and external solver comparisons. |
| Acoustic harmonic | Acoustic operation and typed acoustic diagnostics. | `fea.run_acoustic/v1` contracts and acoustic fixture coverage. | L1-L2 baseline. | Baseline is modal-response based; acoustic field propagation fidelity is not production-grade. |
| Thermal standalone | Thermal run path with thermal payloads and diagnostics. | Thermal operation tests, stability/constitutive diagnostics, fixture coverage. | L1-L2. | Needs heat-equation references, convergence, and external thermal comparisons. |
| Structural transient | Dedicated transient path with time controls and transient payloads. | Transient contracts, residual/energy diagnostics, CPU/GPU fixture coverage. | L2. | Needs time-integration known answers and time-step convergence. |
| Nonlinear structural | Dedicated nonlinear path with increment, line-search, plasticity/contact, and quality signals. | Nonlinear contracts, reference fixtures, policy divergence tests, governance scripts. | L2. | Needs independent nonlinear, plasticity, and contact references. |
| Thermo-mechanical | Coupled thermal/structural context can affect structural/transient/nonlinear paths. | Thermo-field artifact tests, coupling diagnostics, readiness signals. | L1-L2. | Needs coupled known-answer and independent thermo-mechanical references. |
| Electro-thermal | Electro-thermal domain context and Joule-coupling diagnostics exist. | Domain validation and coupled diagnostics coverage. | L1-L2. | Standalone electro-thermal solve depth and references remain limited. |
| Electromagnetic | Electromagnetic run path with source, boundary, material, harmonic, and sweep payloads. | EM contracts, option validation, sweep/resonance metrics, governance thresholds. | L2 for current proxy behavior. | Field quantities are proxy payloads; Maxwell validation and larger workload hardening remain limited. |
| CFD | CFD operation with domain validation and flow diagnostics. | `fea.run_cfd/v1` contracts and CFD fixtures. | L1-L2 baseline. | Fluid behavior is baseline diagnostic/payload coverage, not full Navier-Stokes validation. |
| CHT | Coupled CFD plus thermal operation with CHT diagnostics. | `fea.run_cht/v1` contracts, CFD/thermal validation, coupled diagnostics. | L1-L2 baseline. | Dedicated coupled fluid/thermal field fidelity remains limited. |
| FSI | Coupled structural transient plus CFD operation with FSI diagnostics. | `fea.run_fsi/v1` contracts and FSI fixture coverage. | L1-L2 baseline. | Full two-way FSI field solving is outside the current baseline. |

## Operation Coverage

| Operation family | Status |
| --- | --- |
| `geometry.inspect`, `geometry.load`, `geometry.compute_stats`, `geometry.list_regions`, `geometry.query_entities`, `geometry.capture_view` | Contracted and tested. |
| `geometry.prep_for_analysis`, `geometry.prep_artifact_health` | Contracted, tested, artifact-backed, and governance-linked. |
| `fea.create_model`, `fea.validate` | Contracted and tested. |
| `fea.run_linear_static`, `fea.run_modal`, `fea.run_thermal`, `fea.run_transient`, `fea.run_nonlinear`, `fea.run_electromagnetic` | Contracted, tested, artifact-backed, and dedicated or family-specific paths. |
| `fea.run_acoustic`, `fea.run_cfd`, `fea.run_cht`, `fea.run_fsi` | Contracted, tested, artifact-backed baseline paths. |
| `fea.results`, `fea.results_compare`, `fea.trends` | Contracted and tested. |
| `fea.validate_study`, `fea.plan_study`, `fea.run_study` | Contracted, tested, and artifact-backed. |
| `fea.validate_study_sweep`, `fea.plan_study_sweep`, `fea.run_study_sweep` | Contracted, tested, and artifact-backed. |
| RunMat builtins | Geometry builtins, typed `fea.*` constructors, workflow builtins, and result-query builtins are registered and covered by focused tests. |

## Verification Commands

Focused checks for this surface:

```sh
cargo test -p runmat-runtime --lib analysis:: --no-fail-fast
cargo test -p runmat-runtime --test operation_contracts --no-fail-fast
cargo test -p runmat-runtime --test analysis --no-fail-fast
cargo test -p runmat-runtime --lib builtins::fea --no-fail-fast
cargo test -p runmat-runtime --lib builtins::geometry --no-fail-fast
cargo test -p runmat-config --no-fail-fast
cargo check -p runmat-runtime -p runmat -p runmat-config
```

Full workspace verification:

```sh
cargo fmt -- --check
cargo test --workspace --all-targets --all-features --no-fail-fast
```

## Update Rule

When support changes, update this page in the same change set as the implementation and tests. Say which workflow stage or family changed, what evidence supports the new claim, and whether the V&V level changed.
