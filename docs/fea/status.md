---
title: "Current Status"
category: "FEA"
section: "13.9"
last_updated: "June 22, 2026"
---

# Current Status

This page states what the FEA system supports today, where support is limited, and what evidence backs each claim.

## Status Terms

| Term | Meaning |
| --- | --- |
| Works today | The operation exists, validates inputs, returns typed results or typed errors, and has automated coverage. |
| Evidence-backed | The path writes artifacts, diagnostics, provenance, quality reasons, or benchmark records. |
| Dedicated path | The operation has a family-specific execution path. |
| Limited | The path is usable within stated boundaries, but fidelity, scale, or validation depth is not production-grade for every use. |

V&V levels are defined in [Verification & Validation](/docs/fea/validation). The table below describes current evidence, not a full-family L2 release claim. A family is full L2 only when the whole exposed family contract has solver, field, diagnostic, invalid-case, and backend parity/fallback coverage.

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
| Geometry loading | Works today | `geometry.load/v1` tests | STL, OBJ, PLY, and glTF load without optional CAD dependencies. Source builds without OCCT load STEP metadata only. Official CLI and desktop builds enable OCCT, so STEP, IGES, and BREP import tessellated CAD topology. STEP and IGES also map XCAF ownership metadata when present. |
| Geometry summaries | Works today | runtime summary tests and Desktop contract wiring | Host clients receive source kind, assembly/material evidence when available, mesh counts/bounds, compact mapping summaries, semantic CAD region counts, and CAD region status. Full mapping ranges stay in runtime queries or plot scenes for scale. |
| Regions and entities | Works today | list/query operation tests | Mesh/group regions are format dependent. OCCT CAD import produces mapped face regions. STEP and IGES produce additional selectable semantic regions for CAD labels, bodies, components, assemblies, layers, colors, and materials when the exchange file contains that ownership data. |
| Statistics | Works today | geometry stats tests | Statistics are import-derived and do not prove model suitability. |
| View capture | Works today with adapter fallback | capture operation tests | Output depends on configured capture adapter. |
| Prep artifacts | Works today and evidence-backed | prep conformance and health tests | Prep carries deterministic surface topology summaries plus full element topology vectors for prepared mesh families. Meshing/adaptivity depth continues to expand. |
| Prep-aware runs | Works today and governed | stale/mismatch/untrusted-context tests | Requires matching geometry id and revision. |

## Physics Family Status

| Family | What works today | Evidence | V&V | Boundary |
| --- | --- | --- | --- | --- |
| Linear static structural | Dedicated structural linear-static run path, study support, prep-aware coordinate recovery, stress/strain/reaction/energy fields, known-answer work-energy gates, and result persistence. | Runtime contracts, solver tests, persisted result tests, GPU fallback/parity fixtures, external-reference checks, and report-validator gates. | L2 evidence in progress. | Needs broader analytic/patch families, convergence studies, and independent-reference evidence before full-family L2/production claims. |
| Modal structural | Dedicated modal run path with mode-shape, frequency, modal mass/stiffness, participation, residual, orthogonality, separation, and cluster fields. | Governed modal residual, orthogonality, separation, cluster, contract, and fixture tests. | L2 evidence in progress. | Needs dedicated repeated/near-repeated mode fixtures, known eigenfrequency references, convergence studies, and external solver comparisons before full-family L2/production claims. |
| Acoustic harmonic | Acoustic operation with damped Helmholtz domain-graph pressure solve, typed acoustic fields, frequency-response fields, residual diagnostics, material/source/boundary validation, and known-answer gates. | `fea.run_acoustic/v1` contracts, acoustic fixture coverage, external-reference checks, and report-validator gates. | L2 evidence in progress. | Needs broader impedance/radiation boundary depth, mesh-convergence evidence, and external acoustic references before full-family L2/production claims. |
| Thermal standalone | Thermal run path with temperature, element-domain gradient/heat-flux/source/boundary-flux fields, explicit source/boundary modeling, heat-balance diagnostics, and known-answer gates. | Thermal operation tests, stability/constitutive diagnostics, CPU/GPU fixture coverage, external-reference checks, and report-validator gates. | L2 evidence in progress. | Needs broader governed fixtures around sampled element-coordinate gradients, mesh-convergence studies, and external thermal comparisons before full-family L2/production claims. |
| Structural transient | Dedicated transient path with time controls, displacement/velocity/acceleration/von-Mises/energy/residual fields, and energy-balance diagnostics. | Transient contracts, residual/energy diagnostics, CPU/GPU fixture coverage. | L2 evidence in progress. | Needs time-integration known answers and time-step convergence before full-family L2/production claims. |
| Nonlinear structural | Dedicated nonlinear path with increment, line-search, plasticity/contact fields, prep-aware state recovery, contact/plastic known-answer gates, and typed invalid-case coverage. | Nonlinear contracts, reference fixtures, policy divergence tests, governance scripts, external-reference checks, and report-validator gates. | L2 evidence in progress. | Needs true contact-surface maps, broader nonlinear-law coverage, and independent nonlinear/plasticity/contact references before full-family L2/production claims. |
| Thermo-mechanical | Coupled thermal/structural fields across static, transient, and nonlinear paths with thermal strain/stress, temperature, displacement, von-Mises, and coupling-residual fields. | Thermo-field artifact tests, consistency diagnostics, coupling diagnostics, readiness signals, external-reference checks. | L2 evidence in progress. | Needs broader coupled known-answer cases and independent thermo-mechanical references before full-family L2/production claims. |
| Electro-thermal | Prep-artifact-backed electrical conductance topology with electric potential, E/J, Joule heat, temperature/residual fields, conservation diagnostics, invalid-case coverage, and governed Joule fixtures. | Domain validation, full prep topology gates, coupled diagnostics, external-reference checks, and release-readiness thresholds. | L2 evidence in progress. | Needs broader coupled electrical/thermal reference cases, conservation studies across more authored cases, mesh-convergence evidence, and independent references before full-family L2/production claims. |
| Electromagnetic | Frequency-domain Maxwell edge curl-curl harmonic path with oriented edge DOFs, full prep-backed edge/element incidence, gauge handling, complex vector/flux/E/H/J/loss/energy/Poynting fields, source/boundary/material validation, sweep/resonance metrics, and known-answer gates. | EM contracts, option validation, edge-topology/gauge/residual diagnostics, source/boundary/sweep known-answer gates, external-reference checks, and governance thresholds. | L2 evidence in progress. | Needs broader Maxwell validation cases, convergence studies, larger workload hardening, and independent references before full-family L2/production claims. |
| CFD | Finite-volume incompressible velocity-pressure path with authored inlet/outlet/wall boundary handling, prep-aware control-volume topology, cell-centered public fields, residuals, transient evolution, invalid-case coverage, and known-answer gates. | `fea.run_cfd/v1` contracts, CFD fixtures, external-reference checks, and report-validator gates. | L2 evidence in progress. | Needs broader canonical CFD benchmarks, conservation studies, independent fluid references, and backend parity beyond explicit fallback before full-family L2/production claims. |
| CHT | Coupled CFD plus thermal operation with authored CHT interfaces, prepared interface resolution, full prep topology driven interface graphs, separated fluid/solid temperatures, interface heat flux/jump fields, interface closure diagnostics, and known-answer gates. | `fea.run_cht/v1` contracts, CFD/thermal validation, coupled diagnostics, full topology report-validator gates, and external-reference checks. | L2 evidence in progress. | Needs broader conjugate heat-transfer benchmarks, mesh-convergence evidence, and independent CHT references before full-family L2/production claims. |
| FSI | Partitioned fluid-structure path with authored FSI interfaces, prepared interface resolution, full prep topology driven interface graphs, two-way pressure/displacement feedback, structural traction updates, face-domain pressure/traction, closure diagnostics, and known-answer gates. | `fea.run_fsi/v1` contracts, FSI fixture coverage, full topology report-validator gates, and external-reference checks. | L2 evidence in progress. | Needs broader two-way FSI benchmarks, mesh-convergence evidence, and independent FSI references before full-family L2/production claims. |

## Operation Coverage

| Operation family | Status |
| --- | --- |
| `geometry.inspect`, `geometry.load`, `geometry.compute_stats`, `geometry.list_regions`, `geometry.query_entities`, `geometry.capture_view` | Contracted and tested. |
| `geometry.prep_for_analysis`, `geometry.prep_artifact_health` | Contracted, tested, artifact-backed, and governance-linked. |
| `fea.create_model`, `fea.validate` | Contracted and tested. |
| `fea.run_linear_static`, `fea.run_modal`, `fea.run_thermal`, `fea.run_transient`, `fea.run_nonlinear`, `fea.run_electromagnetic` | Contracted, tested, artifact-backed, and dedicated or family-specific paths. |
| `fea.run_acoustic` | Contracted, tested, artifact-backed, and backed by acoustic-specific domain-graph, known-answer, and invalid-authoring gates. |
| `fea.run_cfd`, `fea.run_cht`, `fea.run_fsi` | Contracted, tested, artifact-backed governed fluid/coupled paths with typed fields, diagnostics, known-answer gates, and invalid-authoring/error coverage. |
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
