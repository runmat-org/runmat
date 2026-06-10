---
title: "Current Analysis Status"
category: "Analysis & Simulation"
section: "13.9"
last_updated: "June 9, 2026"
---

# Current Status

Current status is tracked across workflow stages and physics families. Each row states what works today, what records exist, how far V&V has progressed, and where support remains limited.

For artifacts and governance records behind these status calls, see [Evidence & Artifacts](/docs/runtime/analysis/evidence). For the V&V maturity ladder, see [FEA Verification & Validation](/docs/runtime/analysis/validation).

## Status Terms

| Term | Meaning |
| --- | --- |
| Works today | The operation path exists, validates inputs, returns typed results or typed errors, and has test coverage. |
| Evidence-backed | The path writes artifacts, diagnostics, provenance, quality reasons, or health data that reviewers can inspect. |
| Governed | Readiness, reference, threshold, calibration, or benchmark checks consume records from this path. |
| Limited | The path is usable within stated limits, but fidelity, format depth, scale, or release posture is not uniform across all possible inputs. |
| Dedicated solver-backed | The operation delegates to a solver pipeline for that family. |
| Baseline path | The operation is contracted and tested, but currently reuses a related solver path plus domain diagnostics and gates. |
| Coupled path | The operation combines domain contexts or solver payloads and emits coupling diagnostics; it may be assembled from multiple solver paths rather than one monolithic solver. |

## V&V Levels

| Level | Meaning |
| --- | --- |
| L0 | Contracted operation surface with typed errors and stable payload shape. |
| L1 | Deterministic regression fixtures and contract coverage. |
| L2 | Solver behavior checks such as residuals, convergence diagnostics, quality gates, backend parity, or fallback provenance. |
| L3 | Known-answer, manufactured-solution, patch, mesh-convergence, or time-step-convergence coverage. |
| L4 | Independent solver, literature, standard benchmark, or experimental reference comparison. |
| L5 | Production-ready for a documented use class, with governed thresholds, CI enforcement, owner-approved limits, and status documentation. |

## Workflow Status

| Stage | Status | Current boundary |
| --- | --- | --- |
| Drive from CLI | `runmat run` accepts `.m` files and analysis study files. Named entrypoints can target either kind of file. | Study-file execution runs the study or sweep payload; bytecode emission and positional script arguments remain `.m`-only. |
| Drive from host code | Runtime operation functions are public for geometry, model creation, validation, run execution, result queries, studies, and sweeps. | Hosts still use direct runtime APIs for fine-grained model construction, result queries, and custom orchestration. |
| Drive from study specs | `AnalysisStudySpec` and `AnalysisStudySweepSpec` are serializable JSON/YAML contracts with validate, plan, and run operations. | The runtime study spec embeds `GeometryAsset`; workflows that start with raw CAD or mesh paths need to load geometry first. |
| Drive from RunMat code | Geometry builtins `geometry_inspect` and `geometry_load` are registered. Study builtins `analysis_validate_study`, `analysis_plan_study`, and `analysis_run_study` are registered. | Direct `.m` access currently covers geometry file loading and study/sweep execution. Lower-level model construction and result query operations remain host/runtime surfaces. |
| Bring geometry in | Geometry inspect and load operations cover STL, STEP, OBJ, PLY, and glTF importer paths. | CAD semantic richness varies by format, importer, and source file. |
| Inspect geometry | Stats, region listing, bounded entity queries, and view capture are available. | Region fidelity and entity meaning depend on source geometry and importer detail. |
| Prepare geometry | Deterministic prep profiles produce analysis prep artifacts and health data. | Meshing, adaptive refinement, and topology depth continue to expand by profile. |
| Build models | Geometry can seed solver-agnostic analysis models through profile intents. | Generated models are scaffolds; meaningful runs still need intentional materials, loads, constraints, and domains. |
| Validate models | Model validation and operation-specific validation return typed issues and errors. | Validation covers schema and compatibility. Engineering fit still depends on the chosen model data and assumptions. |
| Run solves | Direct run operations exist for structural, modal, thermal, nonlinear, electromagnetic, acoustic, CFD, CHT, and FSI families. | Domain fidelity varies. Some families have deeper solver coverage; others are baseline paths. |
| Run studies | Study validate, plan, and run operations save artifacts and deterministic fingerprints. | Study quality inherits the limits of the selected run family. |
| Run sweeps | Study sweeps validate, plan, and execute deterministic sequences with failure entries. | Sweeps execute sequentially in `v1`. |
| Inspect results | Persisted run queries expose fields, domain payloads, diagnostics, summaries, quality reasons, and provenance. | Large payload handling and interpretation vary by domain. |
| Compare and trend | Results comparison and trend summaries are runtime operation surfaces. | Trend quality depends on consistent artifacts and comparable run setup. |
| Decide trust | Governance scripts cover benchmark schema, external references, readiness, thresholds, promotion calibration, and prep calibration. | Release status is per domain and fixture family. |

## Geometry Status

| Capability | Status | Records |
| --- | --- | --- |
| Format inspection | Works today | `geometry.inspect/v1` tests and importer coverage. |
| Geometry loading | Works today | `geometry.load/v1` tests for supported importer paths. |
| Stats and queries | Works today | Runtime geometry operation tests. |
| Region listing | Works today | Runtime geometry operation tests. |
| View capture | Works today with adapter fallback | Capture adapter boundary and fallback tests. |
| Analysis prep | Works today and evidence-backed | Prep artifacts, `MeshingPrepResult`, health checks, retention counters. |
| Prep-aware runs | Works today and governed | Run prep validation, stale/mismatch checks, prep calibration scripts. |

Geometry limits:

- CAD-native semantic depth varies by importer and source file.
- Region names and topology data may need host or preprocessing support for complex industrial models.
- Prep artifacts improve reproducibility and trust, but do not by themselves prove that a physics model is well posed.

## Physics Family Status

| Family | Runtime path today | Records | V&V maturity | Current boundary |
| --- | --- | --- | --- | --- |
| Linear static structural | Dedicated solver-backed path. | Runtime contracts, FEA linear-static pipeline, persisted results, quality gates, GPU residency fixtures. | L2, with limited known-answer checks. | Assumes linear static response; model quality still depends on material assignments, loads, and constraints. Needs broader analytic, patch, convergence, and external-reference coverage before production-grade claims. |
| Modal structural | Dedicated modal solver-backed path. | Modal payloads, residuals, orthogonality/separation diagnostics, runtime contracts, CPU/GPU fixture coverage. | L2. | Frequency quality can degrade publishability when residual, orthogonality, or separation gates warn. Known eigenfrequency references and convergence studies are still needed. |
| Acoustic harmonic | Baseline path backed by the modal solver. | `analysis.run_acoustic/v1`, acoustic fixtures, modal payloads, `FEA_ACOUSTIC_PLACEHOLDER` diagnostic, typed invalid-model errors. | L1 to L2 baseline. | Acoustic behavior is modal-response based; acoustic field propagation and acoustic-specific solver fidelity remain outside the current baseline. |
| Thermal standalone | Dedicated thermal pipeline over configured thermal context. | `analysis.run_thermal/v1`, thermal payloads, stability and constitutive diagnostics, thermal fixture coverage. | L1 to L2. | Runtime thermal solves require thermal step data plus thermo-mechanical thermal context; heat-equation known-answer cases and external thermal references remain needed. |
| Structural transient | Dedicated transient solver-backed path. | Transient payloads, residuals, stability/energy diagnostics, adaptive-step controls, CPU/GPU fixture coverage. | L2. | Time-step policy, model quality, and fixture class drive interpretation. Known-answer time integration and convergence checks remain needed. |
| Nonlinear structural | Dedicated nonlinear solver-backed path. | Nonlinear convergence, increment, line-search, plasticity/contact proxy, reference fixture, and quality-reason coverage. | L2. | Constitutive, contact, and workload depth expand by fixture family. Independent nonlinear, plasticity, and contact references remain needed. |
| Thermo-mechanical | Coupled path through thermal and structural solver contexts. | Thermo-mechanical context validation, thermal field artifacts, `FEA_TM_*` diagnostics, readiness signals. | L1 to L2. | Coupling is exposed through selected structural or thermal paths; coupled known-answer and external-reference checks remain needed. |
| Electro-thermal | Coupled path through electro-thermal solver context. | Electro-thermal context validation, Joule-coupling diagnostics, benign/pathological fixture governance. | L1 to L2. | Coupling is exposed through structural, transient, and nonlinear paths; standalone electro-thermal solve depth and coupled references remain outside the current surface. |
| Electromagnetic | Dedicated electromagnetic pipeline with field-proxy and sweep payloads. | Electromagnetic domain validation, run options, diagnostics, vector-potential/flux proxies, sweep and resonance metrics, governance thresholds. | L2 for proxy behavior. | Field quantities are proxy payloads; Maxwell field validation, boundary/source realization references, and larger workload hardening remain limited areas. |
| CFD | Baseline path backed by transient execution plus CFD domain diagnostics. | `analysis.run_cfd/v1`, CFD domain validation, transient payloads, `FEA_CFD_FLOW` diagnostics, fixture coverage. | L1 to L2 baseline. | Fluid behavior is represented by domain validation, flow diagnostics, and transient-style payloads; Navier-Stokes/fluid-field fidelity remains outside the current baseline. |
| CHT | Coupled baseline path combining CFD diagnostics with thermal and transient payloads. | `analysis.run_cht/v1`, CFD and thermal domain validation, `FEA_CFD_FLOW`, `FEA_CHT_COUPLING`, thermal/transient payloads. | L1 to L2 baseline. | Conjugate heat-transfer behavior is limited by the thermal and CFD baseline models; current payloads are thermal/transient rather than dedicated coupled fluid/thermal fields. |
| FSI | Coupled baseline path backed by transient execution plus CFD/FSI diagnostics. | `analysis.run_fsi/v1`, CFD domain validation, transient payloads, `FEA_CFD_FLOW`, `FEA_FSI_COUPLING`, fixture coverage. | L1 to L2 baseline. | FSI behavior is coupling metadata around the structural transient baseline; full two-way fluid-structure field solving remains outside the current baseline. |

## Operation Coverage

| Operation family | Coverage |
| --- | --- |
| `geometry.inspect`, `geometry.load`, `geometry.compute_stats`, `geometry.list_regions`, `geometry.query_entities`, `geometry.capture_view` | Contracted, runtime-backed, and tested. |
| `geometry.prep_for_analysis`, `geometry.prep_artifact_health` | Contracted, runtime-backed, tested, and governance-linked. |
| `analysis.create_model`, `analysis.validate` | Contracted, runtime-backed, and tested. |
| `analysis.run_linear_static`, `analysis.run_modal`, `analysis.run_thermal`, `analysis.run_transient`, `analysis.run_nonlinear`, `analysis.run_electromagnetic` | Contracted, runtime-backed, solver-backed, tested, and governed. |
| `analysis.run_acoustic`, `analysis.run_cfd`, `analysis.run_cht`, `analysis.run_fsi` | Contracted, runtime-backed, baseline solver-backed, tested, and governed. Broader domain fidelity remains limited. |
| `analysis.results`, `analysis.results_compare`, `analysis.trends` | Contracted, runtime-backed, tested, and governance-consumed. |
| `analysis.validate_study`, `analysis.plan_study`, `analysis.run_study` | Contracted, runtime-backed, tested, and artifact-backed. |
| `analysis.validate_study_sweep`, `analysis.plan_study_sweep`, `analysis.run_study_sweep` | Contracted, runtime-backed, tested, and artifact-backed. |
| RunMat builtins `geometry_inspect`, `geometry_load`, `analysis_validate_study`, `analysis_plan_study`, `analysis_run_study` | Registered, runtime-backed, and covered by focused builtin or CLI entrypoint tests. |

## Verification

Baseline Rust checks:

```sh
cargo fmt -- --check
cargo test --all-targets --all-features
```

Focused Rust checks:

```sh
cargo test -p runmat-analysis-core
cargo test -p runmat-analysis-fea
cargo test -p runmat-runtime --test operation_contracts
cargo test -p runmat-runtime --lib analysis
cargo test -p runmat-runtime --lib geometry
cargo test -p runmat-config
cargo test -p runmat --lib commands::script
```

Governance checks:

```sh
python3 -m unittest scripts.tests.test_validate_analysis_report_nonlinear
python3 -m unittest scripts.tests.test_validate_external_reference_benchmark
python3 -m unittest scripts.tests.test_external_reference_baseline
python3 -m unittest scripts.tests.test_generate_external_reference_benchmark
python3 -m unittest scripts.tests.test_release_readiness_nonlinear
python3 -m unittest scripts.tests.test_generate_threshold_ratchet_report
python3 -m unittest scripts.tests.test_validate_threshold_ratchet_report
python3 -m unittest scripts.tests.test_generate_promotion_threshold_calibration
python3 -m unittest scripts.tests.test_validate_promotion_threshold_calibration
python3 -m unittest scripts.tests.test_evaluate_prep_calibration_drift
python3 -m unittest scripts.tests.test_promote_prep_calibration_evidence
```

The merge baseline on June 9, 2026 was green for:

```sh
RUST_TEST_THREADS=1 cargo test --all-targets --all-features
```

## Update Rule

When support changes, update this page in the same change set as the implementation and tests. Status changes should say which workflow stage, domain, or operation changed, what records support the new claim, and whether the V&V maturity level changed.
