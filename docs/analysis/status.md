---
title: "Current Analysis Status"
category: "Analysis & Simulation"
section: "13.6"
last_updated: "June 9, 2026"
---

# Current Status

Current status is tracked across workflow stages and physics families. Each row states what works today, what evidence exists, and where support remains bounded.

## Status Terms

| Term | Meaning |
| --- | --- |
| Works today | The operation path exists, validates inputs, returns typed results or typed errors, and has test coverage. |
| Evidence-backed | The path persists or emits artifacts, diagnostics, provenance, quality reasons, or health data that reviewers can inspect. |
| Governed | Readiness, reference, threshold, calibration, or benchmark policy consumes evidence for this path. |
| Bounded | The path is usable within stated limits, but fidelity, format depth, scale, or release posture is not uniform across all possible inputs. |

## Workflow Status

| Stage | Status | Current boundary |
| --- | --- | --- |
| Drive from host code | Runtime operation functions are public for geometry, model creation, validation, run execution, result queries, studies, and sweeps. | Callers integrate through Rust/runtime APIs or host wrappers; the current CLI does not include a dedicated analysis or study command. |
| Drive from study specs | `AnalysisStudySpec` and `AnalysisStudySweepSpec` are serializable contracts with validate, plan, and run operations. | The runtime study spec embeds `GeometryAsset`; hosts that want path-based files need to load geometry first. |
| Drive from RunMat code | General file and JSON builtins exist, but analysis and geometry operation wrappers are not registered as RunMat-language builtins by the current runtime. | Direct `.m` access requires host-provided wrappers or builtins over the runtime operations. |
| Bring geometry in | Geometry inspect and load operations cover STL, STEP, OBJ, PLY, and glTF importer paths. | CAD semantic richness varies by format, importer, and source file. |
| Inspect geometry | Stats, region listing, bounded entity queries, and view capture are available. | Region fidelity and entity meaning depend on source geometry and importer detail. |
| Prepare geometry | Deterministic prep profiles produce analysis prep artifacts and health data. | Meshing, adaptive refinement, and topology depth continue to expand by profile. |
| Build models | Geometry can seed solver-agnostic analysis models through profile intents. | Generated models are scaffolds; meaningful runs still need intentional materials, loads, constraints, and domains. |
| Validate models | Model validation and operation-specific validation return typed issues and errors. | Validation covers schema and compatibility. Engineering fit still depends on the chosen model data and assumptions. |
| Run solves | Direct run operations exist for structural, modal, thermal, nonlinear, electromagnetic, acoustic, CFD, CHT, and FSI families. | Domain fidelity varies. Some families are deeper governed paths; others are governed baseline paths. |
| Run studies | Study validate, plan, and run operations persist evidence and deterministic fingerprints. | Study quality inherits the limits of the selected run family. |
| Run sweeps | Study sweeps validate, plan, and execute deterministic sequences with failure entries. | Sweeps execute sequentially in `v1`. |
| Inspect results | Persisted run queries expose fields, domain payloads, diagnostics, summaries, quality reasons, and provenance. | Large payload handling and interpretation vary by domain. |
| Compare and trend | Results comparison and trend summaries are runtime operation surfaces. | Trend quality depends on consistent artifacts and comparable run setup. |
| Decide trust | Governance scripts cover benchmark schema, external references, readiness, thresholds, promotion calibration, and prep calibration. | Release-ready status is per domain and fixture family. |

## Geometry Status

| Capability | Status | Evidence |
| --- | --- | --- |
| Format inspection | Works today | `geometry.inspect/v1` tests and importer coverage. |
| Geometry loading | Works today | `geometry.load/v1` tests for supported importer paths. |
| Stats and queries | Works today | Runtime geometry operation tests. |
| Region listing | Works today | Runtime geometry operation tests. |
| View capture | Works today with adapter fallback | Capture adapter boundary and fallback tests. |
| Analysis prep | Works today and evidence-backed | Prep artifacts, `MeshingPrepResult`, health checks, retention counters. |
| Prep-aware runs | Works today and governed | Run prep validation, stale/mismatch checks, prep calibration scripts. |

Geometry boundaries:

- CAD-native semantic depth varies by importer and source file.
- Region names and topology evidence may need host or preprocessing support for complex industrial models.
- Prep artifacts improve reproducibility and trust, but do not by themselves prove that a physics model is well posed.

## Physics Family Status

| Family | Status | Evidence | Current boundary |
| --- | --- | --- | --- |
| Linear static structural | Works today, evidence-backed, governed | Runtime tests, FEA path, persisted results, quality gates. | Baseline structural path; model quality depends on attachments and constraints. |
| Modal structural | Works today, evidence-backed, governed | Modal payloads, diagnostics, runtime tests. | Frequency quality depends on model setup and fixture coverage. |
| Acoustic harmonic | Works today as a governed baseline path | Acoustic contract, diagnostics, comparator and readiness coverage. | Broader acoustic solver fidelity is still a depth area. |
| Thermal standalone | Works today, evidence-backed, governed | Thermal payloads, diagnostics, readiness signals. | Release posture depends on fixture family and policy. |
| Structural transient | Works today, evidence-backed, governed | Transient payloads, residual and stability summaries. | Time-profile and model quality remain important for interpretation. |
| Nonlinear structural | Works today, evidence-backed, governed | Nonlinear convergence, plastic/contact, and coupled-load quality reasons. | Constitutive depth and workload scale are governed incrementally. |
| Thermo-mechanical | Works today in coupled structural paths | Coupling options, field artifacts, readiness signals. | Coupling depth depends on fixture family and promoted field evidence. |
| Electro-thermal | Works today in coupled paths | Joule coupling, benign/pathological fixture governance. | Coupling depth continues through governed fixtures. |
| Electromagnetic | Works today, evidence-backed, governed | Electromagnetic domain, run options, diagnostics, sweep/resonance fields, readiness signals. | Deeper field-solver fidelity and larger workload hardening remain bounded areas. |
| CFD | Works today as a governed baseline path | CFD contract, transient-style execution, CFD diagnostics. | Broader fluid fidelity remains a depth area. |
| CHT | Works today as a governed baseline path | Coupled CFD/thermal operation and governance. | Broader coupled-flow fidelity remains a depth area. |
| FSI | Works today as a governed baseline path | Coupled structural/CFD operation and governance. | Broader FSI fidelity remains a depth area. |

## Operation Coverage

| Operation family | Coverage |
| --- | --- |
| `geometry.inspect`, `geometry.load`, `geometry.compute_stats`, `geometry.list_regions`, `geometry.query_entities`, `geometry.capture_view` | Contracted, runtime-backed, and tested. |
| `geometry.prep_for_analysis`, `geometry.prep_artifact_health` | Contracted, runtime-backed, tested, and governance-linked. |
| `analysis.create_model`, `analysis.validate` | Contracted, runtime-backed, and tested. |
| `analysis.run_linear_static`, `analysis.run_modal`, `analysis.run_thermal`, `analysis.run_transient`, `analysis.run_nonlinear`, `analysis.run_electromagnetic` | Contracted, runtime-backed, solver-backed, tested, and governed. |
| `analysis.run_acoustic`, `analysis.run_cfd`, `analysis.run_cht`, `analysis.run_fsi` | Contracted, runtime-backed, baseline solver-backed, tested, and governed. Broader domain fidelity remains bounded. |
| `analysis.results`, `analysis.results_compare`, `analysis.trends` | Contracted, runtime-backed, tested, and governance-consumed. |
| `analysis.validate_study`, `analysis.plan_study`, `analysis.run_study` | Contracted, runtime-backed, tested, and evidence-backed. |
| `analysis.validate_study_sweep`, `analysis.plan_study_sweep`, `analysis.run_study_sweep` | Contracted, runtime-backed, tested, and evidence-backed. |

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
cargo test -p runmat-runtime --test analysis
cargo test -p runmat-runtime --test operation_contracts
cargo test -p runmat-runtime --lib analysis
cargo test -p runmat-runtime --lib geometry
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

When support changes, update this page in the same change set as the implementation and tests. Status changes should say which workflow stage, domain, or operation changed, and what evidence supports the new claim.
