# Analysis System Roadmap

Last updated: 2026-05-17

## Planning Principles

1. Preserve deterministic behavior and typed contract stability.
2. Ship thin vertical slices with benchmarks and governance gates.
3. Avoid broad platform churn without immediate physics/governance value.
4. No domain is considered complete without benchmark + trend gates.

## Ordered Execution Plan

### Phase A: Finish Priority Domain Fidelity

Scope:

- Thermal/thermo, electro-thermal, plastic/contact realism deepening.
- EM constitutive fidelity continuation.

Exit criteria:

- Domain-specific benchmark envelopes tightened and stable.
- Protected branch readiness remains green without ad-hoc waivers.

### Phase B: EM Completion to Production-Credible Baseline

Scope:

- Frequency-dependent EM constitutive behavior.
- Stronger Maxwell-form solve fidelity and reference validation.
- Extended EM external-reference checks in governance.

Exit criteria:

- EM passes parity-credible domain criteria in `GOAL.md`.
- EM reference suite and trend gates are stable on protected branches.

### Phase C: First Missing Major Family (CFD Core)

Scope:

- Introduce CFD schema/contracts and first steady/transient fluid path.
- Add fluid-specific diagnostics and benchmark suite.

Exit criteria:

- CFD baseline operational under existing contract/governance discipline.

### Phase D: Coupled Family Expansion

Scope:

- CHT first, then FSI.
- Reuse established contract/governance mechanisms.

Exit criteria:

- At least one new coupled family has stable readiness gates.

## Near-Term Slices (Next)

1. EM frequency-dependent material coefficients (`sigma(omega)`, optional dispersive terms).
2. EM stronger external-reference comparator metrics.
3. Thermal/thermo benchmark tightening where drift remains permissive.
4. Plastic/contact constitutive realism increment with benchmark lock-in.

Progress update (2026-05-17):

1. [x] EM frequency-dependent material coefficients baseline landed:
   additive `conductivity_frequency_response` schema, FEA `sigma(omega)` interpolation with dispersive loss coupling terms, and EM fixture governance thresholds.
2. [x] EM stronger external-reference comparator metrics baseline landed:
   external-reference benchmark generation now supports threshold-assertion-backed metrics, with EM `sigma(omega)` comparator entries wired into the M6 baseline.
3. [x] Thermal/thermo benchmark tightening landed:
   tightened thermo gradient/ramp/shock and standalone thermal threshold assertions to reduce permissive drift windows; validated with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib`.
4. [x] Plastic/contact constitutive realism increment with benchmark lock-in landed:
   tightened nonlinear plastic/contact proxy and reference threshold assertions (severity and load realization/amplification bands) to lock benchmark behavior; validated with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
5. [x] Electro-thermal benchmark tightening landed:
   tightened benign/pathological electro-thermal and nonlinear mixed-load electro-thermal threshold assertions (Joule coupling, conductivity spread, severity, temporal variation, and time-scale bands); validated with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
6. [x] Nonlinear convergence benchmark tightening landed:
   tightened nonlinear assembly/stress/softening convergence threshold assertions (line-search backtracks, increment/residual norms, spike/stall counts, and tangent rebuild bands) to reduce permissive drift windows; validated with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
7. [x] CFD contract kickoff landed:
   additive core CFD domain schema (`AnalysisStepKind::Cfd`, model-owned CFD domain fields) plus `analysis.create_model` steady/transient CFD profile templates with runtime coverage tests; validated with `cargo test -p runmat-analysis-core`, `cargo test -p runmat-runtime --lib -- --test-threads=1`, and `cargo test -p runmat-runtime --test analysis`.
8. [x] CFD first run-path baseline landed:
   added additive runtime operation contract `analysis.run_cfd/v1` with `AnalysisCfdRunOptions`, model/domain + option validation, prep-context integration, typed run envelope output, and CFD flow diagnostics (`FEA_CFD_FLOW`) backed by runtime coverage tests.
9. [x] CFD conformance/trend baseline landed:
   added benchmark harness CFD fixtures (cpu/gpu-provider/gpu-fallback) with `FEA_CFD_FLOW` threshold assertions and promoted runtime trends to classify CFD runs as a first-class run kind (`AnalysisRunKind::Cfd`).
10. [x] CHT contract kickoff landed:
    added additive `analysis.create_model` profile `cht_coupled` with baseline coupled template defaults (CFD + thermal steps, seeded CFD domain, and thermo-mechanical coupling domain) plus runtime contract tests.
11. [x] CHT first run-path baseline landed:
    added additive runtime operation contract `analysis.run_cht/v1` with `AnalysisChtRunOptions`, coupled CFD+thermal model/option validation, typed envelope output including both transient+thermal payloads, and first CHT diagnostics (`FEA_CHT_COUPLING` + `FEA_CFD_FLOW`) backed by runtime tests.
12. [x] CHT conformance/trend baseline landed:
    added benchmark harness CHT fixtures (cpu/gpu-provider/gpu-fallback) with coupled threshold assertions (`FEA_CHT_COUPLING` + `FEA_CFD_FLOW`) and promoted runtime trends to classify CHT runs as a first-class run kind (`AnalysisRunKind::Cht`).
13. [x] FSI contract kickoff landed:
    added additive `analysis.create_model` profile `fsi_coupled` with baseline coupled template defaults (transient structural + CFD steps, seeded transient CFD domain, and baseline FSI load template) plus runtime contract tests.
14. [x] Study workflow kickoff landed:
    added additive high-level runtime operations `analysis.validate_study/v1`, `analysis.plan_study/v1`, and `analysis.run_study/v1` with typed study contracts, deterministic study fingerprinting, canonical linear-static operation sequencing, and runtime coverage tests.
15. [x] FSI first run-path and governance baseline landed:
    added additive runtime operation contract `analysis.run_fsi/v1` with `AnalysisFsiRunOptions`, transient+CFD model/domain validation, FSI diagnostics (`FEA_FSI_COUPLING` + `FEA_CFD_FLOW`), study/trend classification wiring (`AnalysisRunKind::Fsi`), and conformance harness fixtures (`fsi_coupled_cpu`, `fsi_coupled_gpu_provider`, `fsi_coupled_gpu_fallback`) with threshold assertions.
16. [x] Study workflow canonical evidence baseline landed:
    `analysis.validate_study/v1`, `analysis.plan_study/v1`, and `analysis.run_study/v1` now persist canonical study evidence artifacts (`validate.json`, `plan.json`, `run.json`) keyed by deterministic study fingerprint under a configurable study artifact root (`RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT`), with typed payload fields exposing evidence artifact paths.
17. [x] Coupled-family artifact op-version classification hardened:
    filesystem analysis artifacts now classify and persist family-accurate `op_version` for CFD/CHT/FSI runs (`analysis.run_cfd/v1`, `analysis.run_cht/v1`, `analysis.run_fsi/v1`) based on diagnostic signatures, with regression coverage to prevent coupled runs from collapsing into generic transient op-version labels.
18. [x] Coupled-family governance validator coverage expanded:
    release/nonlinear benchmark schema validation now requires CFD/CHT/FSI provider fixture threshold-assertion sets (`cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, `fsi_coupled_gpu_provider`) so coupled-family conformance regressions fail policy validation instead of silently passing.
19. [x] Coupled-family external-reference comparator metrics expanded:
    M6 baseline external-reference data now includes threshold-assertion metrics for CFD/CHT/FSI provider fixtures (`cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, `fsi_coupled_gpu_provider`) so coupled-family regressions are evaluated by governance reference comparisons.
20. [x] External-reference artifact validator coverage hardened for coupled families:
    `validate_external_reference_benchmark.py` now requires coupled CFD/CHT/FSI metric sets to be present in generated comparator artifacts so incomplete payloads fail governance validation in enforce mode.
21. [x] External-reference baseline guard test promoted into CI governance suite:
    release-readiness governance unit tests now include `scripts.tests.test_external_reference_baseline` so coupled-family comparator baseline coverage regressions fail CI before artifact generation.
22. [x] Acoustics contract kickoff landed:
    added additive `analysis.create_model` profile `acoustic_harmonic` that provisions a harmonic-template model scaffold (modal-step placeholder) and corresponding runtime contract/profile tests as the first acoustics family entry point.
23. [x] EM external-reference comparator coverage expanded and enforced:
    M6 baseline now includes additional EM threshold-assertion comparators (homogeneous/heterogeneous dispersion and field-quality proxies plus boundary/phased-source metrics), and `validate_external_reference_benchmark.py` now requires core EM fixture metric sets so incomplete EM comparator payloads fail governance validation.
24. [x] Priority-domain external-reference coverage expanded beyond EM:
    external-reference baseline and validator policy now require representative thermo/electro/plastic/contact fixture comparator metrics (`thermo_gradient_pathological_gpu_provider`, `thermo_shock_oscillatory_gpu_provider`, `electro_thermal_joule_pathological_gpu_provider`, `nonlinear_plastic_hardening_reference_complex_gpu_provider`, `nonlinear_contact_frictionless_reference_complex_gpu_provider`) so gaps in non-EM priority-domain comparator payloads fail governance validation.
25. [x] Key-workload performance telemetry enforcement expanded in benchmark-schema validation:
    `validate_analysis_report_nonlinear.py` now requires finite GPU performance fields (`gpu_speedup_ratio`, `gpu_solver_solve_ms`, `gpu_solver_backend`) for representative nonlinear/thermo/electro/CFD/CHT/FSI provider fixtures so release benchmarking cannot omit scale posture signals for key workloads.
26. [x] Acoustics first runtime run-path baseline landed:
    added additive `analysis.run_acoustic/v1` + `AnalysisAcousticRunOptions` (modal-backed harmonic placeholder path), with dedicated acoustic diagnostics (`FEA_ACOUSTIC_PLACEHOLDER`), trends classification (`AnalysisRunKind::Acoustic`), study-run routing, and artifact `op_version` persistence (`analysis.run_acoustic/v1`) under existing contract/evidence discipline.
27. [x] Acoustics conformance/governance baseline landed:
    added acoustic conformance fixtures (`acoustic_harmonic_cpu`, `acoustic_harmonic_gpu_provider`, `acoustic_harmonic_gpu_fallback`) with enforced acoustic threshold assertions, extended nonlinear benchmark-schema governance to require acoustic provider assertion + GPU performance telemetry fields, and extended M6 external-reference baseline plus validator-required fixture metrics to include acoustic comparator coverage.
28. [x] EM external-reference validator hardened to expanded metric-set enforcement:
    external-reference artifact validation now requires the full expanded EM comparator sets for homogeneous/heterogeneous fixtures (including response-coverage, boundary-energy, and source-realization metrics) plus dedicated boundary-penalty/phased-source fixture metrics, closing partial-payload gaps in enforce-mode governance.
29. [x] Key-workload benchmark-schema performance enforcement expanded to EM fixtures:
    `validate_analysis_report_nonlinear.py` now requires EM provider fixture threshold assertion sets (homogeneous/heterogeneous/boundary-penalty/phased-source) and finite GPU performance telemetry fields for those fixtures, with conformance-harness fallback telemetry wiring so `gpu_solver_solve_ms` remains populated when EM-specific cost diagnostics are absent.
30. [x] Release-readiness performance trend gating expanded beyond nonlinear-only fixtures:
    `release_readiness_nonlinear.py` now evaluates key EM/acoustic/CFD/CHT/FSI provider fixtures for minimum GPU speedup and rolling slowdown ratios (with governance-profile defaults and enforceable fixture-presence policy), adding regression blocking posture for non-nonlinear key workloads.
31. [x] Acoustic comparator/assertion governance depth expanded:
    acoustic conformance + external-reference governance now enforce acoustic-specific modal quality metrics (`acoustic_max_m_orthogonality_offdiag`, `acoustic_min_relative_frequency_separation`) for provider fixtures in addition to mode-count/residual controls.
32. [x] EM external-reference fixture coverage expanded beyond core homogeneous/heterogeneous paths:
    added/enforced comparator metrics for sparse-assignments, fallback-heavy, overlap-interference, and boundary-kernel EM provider fixtures, and expanded required metric sets for boundary-penalty/phased-source fixtures in enforce-mode validation.
33. [x] EM native cost telemetry surfaced for governance evidence:
    electromagnetic FEA runs now emit `FEA_EM_COST` diagnostics (`prepared_build_ms`, `solve_ms`, `fallback_apply_count`), and benchmark conformance extraction now prefers these EM-native metrics when populating solver-cost telemetry fields.
34. [x] Release-readiness Maxwell posture gating expanded:
    `release_readiness_nonlinear.py` now enforces profile-tuned EM readiness thresholds and trend checks (energy imbalance, flux divergence, residual norms, source realization/coverage/alignment, and EM breach-rate/trend-ratio limits), with regression coverage in `scripts.tests.test_release_readiness_nonlinear`.
35. [x] EM harmonic solver controls made additive and policy-visible:
    `analysis.run_electromagnetic/v1` now accepts validated harmonic control options (`residual_target`, `harmonic_tolerance`, `harmonic_max_iterations`) that flow into FEA solve behavior and EM diagnostics, replacing fixed harmonic constants with explicit runtime contract controls.
36. [x] EM benchmark-schema fixture coverage expanded for non-core EM scenarios:
    `validate_analysis_report_nonlinear.py` now requires sparse-assignments, fallback-heavy, overlap-interference, and boundary-kernel EM provider fixture assertion sets plus finite GPU performance telemetry fields, broadening Maxwell schema-governance coverage beyond the original core EM fixtures.
