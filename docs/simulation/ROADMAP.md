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
37. [x] Study workflow EM option propagation and validation landed:
    additive `AnalysisStudySpec.electromagnetic_run_options` is now validated by `analysis.validate_study/v1`, rejected when used on non-EM run kinds, and routed by `analysis.run_study/v1` into `analysis.run_electromagnetic/v1`; electromagnetic create-model profile defaults now seed an EM domain so study-driven EM runs are executable under canonical study workflows.
38. [x] Study workflow typed outputs now surface EM execution options:
    additive `analysis.plan_study/v1` and `analysis.run_study/v1` typed payloads now include `electromagnetic_run_options`, with run-study reporting resolved EM defaults when options are omitted, improving programmatic reproducibility without requiring artifact JSON inspection.
39. [x] Study workflow result-extraction ergonomics expanded:
    additive `analysis.run_study/v1` typed output now includes concrete run operation identity (`run_operation`, `run_op_version`) plus propagated `quality_reasons`, so programmatic callers can consume execution verdict context without an immediate follow-up `analysis.results` call.
40. [x] Study planning typed ergonomics expanded:
    additive `analysis.plan_study/v1` typed output now includes planned run operation identity (`run_operation`, `run_op_version`) so automation clients can select downstream run-contract handlers without parsing `operation_sequence`.
41. [x] Study validation typed issue details added:
    additive `analysis.validate_study/v1` payload now includes structured `issues` entries (`code`, `message`) alongside existing `issue_codes`, preserving machine-stable code interfaces while improving automation/debug ergonomics without out-of-band code-to-message mapping tables.
42. [x] Study run typed extraction expanded with solver gates and provenance:
    additive `analysis.run_study/v1` typed payload now exposes `solver_convergence`, `result_quality`, and full `provenance`, allowing automation to consume gate posture and backend/solver lineage without immediate secondary results queries.
43. [x] Study sweep orchestration baseline landed:
    added additive high-level operation `analysis.run_study_sweep/v1` with typed sweep contracts (`AnalysisStudySweepSpec`/`AnalysisStudySweepData`) for deterministic sequential multi-study execution, aggregate run-entry summaries, and persisted sweep evidence artifacts under the study artifact root.
44. [x] Study sweep failure-policy ergonomics expanded:
    additive `analysis.run_study_sweep/v1` now supports policy-controlled failure handling (`fail_fast`) plus typed per-study failure entries (`failed_count`, `failure_entries`) so automation pipelines can choose between fail-fast semantics and partial-progress sweep completion.
45. [x] Study sweep typed validation operation landed:
    added additive `analysis.validate_study_sweep/v1` with typed sweep-level and per-study validation outputs (`AnalysisStudySweepValidateData`, `AnalysisStudySweepValidateEntry`) including structured study issues and persisted sweep validation evidence artifacts for automation-first preflight workflows.
46. [x] Study sweep typed planning operation landed:
    added additive `analysis.plan_study_sweep/v1` with typed per-study plan outputs (`AnalysisStudySweepPlanData`, `AnalysisStudySweepPlanEntry`), planned/failure aggregate counts, continue-on-failure semantics via `fail_fast`, and persisted sweep planning evidence artifacts for automation-ready multi-study preflight planning.
47. [x] EM dispersive phase-attenuation fidelity/governance increment landed:
    electromagnetic FEA now applies dispersive-loss-derived phase attenuation to conductive coupling terms and surfaces phase attenuation diagnostics (`dispersive_phase_attenuation_mean`, `dispersive_phase_conductivity_attenuation_ratio`), with conformance threshold assertions and external-reference/nonlinear schema governance enforcement extended for homogeneous/heterogeneous EM provider fixtures.
48. [x] EM release-readiness phase-attenuation gating increment landed:
    `release_readiness_nonlinear.py` now evaluates EM homogeneous/heterogeneous phase attenuation threshold assertions (`em_*_dispersive_phase_attenuation_mean`, `em_*_dispersive_phase_conductivity_attenuation_ratio`) with profile-tuned minimum thresholds and breach-rate integration so phase-fidelity regressions become branch readiness reasons instead of passive telemetry.
49. [x] EM phase attenuation external/schema enforcement depth expanded:
    external-reference and nonlinear benchmark-schema validators now require both phase attenuation mean and phase conductivity attenuation ratio metrics for homogeneous/heterogeneous EM provider fixtures, with M6 baseline and governance tests updated so missing phase-ratio evidence fails enforcement.
50. [x] EM release-readiness phase-attenuation trend gating expanded:
    `release_readiness_nonlinear.py` now evaluates rolling trend ratios for homogeneous/heterogeneous EM phase attenuation assertions (mean and conductivity attenuation ratio) under profile-tuned thresholds, emitting dedicated readiness reasons when phase-fidelity posture regresses relative to baseline.
51. [x] EM external-reference source-alignment comparator enforcement expanded:
    M6 baseline and validator-required metric sets now include homogeneous/heterogeneous EM source-material-alignment assertion comparators, with nonlinear schema governance requiring the corresponding threshold assertions so source-fidelity evidence gaps fail benchmark/external-reference enforcement.
52. [x] EM external-reference dispersive-coupling comparator enforcement expanded:
    M6 baseline and validator-required metric sets now include homogeneous/heterogeneous EM dispersive-coupling-ratio assertion comparators, with nonlinear schema governance requiring the corresponding threshold assertions so dispersive-coupling evidence gaps fail benchmark/external-reference enforcement.
53. [x] EM release-readiness dispersive-coupling gating expanded:
    `release_readiness_nonlinear.py` now evaluates homogeneous/heterogeneous EM dispersive-coupling-ratio threshold assertions with profile-tuned maximum thresholds, promoting excessive coupling-ratio excursions into explicit readiness reasons and breach-rate accounting.
54. [x] Key-performance release-readiness coverage expanded to additional priority domains:
    `release_readiness_nonlinear.py` now treats representative thermo/electro/plastic/contact fixtures as key-performance workloads (speedup floor + slowdown trend checks), extending scale-regression gating beyond EM/acoustic/CFD/CHT/FSI.
55. [x] EM external-reference source-coverage/anchor comparator enforcement expanded:
    M6 baseline and validator-required metric sets now include homogeneous/heterogeneous EM source-region-coverage and boundary-anchor assertion comparators, with nonlinear schema governance requiring the corresponding threshold assertions so source/boundary-fidelity evidence gaps fail benchmark/external-reference enforcement.
56. [x] EM release-readiness boundary-anchor gating expanded:
    `release_readiness_nonlinear.py` now evaluates homogeneous/heterogeneous EM boundary-anchor-ratio threshold assertions with profile-tuned minimum thresholds, promoting anchor-ratio regressions into explicit readiness reasons and EM breach-rate accounting.
57. [x] EM non-core fixture external-reference/source-boundary comparator coverage expanded:
    M6 baseline and validator-required metric sets now enforce richer sparse/fallback/overlap/boundary-kernel EM assertion comparators (assignment/source-coverage/material-alignment/boundary-anchor and boundary-kernel anchor/leakage signals), with nonlinear schema governance requiring corresponding assertions so non-core EM source/boundary-fidelity evidence gaps fail enforcement.
58. [x] EM boundary-penalty/phased-source comparator governance depth expanded:
    M6 baseline plus external-reference and nonlinear schema validators now require boundary-penalty residual norm comparators (`em_boundary_penalty_real_residual_norm`, `em_boundary_penalty_imag_residual_norm`) and phased-source overlap/interference comparators (`em_phased_source_overlap_ratio`, `em_phased_source_interference_index`) so partial boundary/phased EM fidelity payloads fail enforce-mode policy.
59. [x] EM release-readiness boundary/phased posture gating expanded:
    `release_readiness_nonlinear.py` now evaluates boundary-penalty residual norm assertions and phased-source overlap/interference assertions with profile-tuned thresholds, promoting boundary/phased EM fidelity regressions into explicit readiness reasons and EM breach-rate accounting.
60. [x] EM release-readiness boundary/phased trend gating expanded:
    `release_readiness_nonlinear.py` now evaluates rolling worsening ratios for boundary-penalty residual norm and phased-source overlap/interference assertions, emitting dedicated readiness reasons when boundary/phased EM fidelity posture regresses relative to baseline.
61. [x] EM release-readiness non-core fixture posture gating expanded:
    `release_readiness_nonlinear.py` now evaluates sparse/fallback/overlap/boundary-kernel assertion thresholds (assignment/fallback/source coverage/material alignment/boundary anchor/interference/localization/leakage signals), promoting non-core EM fixture regressions into explicit readiness reasons and EM breach-rate accounting.
62. [x] EM release-readiness non-core fixture trend gating expanded:
    `release_readiness_nonlinear.py` now evaluates rolling worsening ratios for sparse/fallback/overlap/boundary-kernel assertion metrics, emitting dedicated readiness reasons when non-core EM fixture posture regresses relative to baseline.
63. [x] EM release-readiness summary visibility expanded:
    `markdown_summary` output from `release_readiness_nonlinear.py` now includes an explicit EM posture section (core + boundary/phased + non-core static/trend signals), improving operator-facing readiness triage without requiring JSON field inspection.
64. [x] EM sparse/fallback source-realization and energy-imbalance comparator governance expanded:
    M6 baseline plus external-reference/nonlinear schema validators now require sparse/fallback source-realization and energy-imbalance assertion comparators (`em_sparse_source_realization_ratio`, `em_sparse_energy_imbalance_ratio`, `em_fallback_heavy_source_realization_ratio`, `em_fallback_heavy_energy_imbalance_ratio`) so partial non-core EM comparator payloads fail enforce-mode governance.
65. [x] EM release-readiness sparse/fallback source-realization and energy-imbalance gating expanded:
    `release_readiness_nonlinear.py` now evaluates sparse/fallback source-realization and energy-imbalance threshold assertions and rolling trend ratios (`em_sparse_source_realization_ratio`, `em_sparse_energy_imbalance_ratio`, `em_fallback_heavy_source_realization_ratio`, `em_fallback_heavy_energy_imbalance_ratio`) with profile-tuned thresholds, promoting non-core source/imbalance regressions into explicit readiness reasons and EM breach-rate posture.
66. [x] EM release-readiness overlap-source overlap gating expanded:
    `release_readiness_nonlinear.py` now evaluates overlap-interference fixture overlap-ratio threshold assertions and rolling drop trends (`em_overlap_source_overlap_ratio`) with profile-tuned thresholds, promoting non-core overlap-fidelity regressions into explicit readiness reasons alongside existing overlap coverage/material/interference posture signals.
67. [x] EM release-readiness boundary/phased anchor-coverage gating expanded:
    `release_readiness_nonlinear.py` now evaluates boundary-penalty anchor-ratio and phased-source region-coverage threshold assertions and rolling drop trends (`em_boundary_penalty_anchor_ratio`, `em_phased_source_region_coverage_ratio`) with profile-tuned thresholds, promoting boundary/phased source-coverage regressions into explicit readiness reasons.
68. [x] EM release-readiness boundary/phased conditioning-energy consistency gating expanded:
    `release_readiness_nonlinear.py` now evaluates boundary-penalty conditioning-contribution and phased-source energy-consistency threshold assertions and rolling worsening trends (`em_boundary_penalty_conditioning_contribution`, `em_phased_source_energy_consistency_ratio`) with profile-tuned thresholds, promoting additional boundary/phased source-fidelity regressions into explicit readiness reasons.
69. [x] EM release-readiness core sigma/loss/contrast assertion gating expanded:
    `release_readiness_nonlinear.py` now evaluates homogeneous/heterogeneous sigma-response, dispersive-loss, boundary-energy, and region-contrast threshold assertions and rolling worsening trends (`em_homogeneous_sigma_omega_scale_mean`, `em_homogeneous_sigma_omega_response_coverage_ratio`, `em_heterogeneous_sigma_omega_scale_spread_ratio`, `em_homogeneous_dispersive_loss_scale_mean`, `em_heterogeneous_dispersive_loss_scale_mean`, `em_homogeneous_boundary_energy_ratio`, `em_heterogeneous_region_contrast_index`), and adds fallback-heavy source-material-alignment threshold/trend gating (`em_fallback_heavy_source_material_alignment_ratio`) so additional core/non-core EM source-fidelity regressions become explicit readiness reasons.
70. [x] EM release-readiness homogeneous flux-assertion fallback gating hardened:
    `release_readiness_nonlinear.py` now falls back to homogeneous threshold assertion evidence (`em_homogeneous_flux_divergence_proxy`) when field-level flux-divergence telemetry is absent, for both static posture and rolling trend checks, so EM flux readiness reasons remain enforceable with assertion-backed benchmark payloads.
71. [x] EM release-readiness core anchor/coupling/source trend gating expanded:
    `release_readiness_nonlinear.py` now evaluates rolling worsening trends for core homogeneous/heterogeneous boundary-anchor, dispersive-coupling, and source-fidelity assertions (`em_homogeneous_boundary_anchor_ratio`, `em_heterogeneous_boundary_anchor_ratio`, `em_homogeneous_dispersive_coupling_ratio`, `em_heterogeneous_dispersive_coupling_ratio`, `em_heterogeneous_source_realization_ratio`, `em_homogeneous_source_region_coverage_ratio`, `em_heterogeneous_source_region_coverage_ratio`, `em_homogeneous_source_material_alignment_ratio`, `em_heterogeneous_source_material_alignment_ratio`) with profile-tuned thresholds, promoting additional baseline-relative core EM drift into explicit readiness reasons.
72. [x] Acoustic release-readiness assertion posture/trend gating landed:
    `release_readiness_nonlinear.py` now evaluates acoustic harmonic provider assertion thresholds and rolling trends (`acoustic_max_m_orthogonality_offdiag`, `acoustic_min_relative_frequency_separation`, `acoustic_mode_count`, `acoustic_residual_warn_threshold`) with profile-tuned policy defaults, surfacing acoustic quality regressions as explicit readiness reasons and adding operator summary visibility.
73. [x] Coupled-flow release-readiness assertion posture/trend gating landed:
    `release_readiness_nonlinear.py` now evaluates CFD/CHT/FSI provider assertion thresholds and rolling trends for core coupled-flow quality signals (`cfd_reynolds_proxy`, `cht_reynolds_proxy`, `cht_applied_temperature_delta_k`, `fsi_reynolds_proxy`, `fsi_structural_step_count`) with profile-tuned policy defaults, surfacing coupled-flow regressions as explicit readiness reasons and adding operator summary visibility.
74. [x] Nonlinear core release-readiness assertion posture gating landed:
    `release_readiness_nonlinear.py` now evaluates nonlinear assembly/stress/softening core assertion thresholds (`nonlinear_total_increments`, `nonlinear_failed_increments`, `nonlinear_iteration_spike_count`, `nonlinear_stress_total_increments`, `nonlinear_stress_stall_count`, `nonlinear_stress_iteration_spike_count`, `nonlinear_softening_total_increments`, `nonlinear_softening_spike_count`, `nonlinear_softening_backtrack_bursts`) with profile-tuned policy defaults, surfacing nonlinear core convergence regressions as explicit readiness reasons and adding operator summary visibility.
75. [x] Thermo/electro pathological release-readiness assertion trend gating landed:
    `release_readiness_nonlinear.py` now evaluates pathological thermo/electro assertion thresholds and rolling worsening trends (`thermo_gradient_pathological_spread_ratio`, `thermo_gradient_pathological_temporal_variation`, `thermo_shock_oscillatory_temporal_variation`, `electro_thermal_pathological_conductivity_spread_ratio`, `electro_thermal_pathological_temporal_variation`) with profile-tuned policy defaults, surfacing priority-domain pathological drift as explicit readiness reasons and adding operator summary visibility.
76. [x] Plastic/contact reference-complex release-readiness assertion trend gating landed:
    `release_readiness_nonlinear.py` now evaluates complex reference-fixture plastic/contact assertion posture and rolling worsening trends (`plasticity_hardening_reference_complex_load_realization_ratio`, `contact_frictionless_complex_load_amplification_ratio`) with profile-tuned policy defaults, surfacing constitutive load-path drift as explicit readiness reasons and adding operator summary visibility.
77. [x] Thermo/electro pathological external-reference comparator governance depth expanded:
    M6 baseline and `validate_external_reference_benchmark.py` required metric sets now include pathological temporal-variation assertion comparators (`thermo_gradient_pathological_temporal_variation`, `electro_thermal_pathological_temporal_variation`) for thermo-gradient and electro-thermal pathological provider fixtures, closing a fixture-depth mismatch between external-reference comparator governance and nonlinear benchmark-schema assertion requirements.
78. [x] EM core release-readiness assertion-backed fallback coverage expanded:
    `release_readiness_nonlinear.py` now falls back from missing core EM field telemetry to homogeneous/heterogeneous threshold assertions for energy-imbalance, flux-divergence, and source-fidelity posture signals (realization, region coverage, material alignment), and now also applies assertion-backed energy/flux trend fallback so EM drift governance remains enforceable on assertion-centric benchmark payloads.
79. [x] EM core external/schema comparator governance depth expanded:
    nonlinear benchmark-schema and external-reference validator-required metric sets now include additional core EM assertion signals consumed by readiness fallback (`em_homogeneous_source_realization_ratio`, `em_homogeneous_energy_imbalance_ratio`, `em_heterogeneous_flux_divergence_proxy`, `em_heterogeneous_energy_imbalance_ratio`), with M6 baseline comparator entries added so missing core EM fallback evidence fails enforce-mode governance.
80. [x] EM bounded core comparator coverage expanded:
    nonlinear benchmark-schema and external-reference validator-required metric sets now also include additional bounded homogeneous/heterogeneous core assertion comparators (`em_homogeneous_sigma_omega_scale_spread_ratio`, `em_homogeneous_material_heterogeneity_index`, `em_homogeneous_assignment_coverage_ratio`, `em_homogeneous_fallback_coefficient_ratio`, `em_heterogeneous_sigma_omega_scale_mean`, `em_heterogeneous_material_heterogeneity_index`, `em_heterogeneous_assignment_coverage_ratio`), with corresponding M6 baseline entries added so partial core EM comparator payloads fail enforce-mode governance.
81. [x] EM core comparator fixture completeness closed for conductivity spread:
    nonlinear benchmark-schema and external-reference validator-required metric sets now also require the remaining core conductivity-spread assertion comparators (`em_homogeneous_conductivity_spread_ratio`, `em_heterogeneous_conductivity_spread_ratio`), with matching M6 baseline entries so incomplete core EM comparator payloads no longer pass enforce-mode governance.
82. [x] Electro-thermal comparator governance depth expanded for time-scale assertions:
    nonlinear benchmark-schema and external-reference validator-required metric sets now include electro benign/pathological time-scale assertion comparators (`electro_thermal_benign_time_scale_mean`, `electro_thermal_pathological_time_scale_mean`), with matching M6 baseline entries so incomplete electro-thermal comparator payloads fail enforce-mode governance.
83. [x] Coupled-flow transient assertion comparator governance depth expanded:
    nonlinear benchmark-schema and external-reference validator-required metric sets now include coupled-flow transient assertion comparators (`transient_max_residual_norm`, `transient_max_energy_growth_ratio`, `transient_prepared_cache_hit_ratio`, `transient_prepared_cache_misses`) for `cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, and `fsi_coupled_gpu_provider`, with matching M6 baseline entries so incomplete transient comparator payloads fail enforce-mode governance.
84. [x] Coupled-flow release-readiness transient posture/trend gating expanded:
    `release_readiness_nonlinear.py` now evaluates coupled-flow transient assertion thresholds and rolling trend ratios (`transient_max_residual_norm`, `transient_max_energy_growth_ratio`, `transient_prepared_cache_hit_ratio`, `transient_prepared_cache_misses`) with profile-tuned policy defaults, surfacing transient residual/energy/cache regressions as explicit readiness reasons with operator summary visibility.
85. [x] Thermo/thermal comparator governance depth expanded across schema and external-reference enforcement:
    nonlinear benchmark-schema validation now requires standalone-thermal assertion coverage for `thermal_standalone_ramp_gpu_provider`, and external-reference validator-required metric sets now include thermo-mech kickoff, smooth-ramp (artifact and non-artifact), shock-oscillatory (artifact and non-artifact), and standalone-thermal assertion comparators, with matching M6 baseline entries so incomplete thermo/thermal comparator payloads fail enforce-mode governance.
86. [x] Priority-domain comparator governance completeness expanded for electro and constitutive reference-complex severity signals:
    external-reference validator-required metric sets now include benign/pathological electro Joule/severity/temporal/conductivity assertions, thermo pathological heterogeneity, and plastic/contact reference-complex severity peak/mean assertions, with matching M6 baseline entries so incomplete electro/plastic/contact/thermo comparator payloads fail enforce-mode governance.
87. [x] Coupled-flow comparator governance completeness expanded for profile and stepping assertions:
    external-reference validator-required metric sets now also include the remaining CFD/CHT/FSI profile-property and stepping assertions (`*_dynamic_viscosity_pa_s`, `*_inlet_velocity_m_per_s`, `*_turbulence_intensity`, `*_profile_point_count`, `cht_step_count`, `cht_time_step_s`, `fsi_step_count`, `fsi_time_step_s`, `fsi_cfd_profile_point_count`) with matching M6 baseline entries so incomplete coupled-flow comparator payloads fail enforce-mode governance.
88. [x] Core nonlinear and benign-thermo comparator governance baseline expanded:
    external-reference validator-required metric sets now include the remaining schema-governed nonlinear core/proxy/reference and benign-thermo fixture assertions (`nonlinear_assembly_*`, `nonlinear_stress_*`, `nonlinear_softening_*`, `nonlinear_path_mix_*`, `plasticity_nonlinear_*`, `contact_nonlinear_*`, `contact_frictionless_severity_*`, `plasticity_hardening_reference_severity_*`, `thermo_gradient_benign_*`) with matching M6 baseline entries so incomplete nonlinear/benign-thermo comparator payloads fail enforce-mode governance.
89. [x] Schema-to-external comparator parity guardrail added:
    governance tests now assert fixture-level parity between nonlinear benchmark-schema required assertion sets and external-reference validator-required metric sets, so future schema additions cannot silently bypass enforce-mode external comparator policy.
90. [x] Benchmark-schema performance telemetry enforcement expanded across schema-governed provider fixtures:
    `validate_analysis_report_nonlinear.py` now requires finite GPU performance telemetry fields (`gpu_speedup_ratio`, `gpu_solver_solve_ms`, `gpu_solver_backend`) for additional nonlinear, thermo/thermal, and electro provider fixtures, ensuring scale-posture evidence remains present across the widened schema-governed benchmark fixture surface.
91. [x] Release-readiness key-performance fixture depth expanded for schema-governed thermo/electro/nonlinear fixtures:
    `release_readiness_nonlinear.py` now includes additional key-performance fixtures (`nonlinear_assembly_stress_gpu_provider`, `nonlinear_softening_proxy_gpu_provider`, `nonlinear_load_path_mix_gpu_provider`, `nonlinear_plasticity_proxy_gpu_provider`, `nonlinear_contact_proxy_gpu_provider`, `nonlinear_contact_frictionless_reference_gpu_provider`, `nonlinear_plastic_hardening_reference_gpu_provider`, `thermo_mech_kickoff_gpu_provider`, `thermo_gradient_benign_gpu_provider`, `thermo_ramp_smooth_gpu_provider`, `thermo_ramp_smooth_field_artifact_gpu_provider`, `thermo_shock_oscillatory_gpu_provider`, `thermo_shock_oscillatory_field_artifact_gpu_provider`, `thermal_standalone_ramp_gpu_provider`, `electro_thermal_joule_benign_gpu_provider`) for speedup-floor and slowdown-trend gating, reducing scale-regression blind spots across the widened benchmark-schema fixture surface.
92. [x] Meshing/adaptivity kickoff landed with deterministic adaptive refinement profile:
    additive meshing-core/runtime prep profile `adaptive_refine` now runs through `geometry.prep_for_analysis/v1`, applying deterministic refinement-oriented heuristics and quality shaping (`min_scaled_jacobian`, `mean_aspect_ratio`) with runtime contract and conformance tests, establishing first active meshing/adaptivity baseline without contract-version churn.
93. [x] CAD interop ingestion depth expanded with OBJ import baseline:
    `runmat-geometry-io` now supports additive deterministic OBJ ingestion (`obj/v1`) with polygon-face triangulation (including negative-index references), degenerate-face pruning diagnostics, and runtime geometry contract coverage so `.obj` assets load through `geometry.load/v1` without unsupported-format fallbacks.
94. [x] CAD interop ingestion depth expanded with PLY import baseline:
    `runmat-geometry-io` now supports additive deterministic ASCII PLY ingestion (`ply/v1`) with polygon-face triangulation and degenerate-face pruning diagnostics, and runtime geometry contract coverage confirms `.ply` assets pass through `geometry.inspect/v1` + `geometry.load/v1` without unsupported-format fallbacks.
95. [x] CAD interop format detection robustness expanded for extension-less OBJ/PLY payloads:
    geometry format sniffing now identifies OBJ and ASCII PLY payloads from header/content signatures even when file extensions are non-canonical, allowing `geometry.inspect/v1` and `geometry.load/v1` to route supported importers instead of producing unsupported-format errors for extension-less CAD mesh inputs.
