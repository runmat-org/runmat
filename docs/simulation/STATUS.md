# Analysis System Status

Last updated: 2026-05-17

Legend: `[x]` complete baseline, `[~]` active deepening, `[ ]` not started.

## Capability Snapshot

| Capability family | Target | Status |
| --- | --- | --- |
| Runtime contract spine | Stable versioned `analysis.*` + `geometry.*` operations | [x] |
| Structural core | Linear/modal/transient/nonlinear production baseline | [x] |
| Standalone thermal | Constitutive-depth thermal with acceptance envelopes | [~] |
| Thermo-mechanical | Credible coupled behavior + benchmark governance | [~] |
| Electro-thermal | Credible Joule coupling + benchmark governance | [~] |
| Plasticity depth | Broader constitutive realism | [~] |
| Contact depth | Broader contact formulation realism | [~] |
| Maxwell EM | From scaffolding to governance-backed solver path | [~] |
| CFD core | Steady/transient fluid foundation | [~] |
| CHT | Fluid-thermal coupled family | [~] |
| FSI | Structural-fluid coupled family | [~] |
| Acoustics | Acoustic solver family | [~] |
| Meshing/adaptivity | Production meshing/refinement pipeline | [ ] |
| External-reference gating (M6) | Protected-branch enforced external comparators | [~] |
| Performance/scale regression gates | SLO-backed perf readiness across key workloads | [~] |

## EM Snapshot

| EM track | Status |
| --- | --- |
| Phase 0-4 (schema -> first assembled static path) | [x] |
| Phase 5-15 (material/source/boundary/harmonic governance depth) | [x] |
| Phase 16 (frequency sweep + resonance governance) | [x] |
| Frequency-dependent constitutive + higher-fidelity Maxwell path | [~] |

## What This Means

- The system is beyond scaffolding and in governed solver deepening.
- Structural baseline is stable.
- EM is advanced but not final industrial Maxwell parity.
- EM now includes additive material frequency-response support (`sigma(omega)` baseline) with fixture-level governance thresholds.
- EM diagnostics/governance now include dispersive-loss scale and dispersive conductivity coupling ratios for frequency-dependent runs.
- EM fidelity/governance now includes dispersive-loss-derived phase attenuation signals for conductivity coupling (`dispersive_phase_attenuation_mean`, `dispersive_phase_conductivity_attenuation_ratio`) with enforced benchmark/external-reference metric coverage on key EM provider fixtures.
- EM external-reference/nonlinear schema governance now explicitly requires both phase attenuation mean and phase conductivity attenuation ratio metrics for homogeneous/heterogeneous EM provider fixtures, closing partial phase-fidelity evidence gaps in enforce-mode validation.
- EM external-reference/nonlinear schema governance now also enforces source-material-alignment comparator coverage for homogeneous/heterogeneous EM provider fixtures, preventing source-fidelity baseline gaps from passing enforce-mode benchmark/external-reference policy.
- EM external-reference/nonlinear schema governance now also enforces dispersive-coupling-ratio comparator coverage for homogeneous/heterogeneous EM provider fixtures, preventing dispersive-coupling evidence gaps from passing enforce-mode benchmark/external-reference policy.
- EM external-reference/nonlinear schema governance now also enforces homogeneous/heterogeneous source-region-coverage and boundary-anchor comparator coverage, preventing source/boundary-fidelity evidence gaps from passing enforce-mode benchmark/external-reference policy.
- EM external-reference/nonlinear schema governance now also enforces expanded non-core EM source/boundary comparator sets for sparse/fallback/overlap/boundary-kernel fixtures (assignment/source coverage/material alignment/boundary anchor plus boundary-kernel anchor/leakage signals), preventing partial non-core EM fidelity payloads from passing enforce-mode policy.
- EM external-reference/nonlinear schema governance now also enforces boundary-penalty residual norm and phased-source overlap/interference comparator coverage (`em_boundary_penalty_real_residual_norm`, `em_boundary_penalty_imag_residual_norm`, `em_phased_source_overlap_ratio`, `em_phased_source_interference_index`), preventing partial boundary/phased EM comparator payloads from passing enforce-mode policy.
- External-reference comparator generation now supports threshold-assertion-backed EM metrics in the M6 baseline artifact.
- Thermal/thermo governance bands were tightened across gradient/ramp/shock and standalone thermal fixtures to reduce permissive drift windows.
- Plastic/contact nonlinear proxy and reference fixtures now use tighter constitutive severity and load realization/amplification governance bands.
- Electro-thermal benign/pathological fixtures and nonlinear mixed-load coupling now use tighter Joule/conductivity/severity/time-scale governance bands.
- Nonlinear assembly/stress/softening fixtures now use tighter convergence governance bands for backtracks, norms, and spike/stall behavior.
- CFD schema/profile kickoff landed: additive core domain contracts plus `analysis.create_model` steady/transient CFD templates are in place as the first Phase C step.
- CFD first run-path baseline landed: `analysis.run_cfd/v1` now executes through the runtime/FEA stack with typed outputs and first fluid diagnostics (`FEA_CFD_FLOW`) under existing governance posture.
- CFD benchmark/trend baseline landed: conformance harness now includes CFD fixtures with flow-metric threshold assertions, and runtime trends classify CFD runs as a dedicated run family.
- CHT first run-path baseline landed: `analysis.run_cht/v1` now executes coupled CFD + thermal runs with typed transient+thermal payloads and first CHT diagnostics (`FEA_CHT_COUPLING`, `FEA_CFD_FLOW`) under existing quality/provenance posture.
- CHT benchmark/trend baseline landed: conformance harness now includes CHT fixtures with coupled flow/coupling threshold assertions, and runtime trends classify CHT runs as a dedicated run family (`AnalysisRunKind::Cht`).
- FSI first run-path and governance baseline landed: additive `analysis.run_fsi/v1` now executes coupled transient-structural + CFD runs with typed transient payloads, emits FSI diagnostics (`FEA_FSI_COUPLING`, `FEA_CFD_FLOW`), classifies trends under `AnalysisRunKind::Fsi`, and is covered by conformance harness fixtures (`fsi_coupled_cpu`, `fsi_coupled_gpu_provider`, `fsi_coupled_gpu_fallback`) with threshold assertions.
- Study workflow canonical evidence baseline landed: additive high-level runtime operations `analysis.validate_study/v1`, `analysis.plan_study/v1`, and `analysis.run_study/v1` now persist deterministic study evidence artifacts (`validate.json`, `plan.json`, `run.json`) keyed by study fingerprint and expose evidence artifact paths in typed outputs.
- Coupled-family artifact classification hardened: persisted filesystem run artifacts now retain family-accurate `op_version` labels for CFD/CHT/FSI (`analysis.run_cfd/v1`, `analysis.run_cht/v1`, `analysis.run_fsi/v1`) instead of collapsing coupled runs into generic transient labels.
- Coupled-family governance validation hardened: `validate_analysis_report_nonlinear.py` now enforces CFD/CHT/FSI provider fixture assertion coverage (`cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, `fsi_coupled_gpu_provider`) so coupled benchmark regressions are policy-visible in CI.
- External-reference comparator coverage now includes coupled-family threshold metrics in the M6 baseline (`cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, `fsi_coupled_gpu_provider`) so CFD/CHT/FSI regressions are included in governance-side reference comparisons.
- External-reference artifact validation now enforces required coupled-family metric presence for CFD/CHT/FSI fixture records, so incomplete comparator payloads fail governance validation even when schema fields are otherwise well-formed.
- Governance CI now executes `scripts.tests.test_external_reference_baseline` in the release-readiness unittest bundle, so coupled-family baseline coverage regressions are caught before external-reference artifact generation.
- Acoustics contract kickoff landed: additive `analysis.create_model` profile `acoustic_harmonic` now seeds a deterministic harmonic-template model scaffold (modal-step placeholder) under the existing versioned contract surface with runtime contract coverage tests.
- EM external-reference coverage hardened: M6 baseline now spans expanded homogeneous/heterogeneous EM comparator metrics, boundary/phased-source entries, and additional sparse/fallback/overlap/boundary-kernel fixture comparators, with artifact validation requiring complete fixture-level metric sets so partial EM comparator payloads fail enforce-mode governance checks.
- Priority-domain external-reference coverage expanded beyond EM: external-reference baseline + validator policy now require comparator metric presence for representative thermo/electro/plastic/contact fixtures so partial non-EM comparator payloads fail enforce-mode governance checks alongside EM/CFD/CHT/FSI requirements.
- Key-workload benchmark-schema validation now enforces finite GPU performance telemetry fields (`gpu_speedup_ratio`, `gpu_solver_solve_ms`, `gpu_solver_backend`) on representative nonlinear/thermo/electro/EM/acoustic/CFD/CHT/FSI provider fixtures, ensuring release benchmark artifacts retain scale-posture evidence.
- EM benchmark-schema validator coverage now extends beyond core homogeneous/heterogeneous fixtures to additional sparse/fallback/overlap/boundary-kernel provider fixtures, requiring both key EM threshold assertions and finite GPU performance telemetry fields for those records.
- Release-readiness trend governance now includes key-performance fixtures (`electromagnetic_reference_*`, `acoustic_harmonic_gpu_provider`, `cfd_steady_gpu_provider`, `cht_coupled_gpu_provider`, `fsi_coupled_gpu_provider`) with speedup floor and slowdown-ratio checks under profile-tuned policy defaults, extending slowdown gating beyond the nonlinear core fixture subset.
- Release-readiness key-performance coverage now also includes representative thermo/electro/plastic/contact fixtures (`thermo_gradient_pathological_gpu_provider`, `electro_thermal_joule_pathological_gpu_provider`, `nonlinear_plastic_hardening_reference_complex_gpu_provider`, `nonlinear_contact_frictionless_reference_complex_gpu_provider`) for speedup-floor and slowdown-trend gating.
- EM runtime diagnostics now include native solve-cost telemetry (`FEA_EM_COST` with `prepared_build_ms`, `solve_ms`, `fallback_apply_count`), and conformance report extraction now consumes this diagnostic ahead of generic runtime timing fallback so EM performance evidence remains domain-native.
- Release-readiness domain posture governance now includes EM metric thresholds/trends (energy imbalance, flux divergence, residual norms, source realization/coverage/alignment, and breach/trend ratios), extending branch-profiled readiness gating to Maxwell posture quality in addition to performance and external-reference checks.
- Release-readiness EM posture gating now also enforces homogeneous/heterogeneous dispersive phase attenuation thresholds (mean and conductivity attenuation ratio assertions), integrating phase-fidelity regressions into branch-readiness breach accounting.
- Release-readiness EM phase-fidelity posture gating now also enforces rolling worsening limits for homogeneous/heterogeneous phase attenuation assertions (mean and conductivity attenuation ratio), so regressions against historical baseline are surfaced as dedicated readiness reasons in addition to static threshold breaches.
- Release-readiness EM posture gating now also enforces homogeneous/heterogeneous dispersive-coupling-ratio ceilings, promoting excessive coupling-ratio excursions into explicit readiness reasons and EM breach-rate calculations.
- Release-readiness EM posture gating now also enforces homogeneous/heterogeneous boundary-anchor-ratio minimums, promoting anchor-ratio regressions into explicit readiness reasons and EM breach-rate calculations.
- Release-readiness EM posture gating now also enforces boundary-penalty residual norm and phased-source overlap/interference assertion ceilings (`em_boundary_penalty_real_residual_norm`, `em_boundary_penalty_imag_residual_norm`, `em_phased_source_overlap_ratio`, `em_phased_source_interference_index`), promoting boundary/phased EM fidelity regressions into explicit readiness reasons and EM breach-rate calculations.
- Release-readiness EM posture gating now also enforces rolling worsening limits for boundary-penalty residual norm and phased-source overlap/interference assertions, surfacing dedicated readiness reasons when boundary/phased EM fidelity regresses against baseline posture.
- Release-readiness EM posture gating now also enforces non-core sparse/fallback/overlap/boundary-kernel assertion thresholds (assignment/fallback/source-coverage/material-alignment/boundary-anchor/interference/localization/leakage), surfacing dedicated readiness reasons when non-core EM fixture posture regresses.
- EM run controls now expose additive harmonic solver knobs (`residual_target`, `harmonic_tolerance`, `harmonic_max_iterations`) through `analysis.run_electromagnetic/v1`, with runtime option validation and FEA path wiring replacing prior hardcoded harmonic-solver constants.
- Study workflow now propagates additive EM run controls: `AnalysisStudySpec.electromagnetic_run_options` is validated in `analysis.validate_study/v1`, flagged as invalid when attached to non-EM study run kinds, and applied by `analysis.run_study/v1` when dispatching electromagnetic runs; the electromagnetic create-model profile now seeds a default EM domain so EM studies execute end-to-end without manual model patching.
- Study workflow typed outputs now expose EM execution options directly: `analysis.plan_study/v1` carries requested `electromagnetic_run_options`, and `analysis.run_study/v1` carries resolved EM execution options (including defaults when unspecified), so callers can audit study intent/execution parity without parsing evidence artifact files.
- Study workflow run outputs now include direct execution-context fields (`run_operation`, `run_op_version`, and `quality_reasons`) in `analysis.run_study/v1`, improving result-extraction ergonomics for automation paths that need immediate run verdict details.
- Study workflow plan outputs now include direct planned run-operation identity (`run_operation`, `run_op_version`) in `analysis.plan_study/v1`, improving programmatic routing ergonomics for downstream run-contract specific automation.
- Study workflow validation outputs now include typed issue details (`issues[]` with `code` + `message`) in `analysis.validate_study/v1` while preserving `issue_codes`, improving automation-side diagnostics without breaking existing code-based handling.
- Study workflow run outputs now additionally expose gate and lineage extraction fields (`solver_convergence`, `result_quality`, `provenance`) in `analysis.run_study/v1`, so callers can evaluate quality posture and backend/solver provenance directly from study execution responses.
- Study workflow now includes first sweep orchestration baseline via additive `analysis.run_study_sweep/v1`, enabling deterministic sequential execution of multiple study specs with typed aggregate run entries and sweep-level evidence artifacts for automation pipelines.
- Study workflow sweep execution now includes policy-controlled failure handling: `analysis.run_study_sweep/v1` supports `fail_fast` and returns typed partial-failure detail (`failed_count`, `failure_entries`) when configured to continue on invalid studies, improving resilient automation behavior.
- Study workflow now includes additive typed sweep preflight validation via `analysis.validate_study_sweep/v1`, returning sweep-level issue codes plus per-study structured issue details (`issue_codes` + typed `issues`) with persisted validation evidence artifacts for programmatic gate checks before sweep execution.
- Study workflow now includes additive typed sweep preflight planning via `analysis.plan_study_sweep/v1`, returning per-study planned run metadata plus aggregate planned/failure counts under policy-controlled failure handling (`fail_fast`) with persisted sweep planning evidence artifacts.
- Acoustics runtime baseline landed: additive `analysis.run_acoustic/v1` now executes harmonic-profile models through a modal-backed placeholder path with typed options/contracts (`AnalysisAcousticRunOptions`), emits acoustic diagnostics (`FEA_ACOUSTIC_PLACEHOLDER`), classifies trend summaries as `AnalysisRunKind::Acoustic`, and persists acoustic run artifacts with family-accurate `op_version` labels.
- Acoustics conformance/governance baseline landed: benchmark harness now includes acoustic fixtures (`acoustic_harmonic_cpu`, `acoustic_harmonic_gpu_provider`, `acoustic_harmonic_gpu_fallback`) with enforced acoustic threshold assertions, benchmark schema validation now requires acoustic provider assertion + GPU performance telemetry fields, and external-reference baseline/validator policy now requires acoustic comparator metrics so missing acoustic governance evidence fails enforce-mode validation.
- Acoustics governance depth expanded: acoustic provider assertions/external-reference comparators now also enforce modal orthogonality and relative-frequency-separation metrics (`acoustic_max_m_orthogonality_offdiag`, `acoustic_min_relative_frequency_separation`) in addition to mode-count/residual controls, tightening acoustic quality/regression visibility.
- Next material gains come from constitutive fidelity, external references, and missing physics families.
