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
- EM external-reference coverage expanded: M6 baseline now includes additional homogeneous/heterogeneous EM comparator metrics (dispersion + field-quality proxies) plus boundary/phased-source comparator entries, and external-reference artifact validation now requires core EM fixture metric sets so partial EM comparator payloads fail enforce-mode governance checks.
- Priority-domain external-reference coverage expanded beyond EM: external-reference baseline + validator policy now require comparator metric presence for representative thermo/electro/plastic/contact fixtures so partial non-EM comparator payloads fail enforce-mode governance checks alongside EM/CFD/CHT/FSI requirements.
- Next material gains come from constitutive fidelity, external references, and missing physics families.
