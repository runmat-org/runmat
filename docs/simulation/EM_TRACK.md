# Maxwell EM Track

Last updated: 2026-05-17

## Current Position

- Phase 0-4: complete.
- Phase 5-15: complete.
- Phase 16: complete (frequency sweep + resonance governance).

## Achieved Baseline

1. Model-owned EM schema and runtime operation surface.
2. FEA-backed EM solve path with harmonic/block-coupled maturity increments.
3. Rich EM diagnostics and typed quality reasons.
4. `analysis.results` and `analysis.trends` EM governance surfaces.
5. EM reference fixture family with conformance and baseline drift tracking.

## Remaining to EM Completion

1. Frequency-dependent constitutive modeling in EM solve coefficients.
   Baseline complete: material `sigma(omega)` interpolation and dispersive-loss coupling from additive frequency-response points with EM governance thresholds.
2. Higher-fidelity Maxwell-form implementation depth beyond current proxy approximations.
3. Stronger external-reference EM comparators integrated into protected-branch policy.
   Baseline complete: external-reference benchmark generation now compares expanded EM threshold-assertion metrics (including `sigma(omega)`, dispersive-loss, homogeneous/heterogeneous quality proxies, and boundary/phased-source entries) and governance validation enforces complete EM fixture metric-set presence in enforce mode.
4. Continued robustness/performance hardening under larger EM workloads.
   Baseline increment landed: additive EM harmonic solver controls (`residual_target`, `harmonic_tolerance`, `harmonic_max_iterations`) now route from runtime options into FEA solve behavior with validated contracts and diagnostic visibility.
   Baseline increment landed: study workflow orchestration (`analysis.validate_study` / `analysis.run_study`) now validates and propagates additive EM run controls through `AnalysisStudySpec.electromagnetic_run_options`, and the EM create-model profile seeds default EM domain parameters so EM study runs execute without out-of-band model edits.
   Baseline increment landed: study workflow typed outputs now surface requested/resolved EM run options on `analysis.plan_study`/`analysis.run_study`, improving API-level reproducibility for programmatic study orchestration.
   Baseline increment landed: `analysis.run_study` typed output now includes run-operation identity and `quality_reasons`, so EM study automation can extract immediate execution verdict context without a separate results retrieval hop.
   Baseline increment landed: `analysis.plan_study` typed output now includes planned run-operation identity, allowing EM study orchestration clients to select the electromagnetic run-contract path without introspecting operation-sequence arrays.
   Baseline increment landed: `analysis.validate_study` typed output now includes structured issue details (`code`, `message`) so EM study-option validation failures are machine-readable without external issue-code message mapping.
   Baseline increment landed: `analysis.run_study` typed output now includes solver gates and full run provenance (`solver_convergence`, `result_quality`, `provenance`), improving EM automation-side extraction of readiness posture and execution lineage.
   Baseline increment landed: `analysis.run_study_sweep` now supports deterministic sequential multi-study orchestration, including EM studies, with typed aggregate run-entry outputs and sweep evidence artifacts for programmatic study pipelines.
   Baseline increment landed: `analysis.run_study_sweep` now supports continue-on-failure mode (`fail_fast=false`) with typed per-study failure entries, so EM-inclusive sweeps can preserve successful runs while still surfacing invalid-study diagnostics.
   Baseline increment landed: additive `analysis.validate_study_sweep` now provides typed sweep-level and per-study preflight validation (`issue_codes` plus structured issues) with persisted validation evidence artifacts, improving EM-inclusive sweep gating ergonomics before run dispatch.
   Baseline increment landed: additive `analysis.plan_study_sweep` now provides typed sweep-level preflight planning entries and aggregate planned/failure counts under `fail_fast` policy control, improving EM-inclusive sweep orchestration ergonomics before run dispatch.
   Baseline increment landed: electromagnetic FEA now applies dispersive-loss-derived phase attenuation to conductive coupling terms and emits additive phase attenuation diagnostics (`dispersive_phase_attenuation_mean`, `dispersive_phase_conductivity_attenuation_ratio`) with conformance/external-reference governance enforcement for homogeneous and heterogeneous EM provider fixtures.
   Baseline increment landed: release-readiness Maxwell posture governance now consumes homogeneous/heterogeneous EM phase attenuation threshold assertions, adding profile-tuned branch readiness reasons for phase-fidelity regressions beyond existing energy/residual/source posture gates.
   Baseline increment landed: release-readiness Maxwell posture governance now also evaluates rolling worsening ratios for homogeneous/heterogeneous EM phase attenuation assertions (mean and conductivity attenuation ratio), so baseline-relative phase-fidelity drift is branch-gated alongside static threshold posture.
   Baseline increment landed: external-reference and nonlinear schema validators now require both EM phase attenuation mean and phase conductivity attenuation ratio metrics for homogeneous/heterogeneous provider fixtures, preventing partial phase-fidelity comparator payloads from passing enforce-mode governance.
   Baseline increment landed: external-reference and nonlinear schema governance now also requires homogeneous/heterogeneous source-material-alignment assertion comparators, extending EM source-fidelity policy coverage in enforce-mode validation.
   Baseline increment landed: external-reference and nonlinear schema governance now also requires homogeneous/heterogeneous dispersive-coupling-ratio assertion comparators, extending EM dispersive-coupling fidelity policy coverage in enforce-mode validation.
   Baseline increment landed: external-reference and nonlinear schema governance now also requires homogeneous/heterogeneous source-region-coverage and boundary-anchor assertion comparators, extending EM source/boundary-fidelity policy coverage in enforce-mode validation.
   Baseline increment landed: external-reference and nonlinear schema governance now also requires expanded non-core sparse/fallback/overlap/boundary-kernel source/boundary comparator sets (assignment/source-coverage/material-alignment/boundary-anchor and boundary-kernel anchor/leakage signals), extending enforce-mode EM fidelity coverage beyond core homogeneous/heterogeneous fixtures.
   Baseline increment landed: external-reference and nonlinear schema governance now also requires boundary-penalty residual norm comparators (`em_boundary_penalty_real_residual_norm`, `em_boundary_penalty_imag_residual_norm`) and phased-source overlap/interference comparators (`em_phased_source_overlap_ratio`, `em_phased_source_interference_index`), extending enforce-mode policy depth for boundary/phased EM fixtures.
   Baseline increment landed: release-readiness Maxwell posture governance now also consumes homogeneous/heterogeneous dispersive-coupling-ratio threshold assertions with profile-tuned maxima, so coupling-ratio excursions are branch-gated alongside other EM posture signals.
   Baseline increment landed: release-readiness Maxwell posture governance now also consumes homogeneous/heterogeneous boundary-anchor-ratio threshold assertions with profile-tuned minima, so anchor-ratio regressions are branch-gated alongside other EM posture signals.
   Baseline increment landed: release-readiness Maxwell posture governance now also consumes boundary-penalty residual norm assertions (`em_boundary_penalty_real_residual_norm`, `em_boundary_penalty_imag_residual_norm`) and phased-source overlap/interference assertions (`em_phased_source_overlap_ratio`, `em_phased_source_interference_index`) with profile-tuned maxima, so boundary/phased fidelity regressions are branch-gated alongside other EM posture signals.

## Next Recommended EM Slice

- Extend from the landed `sigma(omega)` baseline into stronger dispersive terms and external-reference comparator depth.
