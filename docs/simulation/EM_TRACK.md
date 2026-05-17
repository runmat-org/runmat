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

## Next Recommended EM Slice

- Extend from the landed `sigma(omega)` baseline into stronger dispersive terms and external-reference comparator depth.
