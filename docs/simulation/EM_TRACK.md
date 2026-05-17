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

## Next Recommended EM Slice

- Extend from the landed `sigma(omega)` baseline into stronger dispersive terms and external-reference comparator depth.
