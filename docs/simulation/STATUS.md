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
| CFD core | Steady/transient fluid foundation | [ ] |
| CHT | Fluid-thermal coupled family | [ ] |
| FSI | Structural-fluid coupled family | [ ] |
| Acoustics | Acoustic solver family | [ ] |
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
- Next material gains come from constitutive fidelity, external references, and missing physics families.
