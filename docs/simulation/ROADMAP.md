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
