---
description: Index of the RunMat FEA qualification records, one per capability, each stating its acceptance obligations and the tests that hold it to account.
keywords: [qualification records, validation, acceptance criteria, evidence catalog, FEA verification]
doc_type: records
generated: node scripts/docs/generate-pages.mjs
---
# Validation records

Each qualification record states what one capability must do, the acceptance obligations that pin it down, and the execution envelopes (formulation, element order, execution provider) it must hold in. The tests that support an obligation declare it at the test, and the evidence catalog reports PASS, FAIL or OPEN for every obligation and envelope from an actual test run. The [validation methodology](./methodology.md) explains how to read them.

The records live in the source repository under `docs/simulation/qualification/`; the generated evidence catalog is `docs/simulation/qualification/generated/evidence-catalog.md`.

| Record | File |
| --- | --- |
| Original-source accuracy evidence at cold import | `qualification/accuracy-cold-import.md` |
| Authored matched-impedance duct: pending execution gate | `qualification/acoustic-authored-duct.md` |
| Authored incident-pressure matched duct | `qualification/acoustic-authored-incident-duct.md` |
| Canonical acoustic frequency response — bounded execution evidence | `qualification/acoustic-frequency-response.md` |
| Generated complex acoustic spectra | `qualification/acoustic-generated-spectra.md` |
| Normally incoming acoustic pressure | `qualification/acoustic-incident-pressure.md` |
| First-order local acoustic radiation | `qualification/acoustic-local-radiation.md` |
| Canonical two-fluid acoustic material interface | `qualification/acoustic-material-interface.md` |
| Matching periodic and Bloch pressure | `qualification/acoustic-periodic.md` |
| Acoustic physical-power acceptance | `qualification/acoustic-power-closure.md` |
| Shared right LU and canonical Helmholtz — bounded native qualification | `qualification/acoustic-right-lu.md` |
| Acoustic boundary volume velocity — bounded qualification | `qualification/acoustic-volume-velocity.md` |
| Adaptive nodal field transfer | `qualification/adaptive-nodal-transfer.md` |
| Anchored local-energy coercivity verification | `qualification/anchored-local-energy.md` |
| Atomic study publication — implementation checkpoint | `qualification/atomic-study-publication.md` |
| Shared original attempt history | `qualification/attempt-history.md` |
| Authored canonical acoustic execution envelope | `qualification/authored-acoustic-execution.md` |
| Authored standalone thermal execution | `qualification/authored-standalone-thermal.md` |
| Canonical acoustic harmonic qualification | `qualification/canonical-acoustic.md` |
| Canonical nodal body density — qualification incomplete | `qualification/canonical-nodal-body-density.md` |
| Canonical steady incompressible fluid evidence | `qualification/canonical-steady-fluid.md` |
| Prepared global contact operator and original configuration | `qualification/contact-global-operator.md` |
| Prepared contact in the original nonlinear load controller | `qualification/contact-load-controller.md` |
| Original contact material history and smooth friction contribution | `qualification/contact-material-history.md` |
| Canonical contact pairing and smooth normal contribution | `qualification/contact-pairing-normal.md` |
| Penalty/Coulomb contact point operator — qualification boundary | `qualification/contact-penalty-point.md` |
| Contact preparation admission | `qualification/contact-preparation-admission.md` |
| Coupled solid/contact Newton core | `qualification/contact-solid-newton.md` |
| Canonical contact surface geometry — qualification boundary | `qualification/contact-surface-search.md` |
| Analytic curved Tet10 harmonic qualification | `qualification/curved-solid-harmonic.md` |
| Authored static electrical vertical-chain evidence | `qualification/electrical-authored-static.md` |
| Canonical static electrical qualification | `qualification/electrical-static.md` |
| Electrostatic volume charge — bounded qualification evidence | `qualification/electrical-volume-charge.md` |
| Canonical backward-Euler fluid interval | `qualification/fluid-backward-euler.md` |
| Original authored-boundary fluid balances | `qualification/fluid-boundary-balances.md` |
| Closed-component fluid pressure references | `qualification/fluid-closed-pressure-reference.md` |
| Scheduled Fourier duct qualification witness | `qualification/fluid-fourier-native.md` |
| Prepared CFD geometric reconstruction | `qualification/fluid-geometric-reconstruction.md` |
| Signed canonical mass-flow sources | `qualification/fluid-mass-flow-native.md` |
| Original-field fluid temporal blocks | `qualification/fluid-temporal-blocks.md` |
| Shared timed inputs for canonical fluid intervals | `qualification/fluid-time-signals.md` |
| Original fluid temporal checkpoints and native block admission | `qualification/fluid-transient-checkpoints.md` |
| Portable fresh canonical fluid transient intent | `qualification/fluid-transient-workload.md` |
| Shared frequency publication contract | `qualification/frequency-publication.md` |
| Governed GENERAL sparse solver resources | `qualification/general-sparse-resources.md` |
| Admitted GENERAL sparse solve | `qualification/general-sparse-solve.md` |
| Generated tensor and spatial-field value layout | `qualification/generated-value-layout.md` |
| Harmonic Maxwell continuum qualification | `qualification/harmonic-maxwell-continuum.md` |
| Magnetostatic vector potential — physical volume-current baseline | `qualification/magnetostatic-volume-current.md` |
| Original field inputs across portable structural studies | `qualification/original-field-transport.md` |
| Shared post-mesh signal binding boundary | `qualification/post-mesh-signal-binding.md` |
| Retained steady Joule heating into canonical thermal assembly | `qualification/retained-steady-electrothermal.md` |
| Shared canonical volume-gradient correction | `qualification/shared-gradient-mapping.md` |
| Small-strain J2 solid qualification — implementation in progress | `qualification/small-strain-j2-solids.md` |
| Solid harmonic qualification: independent one-DOF reference | `qualification/solid-harmonic.md` |
| Solid rod harmonic qualification | `qualification/solid-rod-harmonic.md` |
| Solid rod transient qualification — not qualified | `qualification/solid-rod-transient.md` |
| Sparse coercivity preparation and admission | `qualification/sparse-coercivity.md` |
| Source-independent sparse uniqueness certificate | `qualification/sparse-right-inverse.md` |
| Process-local sparse solver preparation reuse | `qualification/sparse-solver-preparation-reuse.md` |
| Structural adaptive iteration authority | `qualification/structural-adaptive-accuracy.md` |
| Structural adaptation from original solved results | `qualification/structural-adaptive-original-results.md` |
| Structural affine transient — bounded continuum evidence | `qualification/structural-affine-transient.md` |
| Canonical solid recovery qualification | `qualification/structural-recovery.md` |
| Structural transient fixed-basis execution | `qualification/structural-transient-fixed-basis.md` |
| Physical tetrahedral edge-space prerequisite | `qualification/tetrahedral-edge-space.md` |
| Nonuniform thermal cooling: spatial qualification window | `qualification/thermal-cooling-window.md` |
| Canonical initial-field input qualification | `qualification/thermal-field-inputs.md` |
| Thermal generated-input qualification | `qualification/thermal-generated-inputs.md` |
| Native thermal radiation and original temperature histories | `qualification/thermal-radiation-native.md` |
| Thermal surface-to-ambient radiation — bounded nonlinear procedure | `qualification/thermal-radiation.md` |
| Solved nonuniform temperature to same-mesh thermoelasticity | `qualification/thermal-thermoelastic-continuum.md` |
