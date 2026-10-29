---
description: Planned RunMat simulation features that the docs describe but the current release does not include yet, grouped by area, with acceptance criteria.
keywords: [release status, roadmap, planned features, not yet in the current release, FEA]
doc_type: release-status
generated: node scripts/docs/generate-pages.mjs
---
# Release status

The RunMat simulation docs describe the complete system. A feature that the current release does not include yet carries the marker [not yet in the current release](./release-status.md) where the docs describe it, and appears on this page under its id. `runmat check` reports the same id when a study uses the feature, so the docs, the diagnostics and the RunMat agent share one key.

This page is generated from `docs/simulation/pending-features.json`. When a feature ships, its entry records the release, and its markers are removed from the docs.

Which procedures each physics family runs is listed in the [family and procedure matrix](./generated/family-procedure-matrix.md).

## Structural mechanics

### structural.nonlinear.sequence

**Nonlinear structural steps inside procedure graphs (preload → nonlinear → modal; thermal prefix → nonlinear).**

- Kind: procedure.
- Ships when: A study whose nonlinear step follows or precedes other steps (`after`) passes `runmat check` and runs to completion; state transfers into and out of the nonlinear step are tested.
- Described in: [Procedure reference](./language/reference/procedures.md).

## Electromagnetics

### electromagnetic.nonlinear

**Nonlinear magnetics with B–H curves.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electromagnetic nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### electromagnetic.transient

**Electromagnetic transient (eddy-current A–φ time stepping).**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electromagnetic transient` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### magnetic.harmonic.vector-potential

**Nonzero prescribed magnetic vector potential in harmonic and magnetostatic studies.**

- Kind: constraint.
- Ships when: A nonzero `magnetic.vector_potential` boundary value runs in magnetostatic and harmonic studies and matches a closed-form field.
- Described in: [Electrical and magnetic constraints](./constraints/electromagnetic.md).

### magnetic.periodic

**Periodic and phase-shifted periodic magnetic boundaries for motor and transformer sectors.**

- Kind: constraint.
- Ships when: `magnetic.periodic` and `magnetic.phase_shifted_periodic` run in magnetostatic and harmonic studies; a sector model matches the full model.
- Described in: [Electrical and magnetic constraints](./constraints/electromagnetic.md).

## Acoustics

### acoustic.nonlinear

**Nonlinear acoustics procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `acoustic nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### acoustic.periodic.rotational

**Rotational (cyclic) acoustic periodicity, and periodicity in acoustic transient studies.**

- Kind: constraint.
- Ships when: Rotated periodic pairs and transient periodic pairs run; a cyclic sector matches the full annulus modes.
- Described in: [Acoustic constraints](./constraints/acoustic.md).

### acoustic.static

**Acoustic static (pressure-only Laplace) procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `acoustic static` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Scope: awaiting a product decision on whether this procedure belongs in the final system.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

## Fluid flow

### fluid.harmonic

**Fluid harmonic (linearized flow about a base state) procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Scope: awaiting a product decision on whether this procedure belongs in the final system.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### fluid.nonlinear

**Fluid nonlinear procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### fluid.periodic

**Periodic fluid boundary pairs (`fluid.periodic`).**

- Kind: constraint.
- Ships when: A study with `fluid.periodic` on matching face pairs checks and runs; a periodic channel matches the infinite-channel Poiseuille profile.
- Described in: [Fluid constraints](./constraints/fluid.md).

## Thermomechanical

### thermomechanical.harmonic

**Thermomechanical harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `thermomechanical harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### thermomechanical.transient

**Thermomechanical transient with two-way coupling.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `thermomechanical transient` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

## Electrothermal

### electrothermal.harmonic

**Electrothermal harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electrothermal harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### electrothermal.nonlinear

**Electrothermal nonlinear procedure (temperature-dependent two-way coupling).**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electrothermal nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

## Conjugate heat transfer

### conjugate-heat-transfer.harmonic

**Conjugate heat transfer harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `conjugate_heat_transfer harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### conjugate-heat-transfer.nonlinear

**Conjugate heat transfer with temperature-dependent properties.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `conjugate_heat_transfer nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

## Fluid–structure interaction

### fluid-structure-interaction.harmonic

**Fluid–structure interaction harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid_structure_interaction harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).

### fluid-structure-interaction.nonlinear

**Fluid–structure interaction nonlinear procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid_structure_interaction nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Family and procedure matrix](./generated/family-procedure-matrix.md).
