---
description: What the current RunMat release runs for each physics family and procedure, which loads, constraints and outputs it accepts, and which documented features are planned, with acceptance criteria.
keywords: [release status, supported analyses, family and procedure matrix, planned features, not yet in the current release, FEA]
doc_type: release-status
generated: node scripts/docs/generate-pages.mjs
---
# Status

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

This page describes RunMat 0.7.0. It is generated from the capability ledger, `docs/simulation/ledger.json`, the one record of what the simulation system can do. The same ledger generates the compiler's table of family and procedure pairs, so `runmat check` and `runmat run` accept exactly what this page lists.

A documented feature that the current release does not include yet carries the marker [not yet in the current release](#planned-features) where the docs describe it, and appears below under its id. `runmat check` reports the same id when a study uses it, so the docs, the diagnostics and the RunMat agent share one key.

## Families and procedures

| Family | Runs in this release | Planned |
| --- | --- | --- |
| [Structural mechanics](./structural.md) | static, modal, prestressed modal, buckling, transient, harmonic, prestressed harmonic, nonlinear, transient (modal superposition), harmonic (modal superposition), prestressed harmonic (modal superposition), response spectrum (modal superposition), random response (modal superposition) | — |
| [Heat transfer](./thermal.md) | static, transient | — |
| [Electromagnetics](./electromagnetic.md) | static, harmonic | [transient](#electromagnetic-transient), [nonlinear](#electromagnetic-nonlinear) |
| [Acoustics](./acoustic.md) | harmonic, modal, transient | [static](#acoustic-static), [nonlinear](#acoustic-nonlinear) |
| [Fluid flow](./fluid.md) | static, transient | [harmonic](#fluid-harmonic), [nonlinear](#fluid-nonlinear) |
| [Thermomechanical](./thermomechanical.md) | static, nonlinear | [transient](#thermomechanical-transient), [harmonic](#thermomechanical-harmonic) |
| [Electrothermal](./electrothermal.md) | static, transient | [harmonic](#electrothermal-harmonic), [nonlinear](#electrothermal-nonlinear) |
| [Conjugate heat transfer](./conjugate-heat-transfer.md) | static, transient | [harmonic](#conjugate-heat-transfer-harmonic), [nonlinear](#conjugate-heat-transfer-nonlinear) |
| [Fluid–structure interaction](./fluid-structure-interaction.md) | static, transient | [harmonic](#fluid-structure-interaction-harmonic), [nonlinear](#fluid-structure-interaction-nonlinear) |

Modal superposition reuses a retained modal basis; every other entry solves the full system directly. The [procedure reference](../language/reference/procedures.md) gives the syntax of each.

## Loads, constraints and outputs

- **Loads** (23): `acoustic_pressure`, `ambient_temperature`, `body_force`, `current_density`, `electric_charge`, `electric_charge_density`, `electric_current`, `force`, `heat_flux`, `heat_generation`, `incident_pressure`, `inlet_velocity`, `magnetic_vector_potential`, `magnetic_volume_current_density`, `mass_flow`, `moment`, `normal_velocity`, `pressure`, `temperature_field`, `terminal_voltage`, `traction`, `volume_velocity`, `wall_motion`. See the [reference](../language/reference/constraints-loads-and-interfaces.md).
- **Constraints** (20): `acoustic.pressure`, `electrical.equipotential`, `electrical.floating_conductor`, `electrical.integral_current`, `electrical.perfect_conductor`, `electrical.potential`, `electrical.potential_reference`, `fluid.moving_wall`, `fluid.no_slip_wall`, `fluid.pressure_reference`, `fluid.slip_wall`, `fluid.symmetry`, `fluid.velocity`, `magnetic.coulomb_gauge`, `magnetic.vector_potential`, `magnetic.vector_potential_ground`, `structural.antisymmetry`, `structural.fixed`, `structural.symmetry`, `thermal.temperature`. See the [reference](../language/reference/constraints-loads-and-interfaces.md).
- **Output operations** (26): `absmax`, `collect`, `component`, `envelope`, `flux`, `imag`, `integral`, `interpolate`, `level`, `magnitude`, `max`, `mean`, `min`, `moment`, `norm`, `phase`, `principal`, `real`, `recover`, `resultant`, `retain_each`, `rms`, `sample`, `sound_power`, `variance`, `von_mises`. See the [reference](../language/reference/operations-and-output-queries.md).

## Planned features

### Structural mechanics

#### structural.nonlinear.sequence

**Nonlinear structural steps inside procedure graphs (preload → nonlinear → modal; thermal prefix → nonlinear).**

- Kind: procedure.
- Ships when: A study whose nonlinear step follows or precedes other steps (`after`) passes `runmat check` and runs to completion; state transfers into and out of the nonlinear step are tested.
- Described in: [Procedures](../language/reference/procedures.md), [Studies, analyses and state](../language/studies-analyses-and-state.md).

### Heat transfer

#### thermal.heat-generation.total-power

**Volumetric heat generation given as a total power (W) on a solid region.**

- Kind: load.
- Ships when: A heat-generation load accepts a power (W) and distributes it over the region's volume; the enclosure study checks and runs.

### Electromagnetics

#### electromagnetic.nonlinear

**Nonlinear magnetics with B–H curves.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electromagnetic nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### electromagnetic.transient

**Electromagnetic transient (eddy-current A–φ time stepping).**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electromagnetic transient` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Finite-element analysis](../index.md), [Procedures](../language/reference/procedures.md).

#### magnetic.harmonic.vector-potential

**Nonzero prescribed magnetic vector potential in harmonic and magnetostatic studies.**

- Kind: constraint.
- Ships when: A nonzero `magnetic.vector_potential` boundary value runs in magnetostatic and harmonic studies and matches a closed-form field.
- Described in: [Electromagnetic constraints](../constraints/electromagnetic.md).

#### magnetic.periodic

**Periodic and phase-shifted periodic magnetic boundaries for motor and transformer sectors.**

- Kind: constraint.
- Ships when: `magnetic.periodic` and `magnetic.phase_shifted_periodic` run in magnetostatic and harmonic studies; a sector model matches the full model.
- Described in: [Electromagnetic constraints](../constraints/electromagnetic.md).

### Acoustics

#### acoustic.nonlinear

**Nonlinear acoustics procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `acoustic nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### acoustic.periodic.rotational

**Rotational (cyclic) acoustic periodicity, and periodicity in acoustic transient studies.**

- Kind: constraint.
- Ships when: Rotated periodic pairs and transient periodic pairs run; a cyclic sector matches the full annulus modes.
- Described in: [Acoustic constraints](../constraints/acoustic.md).

#### acoustic.static

**Acoustic static (pressure-only Laplace) procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `acoustic static` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Scope: awaiting a product decision on whether this belongs in the final system.
- Described in: [Procedures](../language/reference/procedures.md).

#### output.transmission_loss

**Acoustic transmission loss output (`transmission_loss`) between inlet and outlet ports.**

- Kind: output.
- Ships when: `transmission_loss` is a cataloged output operation; the muffler study checks and runs with a receipt against a plane-wave reference.

### Fluid flow

#### fluid.harmonic

**Fluid harmonic (linearized flow about a base state) procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Scope: awaiting a product decision on whether this belongs in the final system.
- Described in: [Procedures](../language/reference/procedures.md).

#### fluid.non-newtonian

**Generalized-Newtonian fluid laws (`carreau`, `power_law`) in the page's unit form.**

- Kind: material.
- Ships when: `law = carreau` and `law = power_law` with consistency written in Pa*s^n check and run; the static-mixer study runs.

#### fluid.nonlinear

**Fluid nonlinear procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### fluid.periodic

**Periodic fluid boundary pairs (`fluid.periodic`).**

- Kind: constraint.
- Ships when: A study with `fluid.periodic` on matching face pairs checks and runs; a periodic channel matches the infinite-channel Poiseuille profile.
- Described in: [Fluid constraints](../constraints/fluid.md).

#### fluid.porous

**Darcy–Forchheimer porous regions in incompressible flow, in the page's material form.**

- Kind: material.
- Ships when: A `porous` material with `law = darcy_forchheimer` assigned to a cell region checks and runs in a fluid static step; the filter study runs with a receipt.
- In this release: run-time / unverified: runmat check admits porous regions.

#### fluid.species-transport

**Species transport in flow: species declarations, advection–diffusion, concentration constraints and fields.**

- Kind: procedure.
- Ships when: A `species` declaration with transport, a concentration constraint and a concentration field checks and runs; the static-mixer study runs.

#### lubrication.family

**Hydrodynamic lubrication: journal bearings with `rpm` speed, film pressure, Elrod cavitation and friction torque.**

- Kind: procedure.
- Ships when: The `rpm` unit, `lubrication.journal` and `lubrication.pressure` constraints, a steady lubrication step with Elrod cavitation, and `film_pressure`, `friction_torque` and `resultant` over film pressure check and run; the journal-bearing study matches a short-bearing reference.

#### output.oscillatory-shear-index

**Oscillatory shear index output (`oscillatory_shear_index`) over a cardiac cycle.**

- Kind: output.
- Ships when: `oscillatory_shear_index` is a cataloged output operation over a transient wall-shear history; the carotid study runs.

#### output.pressure_drop

**Pressure drop output (`pressure_drop`) between two face regions.**

- Kind: output.
- Ships when: `pressure_drop` is a cataloged output operation; cold-plate, cold-plate sweep and filter studies check and run.

#### output.wall-shear-stress

**Wall shear stress output over a wall region in transient flow.**

- Kind: output.
- Ships when: `wall_shear_stress` reduces over a wall face region in a transient fluid step; the carotid study runs.

### Thermomechanical

#### thermomechanical.harmonic

**Thermomechanical harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `thermomechanical harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### thermomechanical.transient

**Thermomechanical transient with two-way coupling.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `thermomechanical transient` through ANALYSIS_IDENTITIES; thermal stress publishes over the time history of a transient thermal analysis (FEA-COUPLING-THERMOELASTIC-002), and a canonical example (one braking stop) runs on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md), [Thermal stress](../physics/thermomechanical.md).

### Electrothermal

#### electrothermal.harmonic

**Electrothermal harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electrothermal harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### electrothermal.nonlinear

**Electrothermal nonlinear procedure (temperature-dependent two-way coupling).**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `electrothermal nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

### Conjugate heat transfer

#### cht.buoyancy

**Buoyancy-driven flow in conjugate heat transfer (Boussinesq with gravity).**

- Kind: load.
- Ships when: A Boussinesq buoyancy term with a gravity vector checks and runs in conjugate heat transfer; a natural-convection benchmark matches its reference.

#### conjugate-heat-transfer.harmonic

**Conjugate heat transfer harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `conjugate_heat_transfer harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### conjugate-heat-transfer.nonlinear

**Conjugate heat transfer with temperature-dependent properties.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `conjugate_heat_transfer nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

### Fluid–structure interaction

#### fluid-structure-interaction.harmonic

**Fluid–structure interaction harmonic procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid_structure_interaction harmonic` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### fluid-structure-interaction.nonlinear

**Fluid–structure interaction nonlinear procedure.**

- Kind: procedure.
- Ships when: `runmat check` and `runmat run` admit `fluid_structure_interaction nonlinear` through ANALYSIS_IDENTITIES; a canonical example runs to completion on the release CLI with a receipt compared to a reference.
- Described in: [Procedures](../language/reference/procedures.md).

#### fluid-structure-interaction.one-way-fluid-loads

**One-way fluid loads into a structural study (`pressure` and wall shear from a solved flow).**

- Kind: load.
- Ships when: A structural static or transient analysis takes the pressure and wall shear stress of an earlier fluid analysis as a load on the wetted faces (`after analysis.<flow>`); the mapped total force and moment equal the fluid resultant, and a study checks and runs.
- Described in: [Fluid–structure](../physics/fluid-structure-interaction.md).

#### fluid-structure-interaction.solid-stress

**Structural stress on the solid of an FSI study (`von_mises_stress`, `stress`).**

- Kind: output.
- Ships when: A fluid_structure_interaction steady or transient analysis publishes element stress and von Mises stress on the solid region from the converged displacement; outputs and snapshot views read them, and a cantilever paddle in channel flow matches the beam root stress of its fluid force resultant.
- Described in: [Fluid–structure](../physics/fluid-structure-interaction.md).

### The `.fea` language

#### language.include.parameterized

**Including a parameterized model with its inputs bound by name.**

- Kind: language.
- Ships when: `include model.<m>(<name> = <value>, …)` checks, binds typed region, material and quantity inputs, and a study using it runs.
- Described in: [Declarations](../language/reference/declarations.md).

#### language.keyword-names

**Keywords allowed as declared names (`input load : force`).**

- Kind: language.
- Ships when: Declared input, let and output names may be keywords; `input load : force = 1.5 kN` checks clean and resolves as `input.load`.
- Described in: [Grammar](../language/reference/grammar.md), [Types, units and expressions](../language/types-units-and-expressions.md).

#### language.plastic-law-identifier

**Plastic law written as an identifier (`law = j2_isotropic_linear_hardening`), like fluid and porous laws.**

- Kind: language.
- Ships when: `plastic { law = j2_isotropic_linear_hardening }` checks; one form for every material `law` member.
- Described in: [Declarations](../language/reference/declarations.md).

#### language.unit.siemens

**The siemens (`S`, `mS`, `kS`) as a unit symbol, so conductivity is written `S/m`.**

- Kind: language.
- Ships when: `runmat-types` accepts `S` as electrical conductance; `conductivity = 5.8e7 S/m` checks and equals `5.8e7 A/V/m`.
- Described in: [Types, units and expressions](../language/types-units-and-expressions.md).

#### language.vector.trailing-unit

**Vectors with one trailing unit in orientation and frame members (`origin = [0, 0, 0] m`).**

- Kind: language.
- Ships when: Every member that expects a vector quantity accepts both `[0 m, 0 m, 0 m]` and `[0, 0, 0] m` through one normalization point; orientation and frame members check clean.
- Described in: [Types, units and expressions](../language/types-units-and-expressions.md).

#### signal.harmonic-nonconstant-value

**Harmonic signals whose value is an expression rather than a constant scalar or vector.**

- Kind: load.
- Ships when: A harmonic signal accepts the page's non-constant value form; the speaker study checks and runs.

#### signal.tabulated-file

**Signals sampled from a CSV file (`samples = file(...)`) with periodic spline interpolation.**

- Kind: language.
- Ships when: `samples = file("<csv>")` and `interpolation = periodic_spline` check and run; the carotid study's heartbeat signal resolves.

### Geometry

#### geometry.select.predicate

**Geometric-predicate region selectors (planar faces by normal, faces within a box).**

- Kind: geometry.
- Ships when: A region selected by a bounded geometric predicate checks with cardinality evidence, resolves to canonical ids, and stays stable across re-import of the same CAD revision.
- Described in: [Geometry, models and assignments](../language/geometry-models-and-assignments.md), [Declarations](../language/reference/declarations.md).

### Results

#### output.coefficient-of-variation

**Coefficient of variation of a field over a region (`coefficient_of_variation`).**

- Kind: output.
- Ships when: `coefficient_of_variation` is a cataloged output operation; the static-mixer study runs.

#### output.interpolate-line-and-surface-meshes

**`interpolate(field, at = point)` on beam and shell meshes, where the solver mesh has no solid elements.**

- Kind: output.
- Ships when: An explicit spatial observation on a line or surface solver mesh resolves to the containing beam or shell element: `interpolate(analysis.static.field.node.displacement, at = [2000 mm, 0 mm, 3000 mm])` on portal-frame.fea runs and publishes the girder midspan displacement.
- Described in: [Outputs, fields and observations](../language/outputs-fields-and-observations.md).

#### output.mode-location

**Study outputs at the `mode` location, such as `analysis.modes.field.mode.modal_frequency`.**

- Kind: output.
- Ships when: `output frequencies = analysis.modes.field.mode.modal_frequency` checks and runs on tuning-fork-modal and publishes one value per mode.
- Described in: [Modal analysis](../guides/modal-analysis.md), [Outputs, fields and observations](../language/outputs-fields-and-observations.md).

#### output.point-sample-assessment

**Assessments on a value sampled at a point, such as `interpolate(field, at = point)`.**

- Kind: output.
- Ships when: `assessment stress_allowable { value = output.top_stress; maximum = 167 MPa; evidence = authored_samples }` with `output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])` checks and runs on cantilever-static.fea and publishes a verdict.
- Described in: [Linear static](../guides/linear-static.md).

#### output.time-mean

**Time-mean of a field over a transient step (`mean` over time).**

- Kind: output.
- Ships when: A `mean` overload over the time domain of a transient field checks and runs.

## Known defects

Defects found in capabilities the current release includes. Each is engineering work with the evidence that shows it; it is removed from the ledger when the witness passes.

#### defect.buckling-eigensolver-convergence

**Buckling eigensolve reports non-convergence with residuals at round-off.**

- Affects: Structural mechanics.
- Fixed when: The buckling example (cantilever.step, 100 MPa axial traction, preload then buckling with two modes, Tet10 at 150 mm) completes on every mesh the mesher produces, and the eigensolver's convergence flag agrees with its residual test.

## Validation by family

- [Structural mechanics](./structural.md)
- [Heat transfer](./thermal.md)
- [Electromagnetics](./electromagnetic.md)
- [Acoustics](./acoustic.md)
- [Fluid flow](./fluid.md)
- [Thermomechanical](./thermomechanical.md)
- [Electrothermal](./electrothermal.md)
- [Conjugate heat transfer](./conjugate-heat-transfer.md)
- [Fluid–structure interaction](./fluid-structure-interaction.md)

The [validation methodology](./methodology.md) explains how cases, references and tests are read.
