---
description: How RunMat checks its fluid flow results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [fluid flow validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Fluid flow validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Fluid static | yes | — |
| Fluid transient | yes | — |
| Fluid harmonic (linearized flow about a base state) procedure | [not yet in the current release](pending:fluid.harmonic) | — |
| Fluid nonlinear procedure | [not yet in the current release](pending:fluid.nonlinear) | — |
| Pressure drop output (`pressure_drop`) between two face regions | [not yet in the current release](pending:output.pressure_drop) | — |
| Darcy–Forchheimer porous regions in incompressible flow, in the page's material form | [not yet in the current release](pending:fluid.porous) | — |
| Generalized-Newtonian fluid laws (`carreau`, `power_law`) in the page's unit form | [not yet in the current release](pending:fluid.non-newtonian) | — |
| Wall shear stress output over a wall region in transient flow | [not yet in the current release](pending:output.wall-shear-stress) | — |
| Oscillatory shear index output (`oscillatory_shear_index`) over a cardiac cycle | [not yet in the current release](pending:output.oscillatory-shear-index) | — |
| Species transport in flow: species declarations, advection–diffusion, concentration constraints and fields | [not yet in the current release](pending:fluid.species-transport) | — |
| Hydrodynamic lubrication: journal bearings with `rpm` speed, film pressure, Elrod cavitation and friction torque | [not yet in the current release](pending:lubrication.family) | — |

## Reference cases

No fluid flow reference case is published on this page yet; the requirement tests below cover the formulation.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-FLUID-TOPOLOGY-001 | Canonical conservative finite-volume projection | 4 | 0 | 4 | 0 | 28 |
| FEA-FLUID-EQUATIONS-001 | Original incompressible laminar mass and momentum equations | 3 | 0 | 4 | 0 | 20 |
| FEA-FLUID-INPUTS-001 | Exact typed fluid boundaries, sources and initial conditions | 4 | 0 | 4 | 0 | 24 |
| FEA-FLUID-RECOVERY-001 | Original fluid fields and physical conservation recovery | 2 | 0 | 2 | 0 | 14 |
| FEA-FLUID-ACCURACY-001 | Independent flow benchmark and convergence acceptance | 3 | 0 | 2 | 0 | 6 |
| FEA-FLUID-EXECUTION-001 | Governed fluid execution and semantic publication | 3 | 0 | 1 | 1 | 36 |
| FEA-FLUID-SOLVERS-001 | Robust, scalable and accurate laminar flow solvers | 5 | 0 | 0 | 0 | 36 |
| FEA-FLUID-BOUNDARY-002 | Complete fluid boundary and excitation set | 5 | 0 | 0 | 0 | 36 |
| FEA-FLUID-PHYSICS-001 | Free surface, rotating frames, lubrication, electrokinetics, particles, compressibility and reduced dimensions | 7 | 0 | 0 | 0 | 26 |
| FEA-FLUID-OUTPUTS-001 | Flow outputs engineers ask for | 5 | 0 | 0 | 0 | 52 |
| FEA-FLUID-VERIFICATION-001 | Published flow verification suite | 1 | 0 | 0 | 0 | 8 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
| FEA-FLUID-TRANSIENT-001 | Physical transient integration and original cold chronology | 4 | 0 | 4 | 0 | 9 |
| FEA-FLUID-MATERIALS-001 | Rheology, porous media and species transport | 4 | 0 | 0 | 0 | 16 |
