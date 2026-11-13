---
description: Which analysis types of each RunMat physics family have a reference check, how many reference checks and automated tests each family has, and when they last ran.
keywords: [validation status, reference checks, automated tests, verification, FEA accuracy]
doc_type: release-status
generated: node scripts/docs/generate-pages.mjs
---
# Status

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Validation for RunMat 0.7.0, by physics family. Each family page lists its reference checks with the problem, reference, tolerance and result.

Reference checks are published benchmark problems compared against a known answer within a stated tolerance. Automated tests are the unit, regression and qualification tests in the RunMat source for that family. Another <!-- tests -->2,656<!-- /tests --> tests cover meshing, the sparse solvers and study planning, which every family uses. Last run is the latest evidence-report run or receipt date among the family's reference checks.

| Family | Analysis types with a reference check | Reference checks | Automated tests | Last run |
| --- | --- | ---: | ---: | --- |
| [Structural mechanics](./structural.md) | Static, Modal, Harmonic, Transient, Buckling, Response spectrum, Nonlinear (plasticity) | 13 | 651 | 2026-11-08 |
| [Heat transfer](./thermal.md) | Steady, Transient | 7 | 317 | 2026-11-08 |
| [Electromagnetics](./electromagnetic.md) | Electrostatics and steady current, Magnetostatics, Harmonic, Transmission-line parameters | 12 | 191 | 2026-11-08 |
| [Acoustics](./acoustic.md) | Modal, Harmonic | 13 | 120 | 2026-11-08 |
| [Fluid flow](./fluid.md) | Steady, Transient | 7 | 159 | 2026-11-08 |
| [Thermomechanical](./thermomechanical.md) | Static | 2 | 16 | 2026-11-08 |
| [Electrothermal](./electrothermal.md) | Steady | 2 | 8 | 2026-11-08 |
| [Conjugate heat transfer](./conjugate-heat-transfer.md) | None yet; planned: Heated pipe wall in fully developed laminar flow against the Nusselt numbers 3.66 (constant wall temperature) and 4.364 (constant heat flux) | 0 | 9 | none |
| [Fluid–structure interaction](./fluid-structure-interaction.md) | None yet; planned: Turek–Hron FSI1, FSI2 and FSI3 flap tip displacement and forces | 0 | 7 | none |

Which procedures each family runs in this release is in the [procedure reference](../language/reference/procedures.md#families-and-procedures).
