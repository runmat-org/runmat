---
description: How RunMat checks its fluid–structure interaction results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [fluid–structure interaction validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Fluid–structure interaction validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Fluid–structure interaction static | yes | — |
| Fluid–structure interaction transient | yes | — |
| Fluid–structure interaction harmonic procedure | [not yet in the current release](pending:fluid-structure-interaction.harmonic) | — |
| Fluid–structure interaction nonlinear procedure | [not yet in the current release](pending:fluid-structure-interaction.nonlinear) | — |
| Structural stress on the solid of an FSI study (`von_mises_stress`, `stress`) | [not yet in the current release](pending:fluid-structure-interaction.solid-stress) | — |

## Reference cases

No fluid–structure interaction reference case is published on this page yet; the requirement tests below cover the formulation.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-COUPLING-INTERFACE-001 | Typed physical interfaces and conservative transfers | 3 | 0 | 0 | 0 | 12 |
| FEA-COUPLING-FSI-001 | Dynamics/fluid interaction with stable conservative ALE execution | 4 | 0 | 0 | 0 | 9 |
| FEA-FLUID-VERIFICATION-001 | Published flow verification suite | 1 | 0 | 0 | 0 | 8 |
| FEA-COUPLING-FSI-002 | Fluid loads on structures and two-way FSI | 4 | 0 | 0 | 0 | 16 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
