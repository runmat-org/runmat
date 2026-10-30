---
description: How RunMat checks its conjugate heat transfer results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [conjugate heat transfer validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Conjugate heat transfer validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Conjugate heat transfer static | yes | — |
| Conjugate heat transfer transient | yes | — |
| Conjugate heat transfer harmonic procedure | [not yet in the current release](pending:conjugate-heat-transfer.harmonic) | — |
| Conjugate heat transfer with temperature-dependent properties | [not yet in the current release](pending:conjugate-heat-transfer.nonlinear) | — |

## Reference cases

No conjugate heat transfer reference case is published on this page yet; the requirement tests below cover the formulation.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-COUPLING-INTERFACE-001 | Typed physical interfaces and conservative transfers | 3 | 0 | 0 | 0 | 12 |
| FEA-COUPLING-CHT-001 | Thermal/fluid conjugate heat transfer | 3 | 0 | 0 | 0 | 3 |
| FEA-FLUID-BOUNDARY-002 | Complete fluid boundary and excitation set | 5 | 0 | 0 | 0 | 36 |
| FEA-FLUID-HEAT-001 | Heat in the fluid: forced and natural convection, CHT and low-Mach flow | 6 | 0 | 0 | 0 | 24 |
| FEA-FLUID-OUTPUTS-001 | Flow outputs engineers ask for | 5 | 0 | 0 | 0 | 52 |
| FEA-FLUID-VERIFICATION-001 | Published flow verification suite | 1 | 0 | 0 | 0 | 8 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
