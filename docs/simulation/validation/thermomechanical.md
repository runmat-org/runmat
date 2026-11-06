---
description: How RunMat checks its thermomechanical results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [thermomechanical validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Thermomechanical validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Thermomechanical static | yes | [Thermal stress in a clamped bar](#thermal-stress-in-a-clamped-bar) |
| Thermomechanical nonlinear | yes | — |
| Thermomechanical transient with two-way coupling | [not yet in the current release](pending:thermomechanical.transient) | — |
| Thermomechanical harmonic procedure | [not yet in the current release](pending:thermomechanical.harmonic) | — |

## Reference cases

### Thermal stress in a clamped bar

[thermal-stress-static.fea](../examples/thermal-stress-static.fea): the same bar clamped at both ends
and held at 373.15 K (80 K above its reference) has the axial stress $E\alpha\Delta T$ = 192 MPa away
from the clamps. The volume-mean von Mises stress is 195.9 MPa; the clamps also restrain lateral
expansion, which raises the stress near the ends.

- References: Axial stress EαΔT for a fully restrained bar.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-COUPLING-INTERFACE-001 | Typed physical interfaces and conservative transfers | 3 | 0 | 0 | 0 | 12 |
| FEA-COUPLING-THERMOELASTIC-001 | Solved-temperature thermoelastic transitions and closure | 3 | 0 | 2 | 0 | 4 |
| FEA-COUPLING-CHAIN-001 | Chained thermal-structural studies | 2 | 0 | 0 | 0 | 24 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
| FEA-NONLINEAR-GOVERNANCE-001 | Converged nonlinear states, supported laws and independent qualification | 5 | 0 | 2 | 0 | 48 |
| FEA-STRUCTURAL-PUBLIC-001 | Structural capabilities the page claims | 10 | 0 | 0 | 0 | 100 |
| FEA-COUPLING-THERMOELASTIC-002 | Thermal stress over a time history (thermomechanical transient) | 3 | — | — | — | — |
