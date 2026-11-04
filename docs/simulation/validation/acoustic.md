---
description: How RunMat checks its acoustics results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [acoustics validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Acoustics validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Acoustic harmonic | yes | — |
| Acoustic modal | yes | — |
| Acoustic transient | yes | — |
| Acoustic static (pressure-only Laplace) procedure | [not yet in the current release](pending:acoustic.static) | — |
| Nonlinear acoustics procedure | [not yet in the current release](pending:acoustic.nonlinear) | — |
| Acoustic transmission loss output (`transmission_loss`) between inlet and outlet ports | [not yet in the current release](pending:output.transmission_loss) | — |

## Reference cases

No acoustics reference case is published on this page yet; the requirement tests below cover the formulation.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-ACOUSTIC-OPERATOR-001 | Canonical material-region Helmholtz operator | 3 | 0 | 3 | 0 | 5 |
| FEA-ACOUSTIC-BOUNDARY-001 | Exact typed acoustic boundaries and complex constraints | 4 | 0 | 5 | 2 | 1 |
| FEA-ACOUSTIC-SOURCE-001 | Shared complex source spectra with conserved realization | 3 | 0 | 4 | 0 | 2 |
| FEA-ACOUSTIC-NUMERICS-001 | Independent acoustic fields, resonance and physical balances | 5 | 0 | 5 | 0 | 5 |
| FEA-ACOUSTIC-ACCURACY-001 | Mandatory acoustic accuracy gates publication | 2 | 0 | 0 | 2 | 2 |
| FEA-ACOUSTIC-EXECUTION-001 | Governed acoustic frequency execution and original-field publication | 5 | 0 | 3 | 0 | 42 |
| FEA-ACOUSTIC-OUTPUTS-001 | Sound pressure level, transmission loss, intensity and power | 5 | 0 | 1 | 0 | 19 |
| FEA-ACOUSTIC-BOUNDARY-002 | Radiation, frequency-dependent impedance and air damping | 4 | 0 | 3 | 0 | 17 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
| FEA-ACOUSTIC-MODAL-001 | Acoustic cavity modal analysis | 3 | 0 | 3 | 2 | 7 |
| FEA-ACOUSTIC-TRANSIENT-001 | Acoustic transient analysis | 3 | 0 | 0 | 1 | 11 |
