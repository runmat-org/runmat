---
description: How RunMat checks its heat transfer results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [heat transfer validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Heat transfer validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Thermal static | yes | [Radiating bar](#radiating-bar) |
| Thermal transient | yes | — |

## Reference cases

### Radiating bar

[radiating-bar.fea](../examples/radiating-bar.fea): a 1 m steel bar ($k$ = 45 W/m/K) takes
1000 W/m² on one end and radiates from the other ($\varepsilon$ = 0.8, 300 K surroundings). The
radiating end satisfies $q=\varepsilon\sigma_{SB}(T^4-T_\infty^4)$, and the heated end is $qL/k$
hotter. RunMat gives 416.6790 K and 438.9013 K; the closed form gives 416.6790 K and 438.9013 K
(agreement to 1e-12). The published energy balance closes: 10 W applied, 10 W radiated. The same
check runs as a test on Tet4 and Tet10 meshes
(`radiating_bar_matches_the_gray_surface_closed_form_and_balances_power`).

- References: Gray-surface radiation q = εσ(T⁴ − T∞⁴) with conduction qL/k.
- Held in CI by: `radiating_bar_matches_the_gray_surface_closed_form_and_balances_power`.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-THERMAL-DOMAIN-001 | Independent thermal domain and regional heat-equation assembly | 3 | 0 | 5 | 0 | 7 |
| FEA-THERMAL-INPUT-001 | Exact thermal boundaries and shared excitation realization | 3 | 0 | 4 | 0 | 8 |
| FEA-THERMAL-SOLVE-001 | Distinct steady and transient solves with nonlinear closure | 3 | 0 | 4 | 0 | 4 |
| FEA-THERMAL-FIELDS-001 | Physical thermal recovery and energy accounting | 3 | 0 | 4 | 0 | 8 |
| FEA-THERMAL-ACCURACY-001 | Independent thermal references and refinement qualification | 4 | 0 | 2 | 1 | 11 |
| FEA-THERMAL-EXECUTION-001 | Governed thermal studies, persistence and execution evidence | 4 | 0 | 4 | 0 | 84 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
| FEA-THERMAL-PUBLIC-001 | Thermal studies the pages run | 5 | 0 | 3 | 0 | 21 |
