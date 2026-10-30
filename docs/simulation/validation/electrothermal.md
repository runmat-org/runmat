---
description: How RunMat checks its electrothermal results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [electrothermal validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Electrothermal validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Electrothermal static | yes | [Complete electromagnetic and electrothermal studies](./electromagnetic.md#complete-electromagnetic-and-electrothermal-studies), [Two-way Joule heating of a bar](#two-way-joule-heating-of-a-bar) |
| Electrothermal transient | yes | — |
| Electrothermal harmonic procedure | [not yet in the current release](pending:electrothermal.harmonic) | — |
| Electrothermal nonlinear procedure (temperature-dependent two-way coupling) | [not yet in the current release](pending:electrothermal.nonlinear) | — |

## Reference cases

### Two-way Joule heating of a bar

A bar with resistivity $\rho(T)=\rho_0(1+\alpha(T-T_0))$ carries a fixed current, with its ends held
at $T_0$ and its sides insulated. The temperature rise $\theta$ solves
$\theta''+\beta(1+\alpha\theta)=0$ with $\beta=J^2\rho_0/k$. The closed form is

$$
\theta(x)=\frac{1}{\alpha}\left(\frac{\cos\big(m(x-L/2)\big)}{\cos(mL/2)}-1\right),\qquad m^2=\alpha\beta .
$$

With $\sigma_0$ = 3 S/m, $\alpha$ = 0.004 /K, $k$ = 1 W/m/K, $I$ = 34.64 A on a unit cube (Tet10):

| cells | sweeps | max $\lvert T-T_{exact}\rvert$ (K) | peak rise (K) | Joule power (W) | $\int I^2\rho(T)/A\,dx$ (W) |
| --- | --- | --- | --- | --- | --- |
| 2 | 8 | 2.21 | 59.95 | 461.47 | 463.52 |
| 4 | 9 | 0.575 | 59.95 | 462.88 | 463.52 |
| 8 | 9 | 0.150 | 59.95 | 463.35 | 463.52 |

The coupling iterates until no element temperature changes by more than $10^{-6}$ of the rise.
Each element takes its conductivity at its mean temperature, so the temperature error falls
fourfold per refinement.

- References: Closed-form bar with resistivity ρ(T) = ρ₀(1 + α(T − T₀)).

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-COUPLING-INTERFACE-001 | Typed physical interfaces and conservative transfers | 3 | 0 | 0 | 0 | 12 |
| FEA-COUPLING-ELECTROTHERMAL-001 | Original Joule-heat transfer and electrical/thermal iteration | 3 | 0 | 4 | 0 | 2 |
| FEA-COUPLING-ELECTROTHERMAL-002 | Two-way and transient Joule and AC-loss heating | 4 | 0 | 1 | 0 | 23 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
