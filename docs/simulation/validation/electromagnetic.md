# Electromagnetic and electrothermal validation

These checks compare RunMat's electromagnetic and electrothermal solutions with closed-form
references. Each reference is independent of RunMat's element matrices. The kernel checks run in
the `runmat-analysis-fea` test suite; the study checks run complete `.fea` studies with
`runmat run`.

## Skin effect in a driven conductor (harmonic fields)

A conducting strip of thickness $t$ under an air gap $g$, returned through a grounded plane, with
uniform fields across its width $W$ and along its length $L$ (a parallel-plate line), has the
impedance

$$
Z = \frac{L}{W}\left[\frac{k\coth(kt)}{\sigma} + i\omega\mu_0 g\right],\qquad k=\frac{1+i}{\delta},
\quad \delta=\sqrt{\frac{2}{\omega\mu_0\sigma}}.
$$

The strip is driven by a 2 A terminal current ($W$ = 1 mm, $L$ = 2 mm, $t$ = 1 mm, $g$ = 0.5 mm,
$\sigma$ = 1 MS/m), meshed with $n$ cells through the thickness and along the current. RunMat
reports the terminal voltage, so $Z = U/I$.

| $t/\delta$ | cells | RunMat $Z$ (mΩ) | exact $Z$ (mΩ) | relative error |
| --- | --- | --- | --- | --- |
| 0.5 | 8 | 2.0116 + 0.8320i | 2.0111 + 0.8328i | 4.2e-4 |
| 0.5 | 16 | 2.0112 + 0.8326i | | 1.0e-4 |
| 0.5 | 32 | 2.0111 + 0.8328i | | 2.6e-5 |
| 2 | 8 | 3.8588 + 11.945i | 3.7956 + 12.012i | 7.3e-3 |
| 2 | 16 | 3.8116 + 11.996i | | 1.8e-3 |
| 2 | 32 | 3.7996 + 12.008i | | 4.5e-4 |
| 4 | 8 | 8.3979 + 39.487i | 8.0045 + 39.994i | 1.6e-2 |
| 4 | 16 | 8.1110 + 39.874i | | 3.9e-3 |
| 4 | 32 | 8.0316 + 39.964i | | 9.8e-4 |

The error falls fourfold per refinement (second order in the impedance). On every solve the Joule
loss equals the real power delivered, $\tfrac12\mathrm{Re}(U\bar I)$, to $10^{-8}$, and the complex
Poynting balance and weak charge continuity hold to $10^{-8}$.

A voltage-driven strip at 1 Hz reproduces the DC resistance $L/(\sigma W t)$ and the gap inductance.
A gap-fed bar inside air, whose terminals do not touch a grounded boundary, closes its terminal
current through the air. The delivered power, $1.8750\times10^{-3}$ W, equals
$\tfrac12 I^2R_{dc}$ = $1.875\times10^{-3}$ W, and the Joule loss matches it to $6\times10^{-5}$.

## Magnetostatic field of a current-carrying bar

A square conductor carrying uniform current density, grounded on every exterior face, is compared
with an independent full-volume solution. The vector potential, flux density and magnetic energy
errors fall with refinement (cells 1, 2, 4: flux-density error 0.166, 0.116, 0.064; energy error
0.80, 0.39, 0.12).

## Electrostatics and steady current flow

A charged dielectric slab (Tet4) has potential, field and energy errors that fall by four, two and
four per refinement: potential 4.6e-2, 1.1e-2, 2.9e-3; field 0.29, 0.14, 0.072; energy 0.25, 0.063,
0.016. Steady current flow uses the same operator with conductivity in place of permittivity. The
terminal power, $\sum V_iI_i$, equals the Joule integral on every solve.

## Two-way Joule heating

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

## Study checks with `runmat run`

These are complete `.fea` studies on a 1000 x 100 x 100 mm bar (quadratic tetrahedra, 50 mm), read
from their published authored outputs.

| Study | Quantity | RunMat | Reference | Relative difference |
| --- | --- | --- | --- | --- |
| Steady current flow, copper ($\sigma$ = 59.6 MS/m), 1 mV across the bar | peak current density | 59 600.000 A/m² | $\sigma V/L$ = 59 600 A/m² | 1e-13 |
| | Joule power (integral of the loss density) | 0.596000 W | $V^2/R$ = 0.596 W | 1e-14 |
| Electrothermal steady, 30 kA, ends at 293.15 K, $\alpha$ = 0.00393 /K, $k$ = 401 W/m/K | peak temperature | 348.757 K | 348.763 K (closed form above) | 1.0e-4 of the rise |
| | Joule power | 1728.69 W | $I^2\int\rho(T)\,dx/A$ = 1728.72 W | 1.6e-5 |
| | terminal voltage | 57.6255 mV | $IR(T)$ = 57.6239 mV | 2.7e-5 |
| | $V\cdot I$ against Joule power | 1728.77 W vs 1728.69 W | equal | 4.7e-5 |

The steady-current row and the two-way Joule-heating bar are also run as tests on every change
(`copper_bar_carries_sigma_v_over_l_and_dissipates_v_squared_over_r` and
`two_way_joule_heating_matches_the_closed_form_bar`).

The electrothermal study publishes two cases, `operating_point` (temperature) and
`operating_point/conduction` (potential, current density and loss at the converged temperatures).

See [electromagnetics](../physics/electromagnetic), [electrothermal analysis](../physics/electrothermal)
and [validation methodology](../validation).
