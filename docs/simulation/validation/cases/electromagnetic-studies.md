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

