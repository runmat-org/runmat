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
