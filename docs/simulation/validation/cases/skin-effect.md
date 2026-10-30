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
