# Solid harmonic qualification: independent one-DOF reference

This case checks physical Tet4 and Tet10 harmonic operators, their modal projection,
complex displacement phase, and mean power balance. It is a numerical verification
case for Stage 4. The complementary [solid rod qualification](solid-rod-harmonic.md)
adds Tet4 continuum face-load, modal-truncation and cold typed-object evidence.
Full formulation promotion also requires continuum convergence,
governed source realization, recovery, execution/restart, and release evidence.

## Geometry, material, and constraints

The domain is the unit tetrahedron with corners `(0,0,0)`, `(1,0,0)`, `(0,1,0)`,
and `(0,0,1)` meters. Tet10 includes its six midside nodes. The curved cases move
the midside node on the first edge by `0.1 m` or `0.2 m` in global X. Canonical
mesh validation and curved-element integration govern the actual assembly.

The material has `E = 100 Pa`, `nu = 0.25`, and `rho = 1 kg/m^3`. Every structural
translation is held at zero except global X at the origin corner. This deliberately
isolates one physical discrete degree of freedom. The load is an explicit nodal
complex force vector, with peak phasor `F = s (1 + 0.4 i) N` and
`s in {1e-12, 1, 1e3}`. The force-vector input bypasses authored region selection;
that authoring path is a separate qualification requirement.

Two damping cases are checked: undamped, and Rayleigh damping with
`alpha = 0.8 s^-1` and `beta = 0.002 s`. Frequencies use
`omega / sqrt(k/m) in {0.25, 0.75, 1, 1.5, 3}`. Exact undamped resonance is
excluded because a bounded steady response does not exist there. Damped resonance
is included. The matrix contains 108 load/frequency cases, each checked through
both the full sparse direct solver and the retained one-mode projection.

## Independent reference

For the sole free corner degree of freedom, direct shape-function integration gives:

| Element | Stiffness `k` (N/m) | Mass `m` (kg) |
| --- | --- | --- |
| Tet4 | `100/3` | `1/60` |
| Straight Tet10 | `20` | `1/420` |
| Curved Tet10 with warp `w` | Independent 24-point integration | `1/420 + w/315` |

These values are not extracted from RunMat's assembled matrices. For Tet4,
`grad(N_1)=(-1,-1,-1)`, `lambda=mu=40 Pa`, and volume is `1/6 m^3`.
For Tet10, `N_1=2 L_1^2-L_1`. The warped geometry has
`det(J)=1+4 w (L_1-L_2)`, which yields the exact mass above. The existing independent
24-point tensor rule integrates the curved stiffness; production uses its own
governed convergence pair.

The forced-oscillator reference follows [MIT 8.03SC, *The Physics of Waves*,
chapter 2, equations 2.14–2.24](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/782069da3820fc514c10c26ae0c15b01_MIT8_03SCF16_Text_Ch2.pdf).
Conjugating that source's `exp(-i omega t)` convention gives RunMat's
`exp(+i omega t)` dynamic stiffness:

$$
\hat u=\frac{\hat F}{k-\omega^2m+i\omega c},\qquad c=\alpha m+\beta k.
$$

The time-averaged input power and viscous loss independently satisfy

$$
\bar P=\tfrac12\operatorname{Re}(\hat F\overline{i\omega\hat u})
=\tfrac12 c\omega^2|\hat u|^2.
$$

The reference checks amplitude and phase on both sides of resonance. A small
complex-equilibrium residual alone cannot establish these physical values.

## Verification and gate boundary

Run the normal library test:

```sh
cargo test --locked -p runmat-analysis-fea --lib physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references
```

The executable assertions require relative complex-displacement error below
`2e-7` for both solvers, positive mean input power for damped cases, power closure
within `2e-7` of the force/velocity power scale, zero constrained displacement,
full physical equilibrium after reactions within `1e-9`, and each solver's relative
residual within `1e-10`.

This case verifies a constrained discrete solid operator and an exactly complete
one-mode basis. It does not provide a mode-truncation study, unconstrained continuum
resonance reference, radiation or nonproportional damping validation, authored
complex-source equivalence, spatial convergence, result-storage/restart parity,
accelerator evidence, or a qualification gate decision. Those remain explicit
requirements of the formulation's qualification inventory.
