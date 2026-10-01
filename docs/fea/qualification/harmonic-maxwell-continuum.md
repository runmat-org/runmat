# Harmonic Maxwell continuum qualification

## Physical authority

Peak phasors use Re(E exp(+i omega t)). The electric-field formulation is

    curl(mu^-1 curl E) - omega^2 epsilon E + i omega sigma E = -i omega J_impressed.

Faraday recovery gives B=i curl(E)/omega; H=B/mu, D=epsilon E and
J_conduction=sigma E. These signs follow the physical Maxwell equations documented
by [MFEM](https://mfem.org/electromagnetics/). The physical curl-conforming weak
form is documented by [Apollo](https://aurora-multiphysics.github.io/apollo/frequency_domain/ComplexMaxwell.html).
The real sparse block has [R,-I;I,R], not a positive-definite certificate.

The implementation reuses canonical persistent oriented edges and integrated
Piola basis values/curls, exact per-element material ownership and the named
15-point volume sampling authority. Sources are original complex physical current
samples, not callbacks, clipped magnitudes, ordinal deposition or synthetic phases.
Full-exterior PEC removes tangential electric circulations, not scalar potentials
or a magnetostatic vector-potential gauge. Recovery remains at original volume
samples rather than averages of unrelated incident-edge vectors.

## Independent manufactured witness

The unit cube mode E=(0,sin(pi x)sin(pi z),0) times a complex phase satisfies PEC:
it vanishes tangentially on x/z faces and is normal on y faces. Its divergence is
zero and curl-curl(E)=2 pi^2 E. With omega=mu=epsilon=1, sigma=0.3,

    J_impressed=(-0.3+i(2 pi^2-1)) E.

The test source is evaluated directly from this continuum expression, never from
RunMat's assembled matrix. The analytic B follows the derivatives of the mode.
Integral |E|^2=1/4 for a unit-magnitude phase, so peak ohmic loss is0.3/8 W.
Refinement compares both complex E and B and physical loss; phase rotation and
sign reversal must rotate both fields without changing power. Affine Tet10 is
the same lowest-order six-edge field approximation as Tet4, not quadratic Maxwell.

Source power is -1/2 integral E dot conjugate(J_impressed). With closed PEC,
its real part equals ohmic loss and its imaginary part equals
2 omega (magnetic energy-electric energy). This complex balance and physical
linear residual do not substitute for spatial convergence evidence.

## Scope and open gates

The isolated numerical suite passes six groups: transverse analytic E/B/loss
refinement, charge-bearing longitudinal gradient refinement, complex phase/sign
and affine geometry order, persistent heterogeneous material ownership and exact
sparse-plan binding, invalid/zero/lossless/unrepresentable inputs, and recovered
complex B normal continuity with per-face PEC flux and global closure. Strict
FEA library/test lint passes. The shared edge suite passes11/11 and magnetic
regression67/67. These are scoped numerical facts, not a full CI pass.

For1/2/4/6 cells per cube side, transverse E L2 errors are
0.45879/0.29614/0.15726/0.10596 and B errors are
1.93640/1.26366/0.67661/0.45603. Four cells initially missed the unchanged B
target0.60; six cells reaches it, with relative ohmic-loss error below0.25.
The longitudinal E relative error decreases0.82404→0.48839 for2→4 cells.
The source is not projected to a divergence-free surrogate. Floating-point
representability admission rejects finite nonzero forcing whose squared norm
vanishes, and rejects vanished displacement coefficients rather than silently
taking a static limit. Exact zero forcing remains valid.

This work is a numerical continuum implementation/verification slice,
not authored/native/restart or production qualification. Initial uniqueness
admission requires positive frequency and strictly positive conductivity in every
element. A converged residual alone cannot establish uniqueness at lossless cavity
resonance; a genuine spectral/nonresonance authority remains mandatory for lossless
envelopes. Affine Tet4/Tet10 geometry only; curved geometry needs governed quadrature.

Remaining full-plan work includes distinct authored E/PEC intent and shared
excitation/material/constraint compilation, replacement of reduced-graph dispatch,
incident/ports/open boundaries, dispersive/complex material laws, lossless cavities
and waveguides, skin-effect/interfaces/frequency references, native immutable
task/resource/accuracy/state/publication execution, numerical restart, and release.
The older harmonic graph path supplies no continuum qualification credit.
