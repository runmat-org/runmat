# Normally incoming acoustic pressure

This bounded CPU FP64 canonical Tet4/Tet10 source uses the shared
typed `AcousticIncidentPressure` action and scalar pressure signal. DSL spelling
is `incident_pressure`; canonical serialized kind is `acoustic_incident_pressure`.
It supplies incoming pressure, unlike `acoustic_pressure`, which fixes total
pressure. No acoustic-private amplitude, phase, generator or field-recovery path
is introduced. This is not Stage 8 completion or qualification.

## Physical contract

With peak phasors and exp(+i omega t), Y=1/(rho*c) is resolved from the unique
exterior owning fluid. A normally incoming wave has outward velocity -Y p_inc;
an outgoing wave has +Y p_out. Thus total v_n=Y(p-2p_inc), and weak Helmholtz
assembly adds outgoing B and b=2i omega integral(N_i Y p_inc). This derivation is
consistent with the optional incident right-hand side of the first-order planar
radiation condition in [COMSOL's radiation theory](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.168.html)
and its [incident-pressure distinction](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.037.html).
COMSOL is a theory reference, not an execution oracle or claim of matching its
wider incident-wave functionality.

The implementation rejects noncoplanar/curved/nonparallel targets, differing
owner density or compressibility (even equal impedance), overlapping sources,
pressure/natural boundaries and periodic pairing. It requires uniform scalar
pressure, fixed frame, no separate direction and no scalar moment projection.
Spatially varying/oblique/spherical incidence, general ports, PML, attenuation
and volume-velocity sources remain outside this source envelope.

Robin source work includes an inlet passive Y|p|²/2 term. The same term is
subtracted from both source work and B loss before reporting net incoming power
(incoming minus reflected) and remaining passive outlet loss. Closure keeps the
original apparent-power scale, tolerance and iteration budget; no dimensional
floor masks small signals. Nonfinite scalar samples/constants/interpolation and
nonfinite apparent-power scales fail closed through shared validation.

## Independent reference and evidence

The [two-fluid reference](acoustic-material-interface.md) supplies exact pressure,
reflection/transmission and uniform-simplex mean velocity. The incident test
removes its prescribed TOTAL inlet pressure and instead injects only A. Actual
sealed half-volume regions have rho/c=(3,2),(5,4), so Z=(6,20), R=7/13,T=20/13.
The unit-area transmitted power for A=1+0.4i is0.0686390532544 W.

All24 cases retain the original fixed numerical gates: Tet4/Tet10, cells2/4/8,
two phases A/iA and axial/45-degree rigid rotation. Pressure refinement must
exceed3(Tet4)/6(Tet10), mean-velocity refinement1.5/3; finest RMS errors must be
below0.004|A| Pa and0.01 m/s. Net input and outlet loss must each be within0.005 W
of the independent power, with algebraic residual<1e-10 and closure<1e-8.
Finest axial A errors are5.50772e-4 Pa/3.03444e-3 m/s(Tet4),
5.66584e-7 Pa/4.62487e-6 m/s(Tet10). Tet10 net input is0.0686390530835 W.
Additional no-scattering matched-duct tests check power at amplitudes1 and1e-12,
which would fail if incident power were counted twice.

Shared projection evidence retains the real action/signal IDs and Pressure
quantity. A long-UTF8-ID admission test counts actual retained evidence and checks
canonical cold workload/result equality and authenticated identity. The resource
plan reserves retained incident-face metadata as well as selector clones and B.
Native tests execute both orders at chunk sizes1024/8192, import cold numerical
objects, separately admit shared state publication without mesh/workload inputs,
then reopen and check bit-exact original complex nodal pressure and element-mean
velocity. The native single-tetrahedron fixture is execution evidence, not a
continuum benchmark or process-separated numerical restart.

Generated-versus-explicit incoming-source equivalence through actual authored
generation, full authored incident CAD studies, general frequency-sweep state,
resonance/uniqueness diagnostics, measured resources, placement and blocking
release evidence remain required. Existing local-radiation coarse velocity
failures are not removed, narrowed or promoted by this incident-source work.
