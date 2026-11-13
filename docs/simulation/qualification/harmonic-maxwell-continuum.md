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

## Authored and retained numerical boundary

Source intent names `electromagnetic { equation = harmonic_maxwell;
reference_frequency = ... Hz }` and homogeneous `electrical.perfect_conductor`
face constraints. PEC means n cross E=0, not a scalar-potential or magnetic-A
ground. The compiled fresh direct singleton frequency must exactly match the
domain frequency; omega=2 pi f. Full exterior coverage is explicit. Constant
element-owned permeability, permittivity and positive conductivity use their SI
values, not ordinal material blocks or temperature/frequency response proxies.

The shared legacy-named MagneticVolumeCurrentDensity action supplies impressed
volume J; boundary CurrentDensity is not reinterpreted. Explicit global vector
constants or complex frequency tables are realized at the exact Hz value and
frozen on canonical Keast15 samples. This is piecewise uniform authored volume
current, not a claim of spatially varying authored-source support.

MaxwellHarmonicWorkload and MaxwellHarmonicSolverResult retain original model,
producer, formulation, sample/material/current identities and genuine E/B/H/D/J
fields. Canonical cold import requires an authenticated expected result digest
and original workload/mesh, and validates constitutive and energy/power summaries
without solving again. Retained validation compares original E/B samples with
their oriented electric edge phasors and evaluates the original physical weak
action locally, without constructing a sparse operator or replacing fields.
The solve and retained validation share one charge-continuity accumulator. This
validates numerical result authority; whole-study replay separately authenticates
the authored lifecycle and its complete descendant closure.

Local compilation preflight checks the selected maximum_resident_bytes before
model validation, projection, property replication or sampled source expansion.
Count bounds are edges<=6T, real block dimension<=12T, support candidates<=144T,
and15 samples per tetrahedron. Shared GMRES restart/iteration sizing is used;
profiled metadata, sparse topology/assembly, retained and cold-validation copies
are conservatively reserved. This is requested local heap, not measured RSS,
measured process allocation or scheduler-attempt reconciliation.

The authored/retained suite passes both mesh orders, source-bound cold import,
constitutive/power checks and substitution negatives (two groups); local resource
preflight passes three groups. Four language tests and all 124 core tests pass.
These checks do not establish Stage qualification.

## Native task and retained-field boundary

The immutable Maxwell task carries original workload, canonical mesh and exact
resource plan. Its worker authenticates bounded plan and source envelopes before
decoding numerical inputs, then rederives the complete resource and execution
identities before quadrature, sparse assembly or shared GMRES. Admission includes
actual canonical input decode budgets, conservative result/state/receipt transport
profiles and numerical reservations. These are requested allocations, not measured
RSS or distributed/GPU execution evidence.

The result exposes six original complex native fields: oriented electric edge
circulation (V), E, B, H, D and conduction J. Shared interleaved storage uses
[edges,2] or [samples,3,2], at exact original Keast15 integration points. Recovery
copies retained values and never solves again; cold imports authenticate original
source, producer, mesh, request, resource authority and immutable payload.

Both-order native task tests execute solve, cold import, field recovery, complex
observation evaluation, complete ElectromagneticComplex state publication and
accuracy evaluation through shared artifact-store programs. Exact phase values
survive cold imports and partition transport. These are native task-chain witnesses,
not by themselves a completed authored-study root, independent spatial evidence, process-isolated
restart, resource measurement or release evidence. Missing mandatory
spatial evidence remains Indeterminate and blocks publication.

## Authored whole-study composition and proxy cutover

The standalone positive-loss, full-exterior PEC, singleton-frequency authored
study now composes shared CAD/meshing, exact model/property/source planning,
governed native numerical tasks, six retained complex fields, sampled observations,
complex state, original accuracy, usage and terminal publication. Magnetic and
Maxwell share one concrete sampled-observation lifecycle and one terminal lifecycle;
placement does not introduce another solver or recompute lost fields.

The completion root replays original saved responses, rederives original authored
tasks/resource reservations, and requires the exact descendant object catalog.
The runtime source compares the complete replayed completion before projection.
Its only publication controls are FP64 and the original residual target; inert
quality-policy and determinism switches are not exposed. Missing, extra,
duplicate or foreign evidence, changed accounting or model/property authority
cannot fill an incomplete result. No numerical solve occurs during reopening.

The native authored witness uses a unit PEC cube, impressed Jy=(0.6+0.8i) A/m²,
sigma=3 S/m, mu=mu0, epsilon=4 epsilon0 and omega=10^6 rad/s. A separately derived
double sine Fourier series supplies complex E/B and peak ohmic power; truncation
is checked independently. The coarse CAD mesh has deliberately broad witness
E/B L2 and power bounds, not qualified convergence tolerances. Both affine
geometry orders use the same lowest-order Maxwell approximation.

The full native authored witness passes for both orders, including all cold-root
and substitution checks. Relative complex E/B L2 errors are 0.33870749/0.36726383;
relative peak-power error is 0.09854191. Independent Fourier truncation bounds are
4.25753e-6 for E and 5.42806e-4 for B. These coarse-mesh results satisfy only the
explicit witness bounds, not the remaining qualification gates.

The witness persists the entire original catalog, drops producer/session handles,
reopens the filesystem object store, compares exact complex completion and
republishes it. It rejects wrong trusted roots, omitted objects, repeated/missing
accuracy closure, rehashed accounting/property substitution, sweeps and numerical
restart. This cold reopening is not a process-separated numerical restart,
measured memory bound, remote solve or Stage qualification.

The old reduced-edge harmonic graph, fabricated vector-potential A fields,
raw runtime fallback, obsolete solver contracts/options and proxy sweep payload
are deleted. Retired wire tags fail closed; no compatibility numerical fallback
is retained. Ordinary mesh rendering requires an authenticated typed completion.

## Numerical evidence

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

## Scope and open gates

This work implements numerical continuum and bounded authored/native completed-study
boundaries, not numerical restart or Stage qualification. Initial uniqueness
admission requires positive frequency and strictly positive conductivity in every
element. A converged residual alone cannot establish uniqueness at lossless cavity
resonance; a genuine spectral/nonresonance authority remains mandatory for lossless
envelopes. Affine Tet4/Tet10 geometry only; curved geometry needs governed quadrature.

Remaining full-plan work includes incident/ports/open boundaries,
dispersive/complex material laws, lossless cavities
and waveguides, skin-effect/interfaces/frequency references, completed authored
native placement equivalence, numerical restart, measured execution accounting,
and release. The deleted harmonic graph supplies no qualification credit.
