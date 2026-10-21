# Magnetostatic vector potential — physical volume-current baseline

## Decision and ownership

Stage 7 remains open. This is a physical **magnetostatic envelope**,
with authored CAD/native execution, governed retained fields and observations,
and authenticated cold completion reopening. It is not Stage 7 qualification,
numerical restart, measured resource or release qualification.
The old reduced harmonic pipeline explicitly rejects magnetic intent;
its graph tests are not evidence for this formulation. Its eventual replacement by
physical Maxwell remains required.

The distinct magnetic domain names the magnetic-vector-potential formulation, with
no invented frequency or electrostatic reinterpretation. The new typed
`magnetic_volume_current_density` is vector J in A/m² on exact volume regions or the
whole model. Existing surface `current_density` retains its existing semantics.
Generic realization preserves complex components; the static magnetic consumer
requires a finite real constant vector in an explicit global frame and fixed volume
distribution. Non-global frames are rejected, not silently treated as global.

`physics/magnetic` owns element permeability/source compilation, physical assembly,
boundary/gauge selection, solve and A/B/H recovery. It reuses shared persistent
topology, oriented-edge basis, quadrature, material assignment, excitation realization,
sparse matrices, canonical reductions and indefinite GMRES. No new crate, private
matrix representation, scalar substitute, callback sampling or penalty gauge is used.

## Physical contract

The numerical equation is

`curl(μ⁻¹ curl A) = J`, `B=curl A`, `H=B/μ`.

Assembly uses actual volume integrals of `μ⁻¹ curl N_i · curl N_j` and `N_i · J`.
The weak Coulomb constraint is `∫A·grad q=0` for zero-boundary corner scalar q,
implemented as a symmetric saddle system using exact integrated coupling. It is
not an SPD claim, diagonal stabilization or gauge penalty. The scalar gradient
kernel comes from oriented endpoint differences. Sources must annihilate it;
incompatible weak divergence is rejected, never cleaned without user authority.

The initial named boundary envelope is explicit full-exterior tangential A=0 on
one connected tetrahedral ball, with a full-volume Coulomb-gauge declaration.
Complete canonical exterior-face inventory, closed spherical boundary, connected
face adjacency, manifold vertex links and volume/boundary Euler characteristics
are checked. Disconnected/cavity/handle topology needs its own nullspace/gauge
contract and is not silently admitted.

Lowest-order six-edge approximation on affine Tet4/Tet10 geometry is supported.
Tet10 geometry is not a claim of quadratic magnetic approximation. General curved
geometry is rejected here until governed rational-integrand quadrature convergence
is supplied. Positive finite isotropic μ is element-owned; σ is not required for
magnetostatics. Frequency and temperature-dependent laws are not inferred.

An invertible common permeability scaling and dimensional gauge-row scaling balance
the saddle equations without changing the field. Acceptance independently checks
the **unscaled physical** K·A−f residual, weak gauge residual, quadrature energy
`U=½∫B·H`, and physical source work `W=∫J·A`, with `2U=W` under this ground policy.
Recovered fields are produced once from the solved edge circulations. Cold result
import validates those retained arrays; it does not reconstruct fields after loss.

## Portable execution and retention contract

The exact realization distinguishes integrated lowest-order edge
curl-curl, element-owned permeability, weak-Coulomb saddle solve, A/B/H/energy
recovery, oriented canonical Tet4/Tet10 topology, volume projection and explicit
full-exterior grounding. Spliced scalar/reduced-graph identities
remain rejected. Generic static selection does not guess an equation. The legacy
solver compatibility table admits only assembled saddle-point GMRES with no
preconditioner for this magnetic workload; electrical CG/PCG and harmonic
procedures cannot substitute for it.

`MagnetostaticWorkload` uses the shared canonical codec and identity foundations.
The retained result binds original workload, producer, exact formulation, mesh,
projection, geometry order and residual target. It retains oriented edge
circulations and all physical quadrature samples, not only a plot or coefficients
requiring later field reconstruction. Cold validation verifies canonical sample
cardinality/order/geometry, affine/topological eligibility, finite SI arrays,
original element permeability, original volume current and retained energy/work
integrals. It assembles no matrix and runs no numerical solve. Cold import requires
an independently authenticated expected byte digest: source binding alone cannot
authenticate arbitrary producer field values or certify their spatial accuracy.
Round-trip tests drop live solver values before reopening both geometry orders,
both current signs and zero current; these are contract-level cold consumers,
not by themselves producer-exit native process or numerical-restart evidence.

Before projected topology, selected properties or numerical assembly are allocated,
checked count-only preflight reserves additive numerical requested-heap allowances
for support/boundary discovery, shared projection estimates, arbitrary material/
assignment/source metadata, contribution discovery and CSR, GMRES basis/Hessenberg/
vectors and retained samples. Bounds use E≤6T, D≤6T+N and Q=15T. Overflow,
unrepresentable u32 iteration counts and ceilings one byte below the reservation
are rejected. This is not measured RSS or a generic scheduler resource plan;
The governed native path additionally derives exact task-specific scheduler and
encoded-output reservations. Independent measured residency evidence remains open.

## Independent continuum evidence

Extruded unit square, constant μ, uniform J=(0,0,J₀), full exterior tangential A=0:
`A=(0,0,a(x,y))`, `−Δa=μJ₀`, `a=0` on the square perimeter. The z end caps are
consistent because A is normal there. Independently separated Fourier/hyperbolic
series supply a and its derivatives; no RunMat solve is used as the reference.
Stable exponential ratios avoid hyperbolic overflow. For truncation N, conservative
uniform tails are `|a_tail|≤2μ|J₀|/(π³N²)` and each derivative tail is
`≤4μ|J₀|/(π²N)`. Field comparisons use N=1023.

Energy is independently integrated analytically from the double sine series:
`U = 32μJ₀²/π⁶ Σ_(m,n odd) 1/[m²n²(m²+n²)]`.
The N=255 truncation has tail `≤8μJ₀²/(3π⁴N³)`, separately checked against N=511.

For μ=J₀=1, the predefined three-window witness measured:

| Cells/axis | L² A error | L² B error | Relative energy error | GMRES iterations |
|---|---:|---:|---:|---:|
| 1 | 0.03717750 | 0.16614785 | 0.80240170 | 1 |
| 2 | 0.02446006 | 0.11642631 | 0.38710798 | 9 |
| 4 | 0.01317592 | 0.06443002 | 0.11824096 | 213 |

Each error decreases by >1.4 per refinement; finest bounds are A<0.02, B<0.10,
energy<0.25. These are low-order baseline gates, not high-accuracy release criteria.
Additional tests verify signed current and physical vacuum-permeability scaling,
affine Tet4/Tet10 equivalence, energy/source-work closure, shared-face B-normal flux
continuity and total flux closure, explicit model materials/ground/gauge compilation,
exact-volume signed source superposition, zero conductivity and zero current.
Invalid material/identity/cardinality, incompatible current, missing ground/gauge,
surface-current substitution, wrong domain and unqualified curved geometry fail closed.

The physical weak-form/source requirements follow
[MFEM's magnetostatic notes](https://mfem.org/tesla-notes/).
Vacuum permeability uses the measured
[NIST CODATA 2022 value](https://www.physics.nist.gov/cuu/pdf/wallet_2022.pdf),
not the historical exact 4π×10⁻⁷ SI convention.

## Authored native whole-study evidence

The exact equation selects magnetic preparation independently of static electrical
and harmonic domains. A fresh independent direct step executes real CAD meshing,
an admitted native magnetic solve, retained field publication, sampled observation
partitions/reduction, shared accuracy evaluation, every declared semantic state,
baseline accounting and terminal publication accounting. Electrical and magnetic
adapters share the terminal lifecycle owner without sharing scientific identities.
An explicit authored per-worker envelope governs placement; fallback task hints do
not silently resize independently configured pools.

The authored unit-box test uses vacuum permeability, J=(0,0,1) A/m², 0.5 m target
size, full exterior grounding and a full-volume weak Coulomb gauge. Both affine
Tet4 and Tet10 runs complete with relative physical and gauge residuals at most
1e-10 and energy/work error at most 1e-9. Against the independent series above,
both report relative energy error approximately 0.11259 and volume-weighted B
relative L² error approximately 0.33548. These are coarse execution witnesses,
not a refinement study or qualified spatial-accuracy claim. Tet10 still uses the
same lowest-order edge approximation.

The test copies the exact completion catalog into an independent filesystem store,
drops the original session/source/executing pools, and reopens the durable root.
The reopened completion equals the original complete DTO, including retained
fields, observations, states, accuracy and accounting. Replay authenticates
immutable responses; it performs no numerical solve. Missing/extra/foreign
observation catalogs and substituted source, sampling or authored accuracy
requirements are rejected, including newly rehashed substitutes.

Reproduce the native witness from this repository:

```sh
cargo test --locked -p runmat-execution-runner-native --features occt-native \
  --test analysis_process_conformance -- --magnetic-authored-only
```

The default OCCT process conformance suite includes this witness. Focused
`--magnetic-only` remains the smaller numerical/task regression.

## Remaining implementation and gates

Complete mandatory spatial accuracy, weighted volume observations, measured resource
qualification and real numerical restart/replacement evidence, then apply release
gates. An algebraic accuracy report is not discretization evidence; missing mandatory
spatial criteria remain Indeterminate and block publication. Cold artifact reopening
is not numerical restart. The standalone authored path rejects unsupported restart
and dependent/mixed sequences.
Extend boundary/topology/material/source envelopes with their own physical evidence;
ports/periodicity, nonlinear magnetic laws, permanent magnetization and force recovery
are not borrowed from this witness. Replace legacy Maxwell proxies at the physical
cutover. GPU/distributed claims require actual placement/operator evidence, not this
serial solve. All wider Stage 7 and cross-family plan obligations remain intact.
