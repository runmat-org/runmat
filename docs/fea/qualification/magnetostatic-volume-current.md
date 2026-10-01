# Magnetostatic vector potential — physical volume-current baseline

## Decision and ownership

Stage 7 remains open. This is a physical **experimental numerical/model-compilation
baseline**, not production admission, authored native execution, restart, or release
qualification. The old reduced harmonic pipeline explicitly rejects magnetic intent;
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
Recovered fields are produced once from the solved edge circulations. No cold import
or reconstruction after loss is implemented or claimed by this baseline.

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

## Remaining implementation and gates

Complete distinct magnetic formulation/admission/resource identity, checked transient
and retained-memory/peak envelopes, portable workload/result/receipts and magnetic
semantic-state/field publication. Carry the actual authored CAD/native process through
full/partial reads, source-bound cold reopening and numerical restart, then apply
independent accuracy and release evidence. The model numerical entry validates allowed
typed state declarations but does not fulfill their governed publication lifecycle.
Extend boundary/topology/material/source envelopes with their own physical evidence;
ports/periodicity, nonlinear magnetic laws, permanent magnetization and force recovery
are not borrowed from this witness. Replace legacy Maxwell proxies at the physical
cutover. GPU/distributed claims require actual placement/operator evidence, not this
serial solve. All wider Stage 7 and cross-family plan obligations remain intact.
