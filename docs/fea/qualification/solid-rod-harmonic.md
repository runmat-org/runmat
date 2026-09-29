# Solid rod harmonic qualification

This Stage 4 case connects physical solid assembly, complex face-load realization,
direct harmonic response, modal extraction/truncation, and cold typed execution to
an independent continuum reference. It supplements the [isolated operator
matrix](solid-harmonic.md); it does not promote the formulation to production.

## Independent problem and reference

The domain is the unit cube, treated as a unit-length/unit-area axial rod, with
`E=100 Pa`, `rho=1 kg/m^3`, and `nu=0`. All transverse translations are held at
zero; axial displacement is held at the left face. Thus `u=(u(x),0,0)` satisfies
three-dimensional solid elasticity exactly, without a slenderness approximation
or Poisson restraint correction. A uniform complex global-X traction
`T=1+0.4i Pa` acts on the right face. Canonical Tet4 refinements have 2, 4, and 8
cells per axis; straight Tet10 refinements have 1, 2, and 4. The shared quadratic
fixture uses canonical global-edge midsides, Tri6 faces and exact source-face
parameters, validated and sealed through the ordinary meshing contract. It is a
test fixture, not production-generated topology. This qualification does not
extend to curved Tet10 or general unrestrained rods with nonzero Poisson ratio.

The wave equation and fixed/free spectrum follow [UT Austin's longitudinal-wave
derivation, equations 5.15–5.22](https://farside.ph.utexas.edu/teaching/315/Waves/node34.html).
Reflecting its coordinate and imposing end traction gives the independently
derived forced solution for the `exp(+i omega t)` convention:

```
c = sqrt(E/rho) = 10 m/s
k = omega/c
u(x) = T sin(k x)/(E k cos(k))
R_left = -T/cos(k)
f_first = c/(4 L) = 2.5 Hz
```

The test uses `k={0.5,1,2}` on both sides of the first resonance `pi/2`, excluding
exact undamped resonance where no bounded steady response exists. Complex phase
comes from the authored frequency table, not from manually injecting a nodal
load. Shared persistent-face resolution, signal realization, and surface
quadrature produce the real and imaginary load vectors. Each phase's resultant
and origin moment are compared with `T A` and `(0,T A/2,-T A/2)`.

## Numerical and execution evidence

The normal library tests are selected with:

```sh
cargo test --locked -p runmat-analysis-fea --lib solid_rod_ -- --nocapture
```

Direct response is compared with the analytic field at every mesh node. The
reference is not RunMat's own coarse mesh, modal response, or assembled matrices.
Success requires convergence under refinement, bounded finest-mesh complex field
error, independent support-reaction agreement, zero constrained translations,
full-order physical equilibrium and zero mean input power for this undamped case.
Field and reaction errors must decrease by at least a factor of 2.5 per refinement
and both be below 2% on the finest Tet4 mesh. Tet10 field and reaction errors must
decrease by at least six per refinement and finish below 0.2%. The direct relative residual is bounded
by `1e-10`, physical closure by `1e-8`, and mean input power by `1e-9 W`.

The 2/4/8-cell complex-field relative errors are:

| Wavenumber | 2 cells | 4 cells | 8 cells |
| --- | --- | --- | --- |
| 0.5 | 0.5114% | 0.1425% | 0.0341% |
| 1 | 2.3766% | 0.6767% | 0.1664% |
| 2 | 25.5902% | 6.5248% | 1.6170% |

The coarse high-frequency result is deliberately not qualified by its solver
residual alone. Its continuum error is large and decreases under refinement.

The Tet10 1/2/4-cell complex-field relative errors are:

| Wavenumber | 1 cell | 2 cells | 4 cells |
| --- | --- | --- | --- |
| 0.5 | 0.223556% | 0.024156% | 0.002421% |
| 1 | 0.859663% | 0.092029% | 0.009051% |
| 2 | 3.927809% | 0.451831% | 0.040283% |

Finest Tet10 support-reaction errors are 0.0000041%, 0.000370% and 0.016467%.
Both phase resultants/moments, full-order physical equilibrium and zero mean
input power pass the same criteria as Tet4. High-order interpolation does not
make the coarse near-resonance mesh accurate merely because GMRES converges.

Fundamental-frequency refinement is checked against 2.5 Hz. On a fixed 27-node
mesh, retained bases of 1, 8, and all 18 axial free DOFs separate modal truncation
from spatial error. Each retained response is compared with the continuum, and
the complete basis must reproduce the direct discretized solution. Direct/modal
agreement is supplemental consistency evidence, not the independent reference.
Fundamental-frequency relative errors are 2.0665%, 0.5901%, and 0.1561%, decreasing
by more than a factor of 3 each time and below 0.2% at 8 cells. On the coarse fixed
mesh, direct/modal response differences decrease from 11.3905% to 5.1476% to
`8.36e-13` for 1/8/18 retained modes; each retained response must remain within 15%
of the independent continuum. Complete-basis agreement is required within `1e-7`.

Tet10 fundamental-frequency errors are 0.218346%, 0.019740% and 0.001439% on
1/2/4 cells, decreasing by more than six and below the pre-set 0.02% finest bound.
The one-cell quadratic mesh also has 27 nodes and 18 axial free DOFs; its retained
1/8/18-mode differences from direct decrease from 11.342269% to 6.185547% to
`2.99e-13`. Each response independently meets the 15% continuum bound. These
constrained axial bases do not qualify unconstrained bending/shear spectra.

The cold execution case serializes the typed workload, entry request, mesh,
physical entry result, and recovered field artifact. A cold direct execution and
field recovery must preserve the hot physical sample and field artifact and
satisfy the independent continuum bound. A different sealed topology or changed
traction may not reuse the original admitted request. This verifies cold object
and authority boundaries; it does not simulate process failure, worker replacement,
checkpoint publication, or a distributed solve.
The cold execution uses the Tet4 8-cell and Tet10 4-cell meshes at `k=1`, with
0.5% and 0.05% continuum field bounds respectively. The recovered complex
displacement artifact is checked against the analytic field, not only compared
with the hot artifact or accepted as nonempty.
Geometry substitution preserves every node ID and constraint target while moving
an interior node and resealing a valid mesh; it must fail on the admitted identity,
not because a target disappeared. Source substitution must likewise fail on identity.

## Gate decision

The envelope remains experimental. Curved Tet10 continuum/recovery,
damped continuum references, authored source-language and generated-input
equivalence, formal resource/accuracy admission, matching solver-specific
restart/replacement evidence, supported accelerator/distributed execution, and
the blocking release lane remain separate obligations. Existing checkpoint
mechanisms are dependencies, not numerical qualification substitutes. No API,
production admission, UI, runtime proxy, or release claim is changed by this case.
