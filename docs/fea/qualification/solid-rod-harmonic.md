# Solid rod harmonic qualification

This Stage 4 case connects physical solid assembly, complex face-load realization,
direct harmonic response, modal extraction/truncation, and cold typed execution to
an independent continuum reference. It supplements the [isolated operator
matrix](solid-harmonic.md); it does not promote the formulation to production.

## Damped modal extension — qualification in progress

For the same constrained straight rod, `phi_1=sin(pi x/2)` has independent modal
mass `1/2`, stiffness `E pi²/8` and end-force amplitude `T`. With Rayleigh
coefficients `alpha=0.8 s^-1`, `beta=0.002 s` and `exp(+i omega t)`, its exact
retained-one-mode field is

```
u_1(x) = 2 T sin(lambda x) /
         [E lambda² - omega² + i omega (alpha + beta E lambda²)]
lambda = pi/2
```

This is a reference for first-mode spatial error, not the full forced PDE response.
The extension separates first-mode mesh/phase/power convergence from truncation:
on fixed 4-cell Tet4 / 2-cell Tet10 meshes (100 free axial DOFs), 1/8/100 retained
modes must approach the direct discretized response; the full basis also has
independent coarse PDE field and power bounds. Cold tests decode the modal state,
workload, request and canonical mesh, rederive preparation and recover complex
displacement against the analytic first-mode field. Missing state and a valid
resealed state naming another topology reject. These are portable-object checks,
not process restart or arbitrary modal-spectrum qualification.

With the original shared ordering, the Tet10 first-mode 1/2/4-cell tests pass
their prescribed refinement factors and finest 0.5% field/power bounds. Finest
complex-field errors are 0.02162%, 0.04138% and 0.02442% at
`k={0.5,pi/2,2}`. Both-order truncation and cold canonical-entry/recovered-phase
checks also pass; these results do not qualify the missing Tet4 refinement.

The first-mode refinement test remains **unqualified**: its required 16-cell
Tet4 stiffness coercivity proof exceeds the shared symbolic update-work limit
before eigensolving. The mesh and its accuracy bound are not reduced, and proof
limits are not bypassed. A bounded graph-separator ordering experiment also
failed the actual modal and electrical cases and was discarded. Passing coarse
truncation/cold checks do not close this missing convergence gate.

```sh
cargo test --locked -p runmat-analysis-fea --lib pipeline::modal::rod_qualification::damped_modal -- --nocapture
```

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
extend to general unrestrained rods with nonzero Poisson ratio. The separate
[analytic curved Tet10 envelope](curved-solid-harmonic.md) qualifies a fixed
quadratic extrusion with its own geometry, force/moment and recovery checks;
it does not turn this straight fixture into general curved-CAD evidence.

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

## Damped continuum extension

The new Rayleigh case uses `alpha=0.8 s^-1`, `beta=0.002 s`, and the same
physical rod/complex end traction. The mass-proportional body drag and
stiffness-proportional viscous stress follow the
[Abaqus material damping definition](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-ovwmatdamping.htm).
The following continuum solution is derived here from that constitutive law
and the rod equation; it is not a solver-to-solver reference:

```
Ec = E*(1+i*omega*beta)
q^2 = rho*(omega^2-i*omega*alpha)/Ec
U(x) = T*sin(q*x)/(Ec*q*cos(q))
R_left = -T/cos(q)
P_in = omega/2 * Im(T*conj(U(1)))
P_loss = omega^2/2 * integral[alpha*rho*|U|^2 + beta*E*|U'|^2] dx
```

The reference evaluates sine/cosine by their entire series in `q^2`, avoiding
branch-dependent complex square roots. Independent Simpson integration of the
continuum loss must agree with analytic boundary power. The FE case compares
complex nodal displacement and support reaction, checks spatial refinement,
and requires positive FE loss and agreement with input power. Frequencies
include `k={0.5,pi/2,2}`: damping permits a bounded response at the undamped
resonance. Cold typed execution/recovery cases extend both Tet4 and Tet10 at
`k=1`, preserving damping in workload serialization and recovered phase.
Both cold cases pass. Initial Tet4 2/4/8 resonance field errors are
45.15398%, 14.30231%, and 3.81678%: convergent but above the pre-set 3% bound.
The test extends Tet4 refinement to 16 cells, preserving the frequency and bound.
The 64-vector/4096-iteration policy did not converge at the finer resonance;
the continuum case uses an explicit 256-vector/8192-iteration policy with the
same `1e-10` residual target. All three focused tests pass. Finest Tet4 complex
field errors at `k={0.5,pi/2,2}` are 0.007790%, 0.97310%, and 0.39216%;
finest Tet10 errors are 0.0024519%, 0.041160%, and 0.040291%.
Tet4/Tet10 field errors decrease by at least 2.5/5 per refinement, respectively;
finest field and reaction bounds are 3%/0.3%. FE input/loss power agreement is
required within `1e-8`; continuum boundary/volume power agrees within `1e-9`.
Finest FE versus continuum input-power bounds are 5%/0.5%. Cold recovery retains
the original 0.5%/0.05% field bounds at `k=1`.
Production defaults are unchanged; the larger Krylov workspace is not measured
or formally admitted. This is numerical and cold-object evidence, not a
process-restart or production qualification claim.

## Gate decision

The envelope remains experimental. Curved Tet10 continuum/recovery (including damping),
authored source-language and generated-input
equivalence, formal resource/accuracy admission, matching solver-specific
restart/replacement evidence, supported accelerator/distributed execution, and
the blocking release lane remain separate obligations. Existing checkpoint
mechanisms are dependencies, not numerical qualification substitutes. No API,
production admission, UI, runtime proxy, or release claim is changed by this case.
