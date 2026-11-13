# First-order local acoustic radiation

## Bounded implementation

`BoundaryOperator::AcousticRadiation` lowers to the passive local outgoing condition
`v_n=p/(rho*c)` on each exact exterior face. The density and compressibility are
the already compiled values of that face's unique adjacent volume element.
`Y=sqrt(beta)/sqrt(rho)=1/(rho*c)` avoids an overflowing impedance product and
an underflowing intermediate `beta/rho`. Missing, multiple, foreign or nonphysical
owners reject; no default medium is substituted. Existing overlap and exterior-face
checks remain authoritative. No new result schema, retained buffer or checkpoint is added.

For `exp(+i omega t)`, linear momentum gives `i omega rho v=-grad p`.
Thus the local condition is `partial_n p=-i omega p/c`. Integration by parts
adds `+i omega integral_face N_i N_j/(rho*c)` to the existing Helmholtz operator.
Positive real admittance produces positive outward mean power
`0.5 integral_face |p|²/(rho*c)`. The retained metric `mean_impedance_loss_w`
includes all passive boundary admittance, both explicit impedance and radiation.
`mean_input_power_w` has the opposite sign to outward inlet power.

[COMSOL's radiation-boundary discussion](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.168.html)
distinguishes planar, spherical and cylindrical radiation and normal-incidence/angular
limitations. It supplies context, not numerical reference values for this qualification.

## Independent evidence and verification status

The canonical child `tests/radiation.rs` solves a unit planar duct with rigid sides,
prescribed complex inlet pressure and a radiation outlet. It uses both an axial duct
with compiled `rho=3,c=2` and a genuinely rotated/resealed 45-degree duct with
compiled `rho=5,c=4`. With `A=1+0.4i` or its quadrature rotation and `k=2*pi*f/c`,
independent references use the physical axial coordinate and rotated velocity direction:

- Pressure: `p(x)=A exp(-ikx)`.
- Volume-average element velocity: `v_x=A E[exp(-ikx)]/(rho*c)`, transverse values zero.
- Outlet outward power and inlet inward power: `|A|²/(2*rho*c)`; inlet outward power is negative.

The exponential volume average uses independent exact simplex moments and its convergent
series, rather than evaluating the wave at the element centroid. Mandatory refinements
are Tet4 `cells=[2,4,8]`, frequency `0.2` for `c=2`, and Tet10 `cells=[1,2,4]`,
frequencies `[0.2,0.5]` for `c=2`. The `c=4` case doubles each frequency to preserve
the tested wave numbers. Both media/orientations and both phases run at every level:
36 independent solves. Tet4 pressure error must reduce by more than 3 per refinement,
with final absolute RMS below `0.004 Pa`; Tet10 must reduce by more than 6, with final
relative RMS below `0.004`. Volume-average velocity RMS must reduce by more than
1.5 (Tet4) or 3 (Tet10) and finish below `0.01 m/s`. These correspond to the expected
first-/second-order element-mean gradient consistency, not a nodal velocity claim.
Final positive outlet loss and separately integrated signed recovered-field inlet/outlet
power must agree with the analytic power within `0.005 W`.
Residual and power closure are separate algebraic checks, not discretization certificates.
Boundary tests distinguish the owning medium from the first material and reject
interior/invalid owners. Existing native publication tests now exercise both impedance
and radiation through cold filesystem result import and separately admitted shared
state publication for both orders and chunk sizes `1024,8192`.

Status: initial canonical run passed 14 tests and failed the new rotated radiation gate:
outlet power `0.0172293077 W` versus independent `0.029 W`. Follow-up diagnostics failed
world-affine gradient reproduction and operator covariance, identifying an incorrect
off-diagonal cofactor in the shared topology inverse. The one-index correction and
direct two-sided inverse/oblique affine regressions pass (34 topology tests). Corrected
canonical rerun passes 16 tests, including rotated affine and operator/pressure/velocity
covariance and all earlier acoustic regressions. Rotated Tet4 outlet power is now
`0.0289609271 W` versus independent `0.029 W`. Radiation then fails the unchanged Tet10
coarse velocity-reduction gate: RMS errors `[0.000498719, 0.000168832, 0.0000461577]`,
first reduction `2.9539` below required `3`. A direct rerun of the retained test binary
in run `75293` reproduces this failure. Its SHA-256 is
`5f9629c88280e4df0c58eca56ee7f3ef35cde491768d791d3b11c40ed6e2d1d8`.
Symbol inspection and bounded disassembly establish that this artifact's velocity-error
closure calls `radiation::mean_wave`, so the reproduced failure uses the independent
volume-average oracle, not a centroid approximation. This hash does not prove the
identity of earlier failed artifacts without preserved failure-time hashes.

The diagnostic test now collects every numerical gate failure across all 36 mandatory
solves before failing, retaining every bound, reference, refinement, physical parameter
and solver setting. An independent literal inventory requires exactly the 12 parameter
combinations times their three refinements, with no duplicate or missing solve.
Execution and invalid-geometry failures still fail closed immediately. This change
only improves failure visibility; it cannot qualify an envelope with a failed metric.
The expanded matrix executes all 36 solves and records exactly four velocity-reduction
failures: Tet10 lower wave number, both phases, in both media/orientations. All other
unchanged pressure, power and velocity gates pass. This envelope remains unqualified;
no threshold or reference is weakened. Acoustic execution qualification remains open.

A new diagnostic (run34938: all36 cases evaluated) separates recovered mean velocity error into the
Galerkin-solution contribution `v_h-v_I` and exact-nodal-interpolation contribution
`v_I-v_exact`. It invokes the existing recovery on analytic nodal phasors without an
additional solve. Independently, the divergence theorem evaluates `mean(grad p_I)`
from all four element faces: affine Tri3 corner integrals are area/3, and affine Tri6
corner integrals are zero while midside integrals are area/3. Outward area vectors
come directly from corner cross products and the omitted-corner orientation test;
volume comes from the corner determinant. This independent face reference must agree
with existing recovery within `1e-11 |A|/(rho*c)` per velocity component. RMS diagnostic
contributions are reported separately and need not add arithmetically because their
cross term can cancel. This adds a recovery check without replacing any continuum
reference, mesh, solver setting or numerical acceptance condition.
Independent face-recovery checks pass near1e-15; the same four velocity-refinement gates
still fail. This rules out the tested exact-nodal mean recovery as their cause, not a
general proof of acoustic accuracy. Strict diagnostic lint initially failed16640 and was
corrected without suppressions; fresh FEA all-target strict lint15404 passes, and joint
FEA/native all-target strict lint52384 subsequently passes. These do not close the four
failed numerical gates.

## Explicit exclusions

Additional finer-window evidence (not a replacement gate): the low-wave-number
Tet10 matrix at cells=[2,4,8] passes all12 solves, both phases and both
media/orientations, in40.04s. Axial velocity RMS errors are
[1.68831505e-4,4.61576889e-5,1.22281979e-5] m/s, reductions3.658/3.775.
Pressure RMS errors are[1.35488986e-4,1.34630947e-5,1.27434702e-6] Pa.
The rotated medium's velocity values scale by0.3; signed recovered power,
positive passive power and original algebraic/physical closure gates also pass.
Independent simplex K/M/B entry and complex lifting tests pass3/3.
The original36-solve matrix was rerun separately and still fails exactly its
same four coarse velocity gates (30.75s). No test, threshold or coarse row was
removed. This is consistent with entry into the expected asymptotic regime,
not a correction or qualification of the declared coarse envelope. The tested
recovery/operator checks do not justify an implementation patch. The finer
test remains an uncommitted diagnostic alongside the unqualified radiation draft.

This is a local first-order absorber, proposed for qualification for planar propagation
normal to the outlet. It is not an exact general exterior-domain Dirichlet-to-Neumann
map, PML, spherical/cylindrical radiation, or a reflection-free oblique-incidence model.
Curved radiation surfaces and heterogeneous-interface wave accuracy remain outside
this bounded evidence, although lowering resolves every face's medium. The only
rotated geometry in the proposed envelope is the explicitly tested rigid 45-degree duct rotation.
Cold artifact reopening does not establish process/iteration restart. Stage 8
qualification remains open; multi-frequency state semantics are unchanged.
