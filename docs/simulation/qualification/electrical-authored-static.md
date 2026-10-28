# Authored static electrical vertical-chain evidence

Status: eight-case authored/native/cold fixture passes. This does not
complete Stage 7, establish restart consumption, or replace the open
largest Tet10 refinement/resource gate recorded in [electrical-static.md](electrical-static.md).

Current verification: the execution bridge passes all 13 electrical tests,
including authenticated cold imports, missing/foreign-object rejection and
coordinator accounting. The first actual authored Tet10 electrostatic process
exposed a missing worker accuracy-source inventory entry. The worker now forwards
every non-phase source object to the shared physics-owned closed inventory
validator instead of maintaining another solver whitelist. Its next actual run
reached singleton completion and exposed missing static-EM operation dispatch;
that explicit tuple is now implemented. The subsequent process reaches retained
field/cold-closure checks but fails the independent potential gate:
1.241018654507542e-7 > 1e-7 normalized error, using the default authored
algebraic target 1e-8. Its project is retained at
`/var/folders/g6/5twygq1501g6j7__xff80g300000gn/T/.tmpOPRj2t`.
An algebraic residual is not a field-error certificate. A common analysis-level
`accuracy { algebraic_relative_tolerance = 1e-11 }` setting is now implemented
and requested explicitly by this fixture. Cross-physics authoring passes two
tests spanning eight procedure shapes and thirteen invalid cases. The fresh
eight-case run passes in 397.46 s, including the independent field/closure and
authenticated cold checks below. A fresh final resource-model revision 47 run
also passes all eight cases in 435.14 s. This remains the bounded
CPU static matrix, not whole-workspace or Stage 7 qualification.
An isolated dependency-closed retention snapshot, excluding unrelated worktree
drafts and using the tracked lockfile, passes the same eight actual CAD/native/
cold cases in 410.40 s. Its bridge suite passes 13/13 and authored-domain tests
2/2. The electrical FEA suite passes 25/26; the sole failure reproduces the
previous largest Tet10 coercivity update-budget gate (43,863,736 updates against
43,840,305), without raising that cap. This retention evidence does not close
that independent refinement/resource gate or establish restart consumption.
No electrical-only hidden solver tolerance, analytic
field-bound relaxation or proof-cap change is used.

The actual-process CLI fixture is
`typed_fea_geometry_cli::electrical::authored_electrical_prism_both_equations_orders_sign_and_gauge_reopen_cold`
(requires `occt-native`). It runs eight authored producer processes: electrostatics
and steady conduction, Tet4 and Tet10, each with opposing-face voltages `(0,2)`
and `(5,3)` V. Mesh target is 0.5 m, maximum deviation 0.02 m. Failure preserves
the project and its artifacts. No fallback kernel, proxy frequency, manufactured
discrete load, relaxed tolerance, increased resource ceiling or replacement mesh
is used.

## Independent reference and exact ownership

OCCT constructs and reimports a 1 m × 1 m × 1 m prism. The study selects its first
two source faces by their canonical persistent identities
(`persistent-entity:sha256:...`), which must resolve to x=0 and x=1 respectively;
all boundary-node coordinates, outward orientation and unit face area are checked
against the actual canonical mesh. Constant prescribed potentials act on those
faces; remaining faces have natural zero flux. Every physical volume element must
retain its authored electrical material owner: relative permittivity 4,
conductivity 3 A/(V m), heating coefficient zero.

For `g=phi(0)` and `V=phi(1)-phi(0)`, the independently derived continuum reference is
`phi=g+V*x`, `E=(-V,0,0)`, `D=epsilon*E` or `J=sigma*E`.
Here `epsilon=4*8.8541878188e-12 F/m`. Energy density and unit-volume total are
`epsilon*V²/2`; Joule density and unit-volume power are `sigma*V²`.
Weak outward terminal flux is `+a*V` at x=0 and `-a*V` at x=1, with
`a=epsilon` or `sigma`; electrode charge has the opposite sign to outward D flux.
The signed/gauged pair checks potential gauge change, field/flux sign reversal,
and invariant quadratic energy/power on identical canonical mesh identities.
This is an affine patch/sign/gauge test, not a mesh-convergence rate claim.

The reference uses literal continuum expressions, not RunMat assembly/recovery.
Its weak-form and SI constant references are the independently documented
[FEniCS Poisson form](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson.html)
and [NIST 2022 CODATA vacuum permittivity](https://www.physics.nist.gov/cgi-bin/cuu/Value?eqep0=).

## Fixed gates and cold closure

Before execution, the analytic normalized absolute-error bound is fixed at 1e-7
for potential, vector fields, densities, integrated energy/power and terminal
flux. Scales are voltage span, field/constitutive magnitude or actual density,
never an arbitrary 1-SI-unit floor. Actual algebraic relative residual, relative
terminal-flux imbalance and energy identity error must each be ≤1e-8; the
explicit authored algebraic target is additionally 1e-11. The tighter solve
request does not change any independent continuum field bound.

The fresh filesystem result store must reopen all four exact native fields:
potential (node), E (volume element mean), D/energy or J/Joule (volume element
mean/density). Full packed real pages and an offset partial page are compared
bit-for-bit with retained fields and authenticated solver arrays, with exact
shapes and node/element ordering. Semantic static state must contain all four
fields and agree with the public manifest.

Only catalog objects named by the returned manifest are copied into an empty
cold cache. Exact root import replays solve, accuracy, declared state,
accounting and publication closure; a foreign trusted-root digest is rejected.
The native electrical completion/source converter then authenticates that
imported chain and each returned catalog object without solving or regenerating
missing field bytes.

## Open scope

This bounded CPU fresh-singleton static/direct envelope admits
prescribed potentials/reference and natural insulation. Charged Poisson, explicit
natural/Robin inputs, integral current, floating/equipotential/periodic terminals,
temperature-dependent laws, imported states, restart, coupling, accelerator/remote
equivalence and release evidence remain outside this claim.
Magnetostatics, Maxwell and electrothermal Stage 7 exit gates remain open.
