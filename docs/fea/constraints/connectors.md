# Structural connectors

Connectors attach regions through a localized constitutive relation. They are not mesh elements
chosen by the mesher and not substitutes for contact.

For relative displacement $\Delta\mathbf u$ and relative velocity $\Delta\dot{\mathbf u}$, the
implemented connector contracts represent $\mathbf f=k\Delta\mathbf u$ and
$\mathbf f=c\Delta\dot{\mathbf u}$. An elastic foundation distributes the same stiffness law over
the measure of its target region.

| Kind | Required fields | Use |
| --- | --- | --- |
| `translational_spring` | `primary`, optional `secondary`, `stiffness` | Relative translation to force |
| `translational_dashpot` | `primary`, optional `secondary`, `damping` | Relative velocity to force |
| `elastic_foundation` | `primary`, `stiffness` | Distributed foundation stiffness |

Each connector names a primary target and, where required, a secondary target. Its coefficients
carry physical units and its frame/direction are explicit. A connector should appear in the Desktop
study preview as a relation between the exact target regions, not as a generic label on the model.

## How do I add a translational spring between two regions?

```fea
connector mount_spring {
    translational_spring {
        primary = geometry.assembly.region.part_mount
        secondary = geometry.assembly.region.frame_mount
        stiffness = 25 kN/m
    }
}
```

The spring force follows $\mathbf f=k(\mathbf u_p-\mathbf u_s)$ under the connector's supported
translation contract. Omit `secondary` only for a spring to the formulation's fixed reference. A
single scalar stiffness is not a full anisotropic bushing; do not use it where directional/coupled
stiffness is required but unsupported.

## How do I add viscous damping with a dashpot?

```fea
connector mount_damper {
    translational_dashpot {
        primary = geometry.assembly.region.part_mount
        secondary = geometry.assembly.region.frame_mount
        damping = 250 N*s/m
    }
}
```

Dashpot force follows relative velocity, $\mathbf f=c(\dot{\mathbf u}_p-
\dot{\mathbf u}_s)$. It contributes to dynamic response and dissipated energy; it does not add
static stiffness. Verify the coefficient against the modeled DOF basis and units.

## How do I model an elastic foundation?

```fea
connector compliant_bed {
    elastic_foundation {
        primary = geometry.plate.region.supported_face
        stiffness = 5 MN/m
    }
}
```

The current typed contract uses force-per-length stiffness. Confirm how the admitted formulation
distributes that scalar over the selected region and inspect the resolved contribution evidence;
do not assume an unimplemented Winkler modulus per area.

## How do I model a compliant mount or isolator?

Use a translational spring for static/dynamic compliance and a dashpot for viscous damping, with the
same physical attachment regions. If both are present, declare both connectors so stiffness and
damping remain separately inspectable. A real elastomer may be nonlinear, frequency dependent, and
multiaxial; the scalar linear contracts are appropriate only when that reduction is justified.

## How do I attach a lumped component to a structure?

Couple the attachment face to a reference point with the appropriate structural coupling, place the
lumped inertia at that reference point through the admitted mass model, and connect it with spring/
dashpot relations as required. Do not spread a point mass across arbitrary face nodes without
preserving its center of mass and rotational inertia.

## How do I model a foundation under a plate or machine base?

Use `elastic_foundation` only when its admitted stiffness measure matches the engineering foundation
law. A scalar `N/m` connector and a Winkler modulus in `N/m^3` have different dimensions. If the
support pressure is $p=k_w u_n$, use a formulation that explicitly accepts $k_w$ per area/volume;
do not enter the same number into a scalar spring field.

## How do I model a spring or damper to ground?

Omit `secondary` only when the admitted connector explicitly interprets omission as the fixed
reference. The primary target still needs a physical region and the resulting force/reaction must be
reported against that reference. Do not create a dummy geometry region merely to imitate ground.

## How do I model a directional spring, bushing, or coupled stiffness matrix?

The current connector contract exposes scalar translational stiffness/damping. It does not claim a
six-DOF bushing, local-axis stiffness vector, nonlinear force-displacement curve, gap, or coupled
matrix. Those need a first-class typed connector with an explicit frame and positive-definiteness/
stability validation; a scalar spring should not be duplicated in coincident regions to imply
unimplemented coupling.

## How do I model a rotational spring or torsional damper?

The current closed connector set is translational. A rotational connector needs moment-per-angle
stiffness, moment-per-angular-velocity damping, an axis/frame, and rotational DOF support. Do not
enter `N*m/rad` into `translational_spring`, whose quantity is force per translation.

## How do I model a nonlinear spring, gap, stop, or one-way support?

A linear translational spring acts in tension and compression with constant tangent. Gaps, stops,
piecewise curves, preload, hysteresis, and unilateral response require a typed nonlinear connector or
contact law with state and convergence behavior. Do not approximate a gap by an enormous linear
stiffness that is active everywhere.

## How do I model frequency-dependent mount stiffness or damping?

The scalar connector values are constant. Viscoelastic or measured dynamic properties need a typed
frequency/temperature-dependent constitutive model with interpolation and causality/passivity checks.
Running a sweep while manually changing constants creates separate models and should retain the
parameter tuple explicitly if used as an approximation.

## How do springs combine in series or parallel?

For one-dimensional ideal springs, $k_{parallel}=\sum_i k_i$ and
$1/k_{series}=\sum_i1/k_i$. In a finite-element assembly, geometry, direction, reference points, and
couplings decide whether connectors are actually in series or parallel. Prefer declaring the physical
connectors and verifying modal/static response over collapsing a multiaxial assembly by arithmetic.

## How do I choose spring and dashpot coefficients for a target natural frequency?

For a one-DOF estimate, $\omega_n=\sqrt{k/m}$ and damping ratio
$\zeta=c/(2\sqrt{km})$. Use the effective participating mass and connector direction of the actual
model, then verify the assembled modal result. This estimate does not replace eigenanalysis when the
attachment excites several structural modes.

## Why does a connector make the model singular?

A translational connector only restrains the relative motion represented by its admitted basis. It
may leave rigid rotation or orthogonal translations free. Inspect connector target coverage and the
remaining rigid-body modes; add only the physical constraints the assembly really has.

## Why did a connector make the model too stiff?

Check for duplicate connectors, an unintended spring to ground, a rigid coupling that suppresses
face deformation, incorrect units, and several connectors acting in parallel on the same DOF. Inspect
strain energy by connector and run a limiting check with the coefficient reduced/increased by a known
factor.

## How do I verify connector forces and energy?

For a linear spring, verify $F=k\Delta u$ and stored energy
$U=\tfrac12k(\Delta u)^2$. For a dashpot, verify $F=c\Delta\dot u$ and nonnegative dissipated energy
under the stated sign convention. Check equal-and-opposite endpoint resultants and the actual relative
motion in the connector frame.

## What should I inspect in the study preview?

The preview should show both endpoint regions, connector kind, coefficient with units, ground status
when secondary is absent, and direction/frame when supported. A label anchored to only one face is
insufficient for a two-ended connector; use the study graph to inspect the complete relation.

## How do I put a spring and dashpot in parallel in RunMat?

Declare two connectors on the same named endpoint regions. They remain separate physical objects,
so static stiffness and transient dissipation can be reviewed independently:

```fea
connector isolator_stiffness {
    translational_spring {
        primary = geometry.machine.region.mount
        secondary = geometry.frame.region.mount
        stiffness = 180 kN/m
    }
}

connector isolator_damping {
    translational_dashpot {
        primary = geometry.machine.region.mount
        secondary = geometry.frame.region.mount
        damping = 1.8 kN*s/m
    }
}
```

The pair implements $\mathbf f=k\Delta\mathbf u+c\Delta\dot{\mathbf u}$ on the admitted connector
basis. Do not add coefficients before authoring unless the physical devices truly share the same
endpoints and direction; separate declarations retain the contribution of each device.

## How do I model several mounts under one machine?

Create one connector pair per physical mount and give every attachment its own CAD region. This
preserves lever arms and allows unequal reactions. A single spring attached to an all-mounts region
can preserve total translational stiffness while losing pitch, roll, and load-sharing behavior.
For each mount, check the actual endpoint displacement difference and equal-and-opposite connector
forces.

## How do I choose connector regions on CAD geometry?

Select the physical attachment footprints, not entire parts. A face region can be coupled to a
reference point when the reduced connector acts at a mount center; the coupling determines how face
motion becomes connector motion. Keep the coupling and connector as separate named objects. If two
mounts share one broad selector, RunMat cannot infer their separate locations or load paths.

## Which analyses use springs, dashpots, and foundations?

Springs contribute stiffness to static, buckling base-state, modal, harmonic, and transient
operators when those procedures admit the connector basis. Dashpots contribute damping to harmonic
and transient response but do not change an undamped static solution. Foundations contribute their
distributed stiffness only on the resolved target. A study must reject a connector when its
procedure, topology, or degrees of freedom cannot represent it; silently dropping a connector is
never a valid fallback.

## What connector results should I request?

Request relative displacement, relative velocity where applicable, connector force, spring energy,
and dashpot dissipation by connector identity. Also request support reactions and global energy
balance. A displacement contour alone cannot show whether a mount used the intended coefficient or
whether several mounts accidentally acted in parallel on the same projected entities.

## How do I model an equipment mount with stiffness and damping?

Use one translational spring and one translational dashpot per physical mount direction. Their forces
act in parallel: $F=k\Delta u+c\Delta\dot u$. Give each mount a separate region or reference point so
its load path and reaction remain identifiable. A single broad foundation is a different model.

Choose coefficients from test data or a stated reduction. For a supported mass $m$ and target
undamped frequency $f_n$, $k=(2\pi f_n)^2m$; for damping ratio $\zeta$,
$c=2\zeta\sqrt{km}$. Multi-axis coupling or rotational stiffness needs a typed matrix/bushing
contract and must not be approximated by unlabeled scalar springs.

## How do I model several isolators under one rigid machine?

Couple the machine attachment points only if the machine is intended to behave rigidly, then declare
each isolator separately at its actual location. Off-axis mount forces create moments and rocking
modes, so preserving only total vertical stiffness is insufficient. Retain mount coordinates,
direction frames, preload assumptions, and force by mount.

Compare rigid-body frequencies and mode shapes with a hand or reduced-order model. An accidental
duplicate selector can put two connectors in parallel and shift every mode.

## How do I decide between a spring, elastic foundation, tie, and contact?

Use a spring for a discrete compliant path, an elastic foundation for distributed traction
proportional to local displacement, a tie for zero relative motion, and contact for unilateral
opening/sliding. These are not interchangeable ways to stabilize a matrix. Select the representation
from the physical load path and expected relative motion.

For sensitivity work, vary the uncertain physical coefficient explicitly and compare reaction,
strain energy, modal frequency, and local stress away from idealized attachment singularities.

## How do connector coefficients scale in a symmetry model?

A discrete physical mount included in the sector retains its full coefficient. A smeared ring or
distributed support represented by a sector uses the coefficient associated with that sector's
physical share. Never divide stiffness merely because the mesh contains fewer nodes. State whether
the input is per mount, per unit area/length, or total assembly stiffness, then verify total stored
energy under a known rigid displacement.

Use springs for elastic compliance, dashpots for rate-dependent force, and a foundation for a
distributed support law. Use [structural constraints](./structural) for kinematic restrictions and
[interfaces](./interfaces) for contact or multiphysics transfer.
