# Connectors

Connectors add a localized stiffness or damping law to a structural model. They are distinct from
mesh elements, kinematic constraints, and contact. RunMat exposes four connector kinds whose names
state both the supported topology and the constitutive law.

| Kind | Target topology | Coefficient | Contribution |
| --- | --- | --- | --- |
| `axial_spring` | two vertices, each resolving to one structural node | `N/m` | stiffness |
| `axial_dashpot` | two vertices, each resolving to one structural node | `N*s/m` | damping |
| `isotropic_ground_spring` | one vertex resolving to one structural node | `N/m` | stiffness |
| `surface_normal_foundation` | one face region | `N/m^3` | stiffness integrated over area |

Targets are named geometry regions, not mesh-node numbers. Region realization maps persistent CAD
entities to canonical solver nodes and faces. A missing, stale, ambiguous, or dimensionally invalid
target is an error; a connector is never silently omitted.

## Axial spring equation and `.fea` syntax

For endpoint positions $\mathbf x_1,\mathbf x_2$, RunMat forms the undeformed unit axis

$$
\mathbf a=\frac{\mathbf x_2-\mathbf x_1}{\|\mathbf x_2-\mathbf x_1\|}.
$$

With endpoint displacements $\mathbf u_1,\mathbf u_2$, the extension, axial force, and stored energy
are

$$
\delta=\mathbf a^T(\mathbf u_2-\mathbf u_1),\qquad
F=k\delta,\qquad U=\frac12k\delta^2.
$$

The two endpoints receive equal-and-opposite forces $\mp F\mathbf a$. Motion perpendicular to
$\mathbf a$ does not stretch the small-displacement connector.

Save `mount_pin` and `frame_pin` as named vertex selections on the CAD asset using geometry selection tools.

```fea
geometry bracket {
    file = "./geometry/bracket.step"
    units = mm
    region mount_pin : vertex { select name "mount_pin"; expect exactly_one }
    region frame_pin : vertex { select name "frame_pin"; expect exactly_one }
}

model supported_bracket {
    connector mount_stiffness {
        axial_spring {
            primary = geometry.bracket.region.mount_pin
            secondary = geometry.bracket.region.frame_pin
            stiffness = 220000 N/m
        }
    }
}
```

Both endpoints are required. To model a connection to a fixed environment, use
`isotropic_ground_spring`; do not omit an endpoint or create an unmeshed dummy part.

## Axial dashpot equation and `.fea` syntax

An axial dashpot uses the same geometric axis and the relative axial velocity

$$
\dot\delta=\mathbf a^T(\dot{\mathbf u}_2-\dot{\mathbf u}_1),\qquad F=c\dot\delta.
$$

It contributes to the damping operator. It adds no static stiffness and stores no elastic energy.
During a transient solve, RunMat recovers the signed relative axial velocity, signed axial force,
global resultant-force vector, and nonnegative instantaneous dissipated power
$P=c\dot\delta^2$ at every accepted state. The final connector force and stored-energy fields are
published with the run, and connector-target force histories use those same recovered states.

```fea
model supported_bracket {
    connector mount_damping {
        axial_dashpot {
            primary = geometry.bracket.region.mount_pin
            secondary = geometry.bracket.region.frame_pin
            damping = 1400 N*s/m
        }
    }
}
```

To put a spring and dashpot in parallel, declare both with the same two endpoint regions. Keeping
them as separate named objects preserves their coefficients and result identities.

## Point spring to ground

An isotropic ground spring resists all three translations at one point:

$$
\mathbf f=-k\mathbf u,\qquad U=\frac12k\mathbf u^T\mathbf u.
$$

```fea
model supported_bracket {
    connector grounded_mount {
        isotropic_ground_spring {
            target = geometry.bracket.region.mount_pin
            stiffness = 75000 N/m
        }
    }
}
```

This is three equal uncoupled Cartesian springs, not an axial spring and not a six-degree-of-freedom
bushing. The target must resolve to exactly one structural node.

## Surface-normal elastic foundation

A surface-normal foundation implements a Winkler law. At each quadrature point on the selected
face, with outward unit normal $\mathbf n$ and interpolated displacement $\mathbf u$, the normal
displacement and traction are

$$
u_n=\mathbf n^T\mathbf u,\qquad \mathbf t=-k_w u_n\mathbf n,
$$

where $k_w$ has units `N/m^3`. RunMat assembles

$$
\mathbf K_f=\int_A k_w\mathbf N^T\mathbf n\mathbf n^T\mathbf N\,dA.
$$

The physical face area therefore controls total stiffness; refining the mesh does not multiply the
foundation stiffness.

Save `grout_bearing_face` as a named face selection on the CAD asset.

```fea
geometry base {
    file = "./geometry/base.step"
    units = mm
    region grout_bearing_face : face { select name "grout_bearing_face"; expect exactly_one }
}

model mounted_base {
    connector grout_support {
        surface_normal_foundation {
            target = geometry.base.region.grout_bearing_face
            modulus = 35000000 N/m^3
        }
    }
}
```

A coefficient in `N/m` is a discrete spring stiffness and is dimensionally invalid here. Convert
soil, grout, or bedding data to the area-distributed modulus required by the physical model rather
than dividing a total spring stiffness by the mesh-node count.

## Choosing the correct relationship

Use an axial spring or dashpot for a discrete two-point load path. Use an isotropic ground spring
when one point is elastically supported in all translations. Use a surface-normal foundation for a
distributed support pressure proportional to normal displacement. Use a tie when relative motion
must be zero, and contact when separation or sliding is part of the model.

These connector kinds do not imply any of the following:

- a directional point spring to ground;
- a local three-axis or six-degree-of-freedom bushing;
- rotational stiffness or damping;
- preload, a gap, a stop, unilateral behavior, or a nonlinear force curve;
- frequency- or temperature-dependent viscoelastic behavior.

Those behaviors require their own typed constitutive contracts. Supplying a coefficient with the
wrong units or duplicating coincident scalar connectors does not create them.

## How do I model an equipment mount with stiffness and damping?

Represent one scalar mount by an axial spring and axial dashpot between the same two point regions.
For supported mass $m$, target undamped frequency $f_n$, and damping ratio $\zeta$, an initial
reduction is $k=(2\pi f_n)^2m$ and $c=2\zeta\sqrt{km}$. The participating mass must be the mass seen
along the connector axis, not necessarily the assembly's total mass.

```fea
model isolated_equipment {
    connector mount_elasticity {
        axial_spring {
            primary = geometry.equipment.region.mount_center
            secondary = geometry.frame.region.mount_center
            stiffness = 180000 N/m
        }
    }

    connector mount_dissipation {
        axial_dashpot {
            primary = geometry.equipment.region.mount_center
            secondary = geometry.frame.region.mount_center
            damping = 1800 N*s/m
        }
    }
}
```

Measured elastomer data can depend on preload, temperature, amplitude, and frequency. A constant
linear coefficient is appropriate only when the intended operating range supports that reduction.

## How do I model a foundation under a plate or machine base?

Select the physical bearing face and use `surface_normal_foundation`. The input is the pressure per
normal displacement, not the total vertical stiffness of the machine. For a flat face of area $A$
under uniform normal motion, total tangent stiffness is $k_wA$.

## How do I attach a lumped component to a structure?

First use a structural coupling to relate the attachment face to a reference point. Place the
lumped mass and inertia at that reference point, then connect physical point locations as required.
The coupling defines how face motion reaches the reduced point; the connector defines compliance.
Do not distribute a point mass over arbitrary face nodes without preserving center of mass and
rotational inertia.

## How do axial springs combine in series or parallel?

Ideal collinear springs use $k_\mathrm{parallel}=\sum_i k_i$ and
$1/k_\mathrm{series}=\sum_i1/k_i$. In a three-dimensional assembly, endpoint positions and axes
determine whether that reduction applies. Declaring the physical connectors retains load paths and
individual results and is preferable when geometry matters.

## How do I choose a connector coefficient from test data?

Use the slope of the measured force-displacement or force-velocity curve over the stated operating
range, convert it to `N/m` or `N*s/m`, and record the test conditions. Compare a RunMat single-
connector calculation with the measured slope before placing the connector in a larger assembly.
Do not infer a Winkler modulus from a point test without an explicit effective-area model.

## How do I model a directional spring to ground?

`isotropic_ground_spring` acts equally in global $x$, $y$, and $z$. It cannot represent one selected
direction. A directional ground spring needs a typed axis or coordinate frame and is not part of
the current closed connector set. An axial spring may provide directional compliance only when its
second endpoint is a real modeled point.

## How do I model a six-degree-of-freedom bushing?

The four connector kinds do not accept translational-rotational coupling matrices. A bushing needs
a local frame, a dimensionally typed $6\times6$ stiffness or damping operator, symmetry/passivity
validation, and rotational DOFs at both endpoints. Do not approximate cross-coupling by stacking
unlabeled scalar connectors.

## How do I model a rotational spring or torsional damper?

Rotational connectors require moment-per-angle or moment-per-angular-velocity coefficients plus a
defined axis. `N*m/rad` is invalid for every connector on this page. Use a formulation with a typed
rotational connector when that capability is available.

## How do I model a nonlinear spring, gap, or stop?

The current spring laws are linear and bilateral. A gap, stop, one-way support, preload, hysteresis,
or piecewise force curve needs state, tangent updates, event handling, and nonlinear convergence
evidence. A very large linear stiffness that remains active in both directions is not a gap model.

## How do I model frequency-dependent mount properties?

Constant `stiffness` and `damping` values do not encode viscoelastic frequency dependence. A sweep
of manually changed constants is a family of linear approximations; retain the parameter tuple and
test provenance for every case. A causal frequency-dependent material or connector needs an
explicit typed constitutive model.

## Can a face be an axial connector endpoint?

No. An axial endpoint must resolve to one vertex and one canonical structural node. If a physical
mount occupies a face, introduce the appropriate distributing or rigid coupling to a point and
connect that point. This keeps face-to-point kinematics separate from connector compliance.

## What happens when both endpoints resolve to the same node?

RunMat rejects the connector. Its axis would have zero length and no physical direction. This often
indicates that two CAD aliases collapsed onto one persistent vertex or that a conformal interface
merged endpoints that were intended to remain distinct.

## How are shell and solid foundation faces integrated?

Explicit linear shell faces use physical triangle quadrature. Solid boundary faces, including
quadratic faces, use the canonical analysis projection and its physical boundary quadrature. Mixed
models select the owning surface representation rather than integrating the same shared face twice.

## How do I check connector mesh convergence?

For axial and point-ground connectors, refine the surrounding structure while preserving the same
persistent endpoint identity. For foundations, refine the selected face and confirm that integrated
reaction and foundation energy converge without growing in proportion to node count. Also inspect
nearby continuum stress separately from the connector resultant.

## Which procedures use dashpot damping?

Transient and harmonic procedures use the damping operator when their admitted formulation includes
connectors. A static solve has zero velocity and therefore no dashpot force. Modal procedures must
state whether damping participates in eigenvalue extraction rather than implying it from the mere
presence of a dashpot.

## What connector results should I request?

Request connector resultant force and stored energy as connector-target outputs. Transient run
results also retain each dashpot's relative axial velocity, axial force, and instantaneous
dissipated power in the typed connector-response history. Pair those with support reactions and
whole-model energy balance. A displacement contour alone cannot prove that a connector used the
intended coefficient or target.

For a harmonic solve using the peak-amplitude convention, RunMat publishes the connector resultant
as a complex force phasor. Springs and foundations use the complex displacement directly. For a
dashpot, the velocity phasor is $\hat{\mathbf v}=i\omega\hat{\mathbf u}$, so its force is
$\hat F=i\omega c\,\mathbf a^T(\hat{\mathbf u}_2-\hat{\mathbf u}_1)$. Stored energy and
dissipated power are quadratic quantities rather than phasors; they are deliberately not labeled
as complex force-like fields. Derive the cycle metric required by the study from the admitted
force and displacement convention.

## What should the Desktop preview show?

The study preview should identify every connector, both endpoints for axial devices, the foundation
face, coefficient and units, and the realized axis or normal basis. At lower zoom levels it may
simplify labels, but it must preserve kind, location, and load-path identity. The study graph should
link the declaration to its geometry regions and result fields.

## Modeling several mounts

Give every physical mount its own point regions and connector declarations. Preserving mount
locations retains the moment arms, load sharing, pitch and roll modes, and force result for each
mount. Replacing several mounts with one connector on a broad region changes the model even when
the sum of the scalar coefficients is unchanged.

For a first estimate of a one-degree-of-freedom supported mass,
$\omega_n=\sqrt{k/m}$ and $\zeta=c/(2\sqrt{km})$. Use the participating mass along the connector
axis, then verify the assembled modes and transient or harmonic response.

## Symmetry and sector models

A discrete mount that is physically present in the modeled sector retains its full coefficient. A
smeared support uses the coefficient belonging to the represented area or circumference. State
whether input data is per mount, per unit area, or for the full assembly. Do not scale a connector
because the mesh happens to contain fewer nodes.

## Diagnosing singular or over-stiff models

An axial connector restrains only relative motion along one axis. Orthogonal translations and rigid
rotations remain free unless other physical supports restrain them. If the model is singular,
inspect its remaining rigid-body modes rather than increasing the connector coefficient.

If the assembly is too stiff, check for duplicate connector declarations, an unintended ground
spring, a rigid coupling that suppresses attachment-face deformation, incorrect units, and multiple
devices acting in parallel. Vary one coefficient by a known factor and compare connector force,
stored energy, reactions, and modal frequency.

## Verification checks

For each axial spring, compare reported extension and force with $F=k\delta$, check
equal-and-opposite endpoint resultants, and compare stored energy with $k\delta^2/2$. For each
ground spring, verify the three force components against $-k\mathbf u$. For each foundation,
check integrated reaction and $k_w\int_Au_n^2/2\,dA$ under a simple prescribed displacement.

A rigid transverse motion of an axial connector must produce zero axial energy. A uniform normal
translation of a flat foundation must produce total tangent stiffness $k_w A$, independent of mesh
density. These limiting cases catch target, direction, unit, and quadrature errors before a complex
study obscures them.

Spring stiffness contributes to static, preload, modal, harmonic, and transient operators when the
procedure admits structural connectors. Dashpot damping affects velocity-dependent procedures but
not a static solution. A procedure that cannot execute a connector kind must reject the study.

Use [structural constraints](./structural) for prescribed kinematics and
[interfaces](./interfaces) for contact and multiphysics transfer.
