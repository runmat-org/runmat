# Structural constraints

Structural constraints remove rigid-body motion, prescribe displacement or motion, relate degrees
of freedom, or join nonconforming regions. Use the least restrictive condition justified by the
physical fixture; an over-constrained model can look numerically stable while reporting false load
paths and stress.

## How RunMat applies structural constraints

The unconstrained algebraic system is $\mathsf K\mathbf u=\mathbf f$ (or its dynamic equivalent).
Essential values partition the DOFs into free and prescribed sets:

$$
\mathsf K_{ff}\mathbf u_f=\mathbf f_f-\mathsf K_{fc}\bar{\mathbf u}_c.
$$

General relations are written $\mathsf G\mathbf u=\mathbf g$ and enforced by the formulation's
declared elimination or multiplier path. Reactions are recovered from the full physical balance,
not from the reduced right-hand side. RunMat proves or rejects nullspace elimination on the actual
constraint space; it does not add hidden weak springs.

| Kind | Required fields | Algebraic meaning |
| --- | --- | --- |
| `structural.fixed` | `target` | Prescribe all structural DOFs on the target to zero |
| `structural.symmetry` | `target` | Zero normal displacement and compatible tangential rotations on a symmetry face |
| `structural.antisymmetry` | `target` | Complementary antisymmetry restriction |
| `structural.prescribed` | `target`, `frame`, one or more of `ux..rz` | Prescribe selected translation/rotation components |
| `structural.prescribed_motion` | `target`, `translations` | Reconstruct translation histories from typed acceleration signals |
| `structural.normal_displacement` | `target`, `value` | Enforce $\mathbf u\cdot\mathbf n=\bar u_n$ |
| `structural.rigid_coupling` | `target`, `reference_point_id`, `position` | $\mathbf u(\mathbf x)=\mathbf u_r+\boldsymbol\theta_r\times(\mathbf x-\mathbf x_r)$ |
| `structural.distributing_coupling` | same as rigid coupling | Transfer reference motion/resultants without rigidizing all target DOFs |
| `structural.tie` | `primary`, `secondary`, optional `tolerance` | Enforce paired displacement continuity |
| `structural.periodic` | pair plus optional transform | Enforce displacement relation across a periodic map |
| `structural.equation` | typed equation terms and right-hand side | Enforce $\sum_i a_i u_i=b$ |

## Fixed and component-wise supports

```fea
constraint encastre {
    structural.fixed { target = geometry.bracket.region.mounting_face }
}

constraint guided_edge {
    structural.prescribed {
        target = geometry.bracket.region.guide_edge
        frame = global
        uy = 0 mm
        uz = 0 mm
    }
}
```

`structural.prescribed` requires an explicit frame and at least one of `ux`, `uy`, `uz`, `rx`,
`ry`, or `rz`. Translational values carry length units; rotations are dimensionless angles.

The accepted frame values are the closed enums `global`, `material`, `element`, and
`boundary_normal_tangent`. This block does not accept an inline Cartesian/cylindrical frame
definition or an arbitrary frame name. Region references resolve from CAD declarations; mesh node
and element IDs are not valid targets. Every structural constraint target has one-or-more
cardinality, while the region declaration's own `expect` controls whether its CAD selector must
resolve exactly once or may resolve several entities.

Prescribed values may be constants or compatible typed signals where the constraint kind admits
them. `structural.prescribed_motion` accepts records containing exactly `dof`, `acceleration`,
`initial_displacement`, `initial_velocity`, and
`reconstruction = piecewise_linear_acceleration_exact`. The acceleration signal must be a real
global vector time history with piecewise-linear interpolation and reject extrapolation; those
requirements make the reconstructed displacement/velocity deterministic at signal breakpoints.

## How do I model a fully fixed or encastre support?

Use `structural.fixed` when the fixture prevents every active translation and rotation supported by
the target formulation:

```fea
constraint encastre {
    structural.fixed { target = geometry.bracket.region.clamped_face }
}
```

The target is normally a face for a solid model, an edge for a shell boundary, or the compatible
endpoint region for a beam. A fixed face is much stiffer than a bolt, bearing, pin, or compliant
fixture; use component-wise constraints, coupling, connectors, or contact when those motions matter.

## How do I model a pinned support?

A pin removes translation at its center while permitting the intended rotation. Do not use
`structural.fixed` on the whole bore: that also suppresses rotation and distorts bearing stress. Use
a reference-point/distributing coupling for the bore and prescribe only the supported translations,
or model pin contact when clearance, separation, and load direction matter. The exact choice depends
on whether pin deformation and contact pressure are part of the result.

## How do I model a roller support?

Use `structural.normal_displacement` with zero value on the support surface when it may slide
tangentially but cannot penetrate normally:

```fea
constraint lower_roller {
    structural.normal_displacement {
        target = geometry.beam.region.roller_surface
        value = 0 mm
    }
}
```

This is bilateral: it also prevents separation. Use frictionless contact when the roller can lose
contact or when contact pressure is needed.

## How do I model a simply supported beam or plate?

Represent the actual pin/roller kinematics: one support removes the required translations and the
other removes only the transverse/normal translation, leaving axial expansion free. For shells or
beams, preserve the rotations that a simple support permits. Confirm the remaining rigid-body modes
with the constraint/nullspace report rather than fixing extra DOFs until the matrix solves.

## How do I constrain rigid-body motion without overconstraining strain?

Use a minimal physical 3-2-1-style component constraint set, a reference-point coupling, or the
actual supports. For a 3D free solid, six independent rigid modes must be removed, but six scalar
rows are sufficient only when they are independent. RunMat reports the remaining nullspace; it does
not add weak springs or silently pin arbitrary nodes.

## How do I use symmetry in a structural FEA model?

Use `structural.symmetry` only on a geometric and loading symmetry plane. It removes displacement
normal to the plane and the complementary rotational components required by the formulation:

```fea
constraint midplane_symmetry {
    structural.symmetry { target = geometry.half_model.region.cut_face }
}
```

Loads, material orientation, contact, and downstream state must share the same symmetry. Reaction
resultants from the reduced model must be scaled/interpreted for the represented fraction.

## How do I model structural antisymmetry?

Use `structural.antisymmetry` when the solution changes sign across the cut plane rather than
mirroring it:

```fea
constraint antisymmetric_cut {
    structural.antisymmetry { target = geometry.half_model.region.cut_face }
}
```

Antisymmetry is common for particular modal subspaces and antisymmetric load cases. It is not an
alternative spelling for symmetry; choosing the wrong one removes the modes you intended to solve.

## How do I prescribe only selected displacement or rotation components?

Use `structural.prescribed` with an explicit frame. Omit every component that should remain free:

```fea
constraint sliding_guide {
    structural.prescribed {
        target = geometry.rail.region.guide
        frame = global
        uy = 0 mm
        uz = 0 mm
        rx = 0 rad
        rz = 0 rad
    }
}
```

At least one component is required. `ux`, `uy`, and `uz` carry length; `rx`, `ry`, and `rz` carry
dimensionless angle. The frame is part of the constraint identity—changing it changes the problem.

## How do I constrain motion in a local coordinate system?

Choose one of the supported frame enums: `material`, `element`, or `boundary_normal_tangent` (or
`global`). The current constraint syntax does not declare a new coordinate frame and does not accept
a user-chosen frame identifier. A cylindrical radial guide is not correctly represented by setting
one global Cartesian component to zero over a curved surface. Use
`structural.normal_displacement` when the desired direction is the local surface normal. If the
required basis cannot be represented by the four supported enums, that constraint is currently
unsupported rather than inferred from CAD axes.

## How do I prescribe a nonzero displacement?

Set the selected component to a typed length value:

```fea
constraint imposed_extension {
    structural.prescribed {
        target = geometry.coupon.region.pulled_end
        frame = global
        ux = 0.5 mm
    }
}
```

Keep the other components omitted if they remain free. A nonzero prescribed displacement produces
reaction work; it is not a force load and should not also be paired with an incompatible force on
the same constrained DOF.

## How do I apply a time-varying prescribed displacement?

Use a compatible typed prescribed-value signal where the chosen procedure admits it. If the source
data are acceleration, use `structural.prescribed_motion` so RunMat owns deterministic integration
and initial conditions. Do not numerically integrate acceleration in an undocumented preprocessing
step and then lose its reconstruction evidence.

## How do I prescribe base acceleration or support motion?

Use `structural.prescribed_motion` when acceleration history drives one or more translation DOFs.
Each entry states the exact integration/reconstruction contract:

```fea
signal base_accel : acceleration over time {
    samples = [
        { at = 0 s, value = [0 m/s^2, 0 m/s^2, 0 m/s^2] },
        { at = 10 ms, value = [0 m/s^2, 0 m/s^2, 20 m/s^2] },
        { at = 20 ms, value = [0 m/s^2, 0 m/s^2, 0 m/s^2] },
    ]
    interpolation = linear
    extrapolation = reject
    frame = global
}

constraint shaker_motion {
    structural.prescribed_motion {
        target = geometry.fixture.region.mount
        translations = [{
            dof = uz,
            acceleration = signal.base_accel,
            initial_displacement = 0 mm,
            initial_velocity = 0 m/s,
            reconstruction = piecewise_linear_acceleration_exact,
        }]
    }
}
```

The signal must be a real global vector acceleration time history with linear interpolation and
reject extrapolation. Initial displacement/velocity determine the integration constants. For a
response-spectrum or random-vibration base input, use the corresponding typed spectrum contract
rather than converting it to an arbitrary force history.

## How do I prescribe displacement normal to a curved surface?

Use `structural.normal_displacement` to constrain only $\mathbf u\cdot\mathbf n$:

```fea
constraint roller_surface {
    structural.normal_displacement {
        target = geometry.part.region.roller_contact
        value = 0 mm
    }
}
```

This is suitable for an ideal frictionless roller/contact surrogate when separation and contact
status do not need to be solved. The region must provide a well-defined surface normal; use contact
when unilateral opening/closing matters.

## How do I model frictionless contact without bonding two faces?

Use a `contact.frictionless` interface, not `structural.tie` or a zero normal-displacement constraint.
Contact enforces nonpenetration only while active and permits opening/sliding. See
[interfaces](./interfaces#how-do-i-model-frictionless-structural-contact) for the paired-region
declaration and contact evidence to inspect.

## How do I model bonded contact?

Use `structural.tie` for perfect displacement continuity when opening and slip are impossible. Use a
finite-stiffness interface or cohesive law when adhesive compliance, damage, or separation matters.
A tie is an exact kinematic relation, not a very stiff adhesive.

## Coupling and ties

```fea
constraint bolt_head {
    structural.rigid_coupling {
        target = geometry.bracket.region.bolt_head
        reference_point_id = "bolt_axis"
        position = [20 mm, 10 mm, 4 mm]
    }
}

constraint bonded_faces {
    structural.tie {
        primary = geometry.assembly.region.face_a
        secondary = geometry.assembly.region.face_b
        tolerance = 0.02 mm
    }
}
```

Omit `tolerance` only when topology is matching. Primary and secondary roles are intentional and
remain visible in evidence and the study preview.

## Should I use a rigid coupling or a distributing coupling?

`structural.rigid_coupling` makes the target move as a rigid region relative to the reference point.
Use it for a genuinely rigid spider, tool, or idealized attachment. `structural.distributing_coupling`
transfers reference-point motion/resultants while allowing target deformation:

```fea
constraint distributed_bolt_load {
    structural.distributing_coupling {
        target = geometry.bracket.region.bolt_hole
        reference_point_id = "bolt_center"
        position = [20 mm, 10 mm, 4 mm]
    }
}
```

Both require a stable reference-point id and position. A rigid coupling can artificially stiffen a
large face; prefer distributing coupling when the connected surface should still deform.

## How do I model a bolt, weld, or rigid attachment without detailed geometry?

Choose the abstraction by the result you need. A distributing coupling plus connector can transfer
bolt resultants without rigidizing the entire flange. A tie represents a perfectly bonded weld only
when local weld stress is outside scope. Springs/dashpots represent known compliance. Detailed
contact/preload is required for separation, slip, bearing, or bolt-load distribution. Do not call a
fixed face a bolt model.

## How do I tie two nonmatching structural surfaces?

Use `structural.tie` for permanent displacement continuity without separation or slip:

```fea
constraint bonded_joint {
    structural.tie {
        primary = geometry.assembly.region.coarse_face
        secondary = geometry.assembly.region.fine_face
        tolerance = 0.02 mm
    }
}
```

The projection must cover the intended paired area. `tolerance` is a geometric matching tolerance,
not adhesive thickness or compliance. Use contact or a connector/interface law when opening, slip,
friction, or finite stiffness matters.

## Periodicity

```fea
constraint repeating_cell {
    structural.periodic {
        primary = geometry.cell.region.x_min
        secondary = geometry.cell.region.x_max
        translation = [10 mm, 0 mm, 0 mm]
        tolerance = 1 um
    }
}
```

A periodic transform is either absent, one translation, or a complete rotation described by
`rotation_origin`, `rotation_axis`, and `rotation_angle`. Partial transforms are rejected.

## How do I model rotational periodicity or cyclic symmetry?

Specify all three rotation fields together:

```fea
constraint cyclic_sector {
    structural.periodic {
        primary = geometry.sector.region.low_angle_face
        secondary = geometry.sector.region.high_angle_face
        rotation_origin = [0 mm, 0 mm, 0 mm]
        rotation_axis = [0, 0, 1]
        rotation_angle = 15 deg
        tolerance = 1 um
    }
}
```

The transform maps primary to secondary. A mismatched direction or angle produces wrong pairing even
when faces look coincident in a viewer; inspect the resolved pair and coverage evidence.

## How do I model translational periodicity in a unit cell?

Use `structural.periodic` with the primary-to-secondary translation. Ordinary periodicity enforces
the admitted equal mapped displacement relation. A homogenization problem with macroscopic strain
needs an affine displacement relation or constraint equations in addition to periodic fluctuations;
plain equality alone does not impose that strain state.

## How do I model sector harmonics or phase-shifted cyclic symmetry?

The structural periodic kind shown here owns geometric periodicity but does not claim a complex
phase-shifted structural sector contract. Do not insert phase into `rotation_angle`: that field is
the geometric sector transform. A harmonic-index formulation needs a distinct typed phase relation
and complex structural space.

## How do I write a multi-point structural constraint equation?

Use `structural.equation` for a linear relation between translation DOFs:

```fea
constraint equal_vertical_motion {
    structural.equation {
        terms = [
            { target = geometry.frame.region.point_a, dof = uy, coefficient = 1 },
            { target = geometry.frame.region.point_b, dof = uy, coefficient = -1 },
        ]
        right_hand_side = 0 mm
    }
}
```

The relation is $\sum_i a_i u_i=b$. Each term has exactly `target`, `dof`, and dimensionless
`coefficient`; the right-hand side has length. Equations can create redundant or contradictory rows,
so review rank/nullspace diagnostics and avoid duplicating relations already imposed by ties or
fixed supports. The current equation contract accepts only `ux`, `uy`, and `uz`; rotational equation
terms (`rx`, `ry`, `rz`) are unsupported and must not be approximated with dimensionally ambiguous
translation coefficients.

## Why is my structural model singular?

Inspect the reported nullspace by connected component. Typical causes are a free body, a connector
that restrains only one direction, an open contact pair, a disconnected part, or a coupling whose
target resolved to no entities. Add only physically present restraints. A singular model is not
repaired by hiding weak springs in solver options.

## Why are support stresses mesh dependent?

An abrupt fixed boundary can create a local stress singularity or an unrealistically stiff load
path. Model fixture compliance/contact when it matters, compare reaction resultants, and evaluate
stress away from the idealized edge. Convergence of displacement or total reaction does not imply
convergence of peak stress at a singular support.

## How do I verify support reactions and constraint equations?

Check resultant force and moment about a stated origin, reactions by support and component,
constraint residuals, equation rank, and energy/work balance. For symmetry or fractional models,
apply the correct geometric scaling to reported totals. For coupled reference points, include both
force and moment transfer.

## What should I inspect in the Desktop study preview?

Confirm each annotation is anchored to the intended CAD region, arrows/components use the expected
frame, paired constraints show both sides, reference points are in the right position, and no hidden
face was captured by a broad selector. Zoom-dependent annotation simplification changes only the
display; the study graph and checked presentation retain the full typed constraint.

## How do I avoid over-constraint and conflicting supports?

Use geometry-based free-body reasoning before adding constraints. For a 3D free solid, six rigid-body
modes must be removed, but removing more motion than the fixture transmits changes the solution.
RunMat detects duplicate identities, incompatible target topology, conflicting prescribed values,
invalid pairs/transforms, and a remaining operator nullspace. It cannot decide that a physically
incorrect fixture is wrong merely because its matrix is nonsingular; review reaction directions and
run sensitivity studies with plausible fixture compliance.

## How do I constrain a bolt pattern or mounting-foot assembly?

Create one named region per physically distinct bolt group or foot. Use fixed support only when the
attachment truly suppresses every admitted displacement and rotation. When load distribution,
fixture compliance, or bolt-line motion matters, couple each footprint to its own reference point
and connect that point to the fixture model. Combining all feet into one rigid region can suppress
chassis distortion and erase the moments created by their spacing.

## How do I model a cylindrical bearing support?

A radial support normally blocks radial motion while allowing axial translation and rotation; a
thrust bearing also restrains an axial component. The current source surface cannot declare a
cylindrical frame. Use `structural.normal_displacement` on a cylindrical bearing surface only when
its local-normal bilateral idealization matches the support. Otherwise use a contact/coupling
formulation that explicitly owns cylindrical kinematics. Do not label a global component as
“radial.” Verify reactions decomposed about the actual bearing axis and check that allowed axial and
circumferential motions remain free.

## How do I prescribe a rotation without fixing translation?

Use `structural.prescribed` with rotational degrees of freedom selected in an explicit frame. The
target formulation must own rotations; solid displacement-only nodes cannot accept an independent
rotation without a reference-point or coupling construction. Do not approximate a prescribed angle
with arbitrary tangential displacements unless rigid-body kinematics and rotation center are
explicitly derived.

## How do I compare fixed, pinned, and elastic support assumptions?

Author each support model as a separate named model or sweepable study case, then compare global
deflection, reaction distribution, strain energy, modal frequencies, and stresses away from support
singularities. Fixed and pinned supports are kinematic limits; an elastic mount belongs in a
connector. Keeping variants explicit prevents a solver setting from silently changing the fixture.

## What structural constraint evidence should be archived?

Retain persistent target identity, projected entity and DOF counts, constraint rank, redundant-row
diagnostics, reference frames, pair coverage and transforms, reaction force and moment by constraint,
and normalized residual. For moving constraints, also retain signal identity and support work. This
evidence explains why geometrically similar runs can have different load paths.

## How do I choose the minimum constraints for a 3D solid?

A free 3D solid has six rigid-body modes. Remove only the motions excluded by the physical fixture.
One common minimal stabilization pattern fixes three translations at one point, two at a second
point, and one at a third non-collinear point, but that pattern is valid only when those locations
represent the fixture. A face clamp is not a generic numerical remedy: it changes local strain and
load path.

Before accepting the model, inspect the operator nullspace and the reaction distribution. A
zero-pivot message identifies missing or dependent equations; it does not tell you which support is
physical. For an intentionally free-free modal analysis, retain the six near-zero rigid modes rather
than adding supports.

## How do I constrain a shell or beam model without suppressing rotations?

Shell and beam nodes can own rotational degrees of freedom that solid nodes do not. State the exact
translation and rotation components with `structural.prescribed`; do not use `structural.fixed` when
the physical joint is pinned or releases bending. A hinge commonly restrains translations while
leaving rotation about its axis free.

```fea
constraint beam_hinge {
    structural.prescribed {
        target = geometry.frame.region.hinge
        frame = global
        ux = 0 mm
        uy = 0 mm
        uz = 0 mm
        rx = 0 rad
        ry = 0 rad
    }
}
```

The omitted `rz` remains free in this example. Confirm that the selected element basis owns
those rotations and inspect hinge moment; a solid target must reject independent rotational DOFs.

## How do I model a remote support or remote load point?

Create a reference point and use `structural.rigid_coupling` when the attached region should move as
a rigid section, or `structural.distributing_coupling` when the resultant should be distributed
without rigidifying all in-plane deformation. Apply the support or load to the reference point, and
retain the coupling's force and moment transfer.

Do not couple an entire flexible flange merely because the remote point is convenient. Compare
coupled-area deformation, reaction moment, and strain energy with a resolved fixture or a smaller
attachment patch.

## How do I constrain thermal expansion without creating artificial thermal stress?

Use a kinematic scheme that removes rigid motion while allowing the intended expansion: for example,
one datum point, one sliding direction at a second point, and one plane-normal restraint at a third.
A fully fixed face creates real restraint stress only when the assembly is actually clamped that
way. For symmetry sectors, constrain the symmetry-normal displacement and leave compatible
tangential expansion free.

Run a uniform-temperature-change check. A freely expanding homogeneous body should recover nearly
zero stress; nonzero reaction or strain energy identifies unintended restraint, mismatched thermal
expansion, or an inconsistent temperature reference.

## How do I author a three-direction guide without clamping the part?

Name separate CAD regions for the physical guide surfaces and prescribe only the components each
surface transmits:

```fea
constraint axial_datum {
    structural.prescribed {
        target = geometry.carriage.region.datum
        frame = global
        ux = 0 mm
        uy = 0 mm
        uz = 0 mm
    }
}

constraint lateral_guide {
    structural.prescribed {
        target = geometry.carriage.region.lateral_guide
        frame = global
        uy = 0 mm
        uz = 0 mm
    }
}

constraint vertical_roller {
    structural.normal_displacement {
        target = geometry.carriage.region.roller_track
        value = 0 mm
    }
}
```

This is a bilateral idealization: the normal-displacement roller cannot lift off. Check the
constraint-rank report, then verify that axial thermal expansion produces the intended free motion
and that reactions appear only in transmitted directions. If separation is physical, replace the
bilateral normal condition with frictionless contact.

Further reading: [structural studies](../physics/structural), [structural loads](../loads/structural),
and [geometry regions](../geometry).
