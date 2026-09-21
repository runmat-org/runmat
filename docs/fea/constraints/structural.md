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
fixed supports.

## How do I avoid over-constraint and conflicting supports?

Use geometry-based free-body reasoning before adding constraints. For a 3D free solid, six rigid-body
modes must be removed, but removing more motion than the fixture transmits changes the solution.
RunMat detects duplicate identities, incompatible target topology, conflicting prescribed values,
invalid pairs/transforms, and a remaining operator nullspace. It cannot decide that a physically
incorrect fixture is wrong merely because its matrix is nonsingular; review reaction directions and
run sensitivity studies with plausible fixture compliance.

Further reading: [structural studies](../physics/structural), [structural loads](../loads/structural),
and [geometry regions](../geometry).
