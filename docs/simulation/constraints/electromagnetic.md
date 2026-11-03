# Electromagnetic constraints

Electric-potential constraints remove the additive nullspace in
$\nabla\cdot(\sigma\nabla V)=0$. Magnetic vector-potential constraints/gauges remove the gradient
nullspace of curl–curl formulations. Periodic pairs relate fields across a geometric map; a complex
phase-shifted pair enforces $\mathbf A_s=e^{i\phi}\mathcal T(\mathbf A_p)$.

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `electrical.potential` | `target`, `value` | Prescribe electric potential |
| `electrical.equipotential` | `target` | One unknown potential across a conductor region |
| `electrical.floating_conductor` | `target` | Isolated conductor without an invented ground |
| `electrical.integral_current` | `target`, `value` | Constrain integrated normal current |
| `electrical.potential_reference` | `target` | Establish the electrical reference |
| `electrical.periodic` | periodic pair/transform | Pair periodic electric regions |
| `magnetic.vector_potential` | `target`, `value` | Prescribe magnetic vector potential |
| `magnetic.vector_potential_ground` | `target` | Ground tangential vector-potential DOFs |
| `magnetic.coulomb_gauge` | `target` | Apply $\nabla\cdot\mathbf A=0$ gauge intent |
| `magnetic.periodic` | periodic pair/transform | Pair periodic magnetic regions |
| `magnetic.phase_shifted_periodic` | periodic pair/transform plus `phase` | Complex phase relation across the pair |

All targets and pair members are CAD region references. `value` on the current
`magnetic.vector_potential` essential constraint is a scalar quantity in T·m; this declaration does
not accept a three-component vector or a coordinate frame. That is the exact present source
contract, not a shorthand for an arbitrary vector boundary. The magnetic initial-condition
contract is different and does accept a framed three-component vector.

```fea
constraint ground_terminal {
    electrical.potential {
        target = geometry.busbar.region.ground
        value = 0 V
    }
}

constraint current_terminal {
    electrical.integral_current {
        target = geometry.busbar.region.input
        value = 100 A
    }
}
```

## How do I prescribe a terminal voltage?

Use `electrical.potential` when the potential is known:

```fea
constraint high_terminal {
    electrical.potential {
        target = geometry.busbar.region.high_terminal
        value = 48 V
    }
}
```

At least one independent reference is required for a potential formulation. Prescribing both voltage
and incompatible total current on the same terminal overdetermines the model.

## How do I model a perfectly conducting equipotential terminal?

Use `electrical.equipotential` when the region shares one unknown voltage but that voltage is solved:

```fea
constraint floating_terminal_surface {
    electrical.equipotential {
        target = geometry.device.region.terminal_surface
    }
}
```

This constrains variation across the region without assigning the potential value. It is useful for
an ideal terminal connected through a separate integral-current condition.

## How do I model an isolated floating conductor?

```fea
constraint isolated_shield {
    electrical.floating_conductor {
        target = geometry.device.region.shield
    }
}
```

The conductor remains equipotential while its absolute potential follows charge/current balance. Do
not ground it merely to remove a nullspace; that changes the electrical circuit.

## How do I prescribe total current instead of current density?

Use `electrical.integral_current` to constrain
$\int_\Gamma\mathbf J\cdot\mathbf n\,d\Gamma=I$:

```fea
constraint supplied_current {
    electrical.integral_current {
        target = geometry.busbar.region.input_terminal
        value = 100 A
    }
}
```

The target needs a consistent orientation/terminal role. Use `current_density` under loads when the
local intensity is known instead of only its resultant.

## How do I establish the zero-voltage reference without prescribing another value?

```fea
constraint voltage_datum {
    electrical.potential_reference {
        target = geometry.device.region.reference_terminal
    }
}
```

This supplies the potential datum required by the formulation. Use `electrical.potential { value =
0 V }` when a physical grounded terminal is actually part of the model.

## How do I make an electric field periodic?

```fea
constraint electric_unit_cell {
    electrical.periodic {
        primary = geometry.cell.region.x_min
        secondary = geometry.cell.region.x_max
        translation = [5 mm, 0 mm, 0 mm]
        tolerance = 1 um
    }
}
```

The transform maps primary to secondary and must match the CAD periodicity. Plain electrical
periodicity has no phase shift; a driven macroscopic voltage gradient needs a formulation that
represents its affine component explicitly.

## How do I prescribe magnetic vector potential?

```fea
constraint imposed_vector_potential {
    magnetic.vector_potential {
        target = geometry.motor.region.outer_boundary
        value = 1e-6 T*m
    }
}
```

The value has magnetic-vector-potential units. Use this only when the field value itself is known;
insulation is a different natural boundary. The current essential-constraint syntax prescribes one
scalar potential quantity. A general vector-valued Dirichlet boundary with authored component/frame
semantics is unsupported; do not supply `[Ax, Ay, Az]` here.

## How do I ground magnetic vector potential?

```fea
constraint vector_potential_ground {
    magnetic.vector_potential_ground {
        target = geometry.motor.region.ground_boundary
    }
}
```

The edge-based realization constrains compatible tangential edge DOFs on the resolved
boundary. Inspect boundary-localization evidence; a named region that maps to no boundary edges must
fail rather than become a global penalty.

## How do I apply a Coulomb gauge?

```fea
constraint vector_potential_gauge {
    magnetic.coulomb_gauge {
        target = geometry.motor.region.field_domain
    }
}
```

The gauge expresses $\nabla\cdot\mathbf A=0$ intent and removes gradient null modes. A gauge controls
representation; it is not a physical magnetic boundary or source.

## How do I model magnetic periodicity or a rotating sector?

Use `magnetic.periodic` with a translation or complete rotation:

```fea
constraint magnetic_sector {
    magnetic.periodic {
        primary = geometry.sector.region.low_angle_face
        secondary = geometry.sector.region.high_angle_face
        rotation_origin = [0 mm, 0 mm, 0 mm]
        rotation_axis = [0, 0, 1]
        rotation_angle = 10 deg
    }
}
```

Rotation axis is dimensionless; origin carries length; angle is dimensionless. Validate field
orientation/sign under the map, not only geometric face pairing. The accepted transform is either
identity, one translation, or this complete rotation. Translation and rotation cannot be combined;
partial rotations are rejected. Optional `tolerance` is a length and changes pairing from matching
to geometric-tolerance pairing.

## How do I apply phase-shifted magnetic periodicity?

```fea
constraint traveling_wave_sector {
    magnetic.phase_shifted_periodic {
        primary = geometry.sector.region.low_angle_face
        secondary = geometry.sector.region.high_angle_face
        rotation_origin = [0 mm, 0 mm, 0 mm]
        rotation_axis = [0, 0, 1]
        rotation_angle = 10 deg
        phase = 20 deg
    }
}
```

This enforces a complex Bloch/sector phase relation. It requires a complex harmonic solution space;
using it in a real static procedure must fail type/admission checks.

Use a potential reference or an otherwise well-posed potential condition; do not rely on matrix
regularization to choose an arbitrary voltage datum. Magnetic vector-potential formulations likewise
need an explicit gauge/reference compatible with the chosen domain.

Phase-shifted periodic constraints use the same primary/secondary mapping and transform fields as
other periodic constraints, plus a typed `phase`.

## What is the difference between electrical ground and potential reference?

`electrical.potential_reference` removes the additive voltage datum without claiming a driven
terminal voltage beyond that reference semantics. `electrical.potential` prescribes an actual value
on its target. In a simple DC model, both may establish zero on a return terminal, but their physical
ownership and result interpretation differ. Use the declaration matching the formulation and avoid
several arbitrary references on one connected conductor network.

## How do I model several electrically isolated conductor islands?

Each disconnected island needs a well-posed potential relation: a driven/reference terminal or an
`electrical.floating_conductor` contract with its applicable integral charge/current condition.
Do not ground every island merely to remove singularities; that creates conductive references that
may not exist. Inspect connected components and nullspace evidence before selecting constraints.

## How do I model an equipotential electrode whose voltage is unknown?

Use `electrical.equipotential` to make the conductor surface share one unknown potential, then add
the physical integral relation that determines it, such as a floating-conductor charge condition.
A uniform `electrical.potential` would instead prescribe voltage. The result should report the
solved electrode potential and integral charge or current balance.

## How do I verify magnetic periodic-sector constraints?

Inspect geometric pair coverage, rotation origin, axis and angle, transformed vector orientation,
phase sign, and constraint residual. Then compare integrated flux and energy on corresponding
sectors. For phase-shifted periodicity, validate against a known traveling-wave direction; a
reversed primary/secondary pair changes the intended phase relation.

## What electromagnetic constraint evidence should be archived?

Retain terminal/reference target identity, constrained DOF counts, connected conductor components,
floating/equipotential integral residuals, magnetic gauge rank, periodic mapping coverage, and phase
convention. These facts distinguish a physical field solution from one stabilized by an unintended
datum or gauge.

## How do I constrain an electrostatic capacitor model?

Ground one electrode with `electrical.potential_reference` and prescribe or excite the other
electrode. Mark each conductor surface `electrical.equipotential` when the conductor is represented
only by its boundary. Do not ground the dielectric exterior or every conductor; that suppresses the
potential difference being solved.

Check the applied voltage difference, equal-and-opposite electrode charge, electric energy, and
normal displacement flux. Edge-field peaks require geometric radius and mesh convergence rather
than additional constraints.

## How do I constrain a DC conduction model with several terminals?

Give the conductor network one potential reference, then use prescribed terminal potential or
integral current on the remaining terminals. Every disconnected conducting component must be
grounded, driven, or explicitly floating. Do not add one arbitrary zero-potential node per mesh
partition; the constraint belongs to the physical connected component.

The algebraic sum of terminal currents must close. Archive current by terminal, Joule power, and the
reference/floating ownership of every conductor island.

## How do I set up a current-driven busbar with an unknown terminal voltage?

Use an equipotential input terminal, prescribe its integrated current, and provide one return
reference. Keep the equipotential and current constraints separate so each physical role remains
visible:

```fea
constraint input_is_equipotential {
    electrical.equipotential {
        target = geometry.busbar.region.input_terminal
    }
}

constraint input_current {
    electrical.integral_current {
        target = geometry.busbar.region.input_terminal
        value = 250 A
    }
}

constraint grounded_return {
    electrical.potential {
        target = geometry.busbar.region.return_terminal
        value = 0 V
    }
}
```

The solved input voltage is an output; prescribing it as well would generally overdetermine the
terminal. Verify that terminal currents sum to zero under the sign convention, compare
$P=IV$ with $\int_\Omega \mathbf J\cdot\mathbf E\,d\Omega$, and inspect voltage drop and current
density through at least one mesh refinement. Corner current-density peaks are not qualified by
terminal balance alone.

## How do I distinguish magnetic insulation, vector-potential ground, and Coulomb gauge?

`magnetic.insulation` is a physical boundary condition on magnetic flux. A
`magnetic.vector_potential_ground` removes boundary freedom in the vector potential. A
`magnetic.coulomb_gauge` removes nonphysical gauge freedom in the domain. They are not aliases and
should not be stacked until the matrix factors.

Choose the physical truncation first, then the gauge required by the formulation. Verify boundary
flux, gauge residual, operator rank, and invariance of observable $\mathbf B=\nabla\times\mathbf A$
under admissible gauge choices.

## How do I model a periodic motor, generator, or transformer sector?

Pair geometrically corresponding sector faces with `magnetic.periodic` for equal fields or
`magnetic.phase_shifted_periodic` for a declared electrical/spatial phase. State the rotation axis,
origin, angle, and primary/secondary direction. The phase relation must match pole count, winding
harmonic, and sector angle; geometry alone cannot infer it.

Validate the sector against a larger-angle or full model at representative operating points. Compare
pair residual, integrated flux, torque/force, stored energy, and transformed vector orientation.

See [electromagnetic studies](../physics/electromagnetic) and
[electromagnetic loads](../loads/electromagnetic).
