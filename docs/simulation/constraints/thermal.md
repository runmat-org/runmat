# Thermal constraints

Thermal constraints operate on the nodal temperature vector. A prescribed value partitions the
conductivity system just as an essential displacement partitions structural equilibrium. Paired
constraints have the forms

$$
T_p-T_s=0\quad\text{(tie/periodic)},\qquad
T_p-T_s=\Delta T\quad\text{(affine jump)}.
$$

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `thermal.temperature` | `target`, `value` | Prescribe temperature on a region |
| `thermal.tie` | `primary`, `secondary`, optional `tolerance` | Enforce temperature continuity across paired regions |
| `thermal.periodic` | pair plus optional periodic transform | Enforce periodic temperature across mapped regions |
| `thermal.affine_jump` | periodic pair/transform plus `jump` | Enforce a declared temperature difference |

Every `primary` and `secondary` is a CAD region reference, not a mesh-entity list. Omitting
`tolerance` requests matching pairing; specifying it selects geometric pairing with a typed length.
A periodic transform is exactly one of: identity (no transform members), `translation = [x,y,z]`,
or the complete set `rotation_origin`, `rotation_axis`, and `rotation_angle`. Translation and
rotation cannot be combined, and a partial rotation is rejected. `rotation_origin` and translation
carry length, the axis is dimensionless, and the angle is a dimensionless angle.

```fea
constraint cold_plate {
    thermal.temperature {
        target = geometry.part.region.cold_face
        value = 293.15 K
    }
}

constraint bonded_interface {
    thermal.tie {
        primary = geometry.part.region.side_a
        secondary = geometry.part.region.side_b
        tolerance = 5 um
    }
}
```

## How do I hold a surface at a known temperature?

Use `thermal.temperature` for a Dirichlet condition $T=\bar T$:

```fea
constraint coolant_wall_temperature {
    thermal.temperature {
        target = geometry.part.region.coolant_wall
        value = 285.15 K
    }
}
```

The value may be a compatible prescribed signal where the procedure admits one. Do not use a fixed
temperature to approximate finite convection unless the wall temperature is genuinely controlled;
it removes thermal resistance and can underpredict component temperature.

## How do I prescribe a time-varying boundary temperature?

Define a temperature signal and reference it from the prescribed constraint when the selected
transient procedure admits signal-backed prescribed values. The signal owns time sampling and
interpolation; the constraint owns the geometric target:

```fea
signal oven_cycle : temperature over time {
    samples = [
        { at = 0 s, value = 293.15 K },
        { at = 10 min, value = 423.15 K },
        { at = 40 min, value = 423.15 K },
        { at = 55 min, value = 293.15 K },
    ]
    interpolation = linear
    extrapolation = hold_endpoint
}

constraint programmed_wall {
    thermal.temperature {
        target = geometry.oven.region.controlled_wall
        value = signal.oven_cycle
    }
}
```

Use a constant value for steady analysis. Use a signal only when the procedure has an independent
time or frequency domain against which it can be sampled.

## How do I set an initial temperature without holding it for the whole solve?

Use a thermal initial condition, not `thermal.temperature`. An initial condition sets $T(\mathbf
x,t_0)$; a constraint enforces $T=\bar T$ for as long as it is active. The distinction matters for
cool-down, warm-up, quenching, and thermal-shock studies. See
[initial conditions](./initial-conditions#how-do-i-set-the-initial-temperature-for-transient-heat-transfer)
for the exact declaration.

## How do I model an isothermal component or surface?

If the actual temperature is known, apply `thermal.temperature` to the whole controlled region. If
only the fact that the region is spatially isothermal is known while its temperature must be solved,
a fixed temperature is wrong: it supplies an artificial heat reservoir. Use an admitted
equipotential/lumped thermal relation when available, or model the high-conductivity body explicitly.
The current closed thermal constraint set does not disguise an unknown isothermal temperature as a
Dirichlet value.

## How do I tie temperatures across two nonmatching meshes?

Use `thermal.tie` when the interface has no temperature jump:

```fea
constraint perfect_bond {
    thermal.tie {
        primary = geometry.assembly.region.part_a_face
        secondary = geometry.assembly.region.part_b_face
        tolerance = 10 um
    }
}
```

The pairing projection must cover the intended interface. The tolerance controls geometric pairing,
not contact conductance. For finite contact resistance, use a conjugate-heat-transfer interface with
conductance/resistance rather than weakening the tie numerically.

## How do I model perfect thermal contact between two solids?

Use `thermal.tie` for two independently meshed solid faces when temperature is continuous and no
interface resistance is intended. If both materials already share conforming nodes, no duplicate
tie should be necessary: the shared degrees of freedom already impose continuity. Use
[thermal contact conductance](./interfaces#how-do-i-specify-thermal-interface-conductance) when a
finite temperature jump is physical.

## How do I model thermal contact resistance?

Do not use `thermal.affine_jump`: it prescribes a known jump and does not make the jump proportional
to heat flux. Declare a thermal interface with conductance $h_c$ or resistance $R_c=1/h_c$. The
interface law is $q_n=h_c(T_p-T_s)$, while a tie is the limiting perfect-contact relation
$T_p=T_s$.

## How do I model an insulated or adiabatic boundary?

For the standard heat equation, a boundary with no declared thermal flux naturally has
$-\mathbf n\cdot k\nabla T=0$. Leave that face out of heat-flux, convection, radiation, and
temperature declarations. Do not add a `0 K` temperature constraint: that is a cold isothermal
boundary, not insulation.

## How do I model convection instead of a fixed wall temperature?

Convection is a natural boundary condition,
$-\mathbf n\cdot k\nabla T=h(T-T_\infty)$, and belongs in a `thermal.convection` load. It keeps the
wall temperature as an unknown. See
[thermal loads](../loads/thermal#how-do-i-model-convection-to-air-or-coolant) for exact `.fea`
syntax.

## How do I model periodic temperature in a repeating cell?

Use `thermal.periodic` when paired faces have the same temperature after geometric mapping:

```fea
constraint repeating_temperature {
    thermal.periodic {
        primary = geometry.cell.region.x_min
        secondary = geometry.cell.region.x_max
        translation = [10 mm, 0 mm, 0 mm]
        tolerance = 1 um
    }
}
```

The transform is identity, one translation, or a complete rotation. Equal periodic temperature by
itself represents no macroscopic gradient; use an affine jump when the unit cell carries a known
temperature difference.

For a repeating cell with a known temperature drop:

```fea
constraint unit_cell_gradient {
    thermal.affine_jump {
        primary = geometry.cell.region.inlet_face
        secondary = geometry.cell.region.outlet_face
        translation = [10 mm, 0 mm, 0 mm]
        jump = 5 K
    }
}
```

## How do I impose a temperature gradient on a periodic unit cell?

`thermal.affine_jump` enforces periodic fluctuations plus a known macroscopic difference:

$$T(\mathbf x+\mathbf L)-T(\mathbf x)=\Delta T.$$

The `jump` has temperature-difference units. Its sign follows primary-to-secondary mapping, so verify
the transform and expected heat-flow direction. A jump without a compatible periodic pair is
rejected. `jump` may be a constant or a compatible scalar temperature-difference signal admitted by
the procedure; it is never inferred from heat flux or material conductivity.

## How do I model a repeating thermal sector around an axis?

Use `thermal.periodic` with a rotational transform when the solved temperatures repeat exactly from
one sector face to the next. Use `thermal.affine_jump` only when the repeated cell carries a known
macroscopic temperature difference. A rotation describes geometry correspondence; the `jump`
describes thermodynamic difference. They are independent fields and should not be inferred from one
another.

## How do I impose a heat-flow rate through a periodic cell?

The affine constraint controls temperature difference, not total heat rate. To prescribe heat rate,
apply a compatible heat-flux/resultant boundary load and use ordinary periodicity on the fluctuation
field when the formulation supports that homogenization setup. Do not tune `jump` until the desired
power appears unless the study is explicitly solving an inverse problem.

## Can I prescribe temperature on a point, edge, face, or volume?

The target must satisfy the topology admitted by the formulation and resolve before meshing. A face
is the normal boundary target. A solid-region temperature constraint fixes every projected thermal
degree of freedom and is usually equivalent to an ideal thermostatted body; use it deliberately.
Point or edge constraints are valid only if the chosen formulation admits those boundary roles.
RunMat rejects an incompatible solid/cell or unsupported target rather than silently expanding it.

## Why is my steady thermal solve singular?

A connected conduction component with only zero-flux boundaries has an arbitrary additive constant.
Give it one physical temperature reference, positive convection/radiation to a known environment, or
another admitted condition that establishes absolute temperature. Each disconnected thermal
component needs its own path to a reference. A tie only joins components if its region pairing
actually resolves.

## Why is the predicted temperature too low near a support or coolant wall?

A prescribed temperature supplies unlimited heat transfer needed to hold that value. If the real
support has finite contact resistance or the coolant has finite film conductance, use an interface or
convection load instead. Also check that a broad region selector did not fix hidden faces or the
entire solid.

## Why does a thermal tie or periodic pair fail validation?

Verify exact region identity, primary-to-secondary transform, target topology, pairing coverage,
maximum projection distance, and duplicate ownership. Do not use a large tolerance to compensate for
different interface shapes. A failed pair is a geometry/modeling issue until the evidence shows
otherwise.

## What should I inspect after applying thermal constraints?

Inspect resolved target measure, projected node count, imposed-temperature extrema, tie/periodic
pair coverage, maximum pairing distance, constraint residual, total boundary heat flow, and steady
energy balance. For affine jumps, verify sign by comparing mean temperature on the primary and
secondary faces.

Prescribed temperature is an essential condition. Convection and heat flux are natural boundaries
and belong in [thermal loads](../loads/thermal). Check that the chosen combination admits heat flow
and does not impose incompatible temperatures on the same DOFs.

For steady pure-Neumann conduction, adding one physical temperature reference or a positive
convection boundary removes the constant-temperature nullspace. RunMat rejects an underconstrained
connected component rather than choosing an arbitrary mean temperature.

## How do I model hot and cold controlled boundaries in one thermal study?

Declare each reservoir as its own `thermal.temperature` constraint so its heat flow and target stay
separately attributable:

```fea
constraint hot_reservoir {
    thermal.temperature {
        target = geometry.sample.region.hot_face
        value = 373.15 K
    }
}

constraint cold_reservoir {
    thermal.temperature {
        target = geometry.sample.region.cold_face
        value = 293.15 K
    }
}
```

Request reaction heat flow on both constraints and compare it with volume and natural-boundary
sources. Do not apply both values to overlapping faces or use one broad selector that captures both
ends.

## How do I model thermal shock at a surface?

Choose the physical boundary law. If a bath or controller truly fixes surface temperature, use a
time-varying `thermal.temperature`. If heat transfer is limited by a film, use time-varying ambient
temperature with `thermal.convection`; the wall remains unknown. Initialize the body separately at
its pre-shock temperature. The initial field, boundary signal, and first-step side limit determine
whether the shock is intentional or an input discontinuity.

## How do I author a hot-part cooldown without accidentally clamping its temperature?

Put the starting temperature in an `initial` declaration and the finite cooling law in a
convection load. Do not replace either with a continuing fixed-temperature constraint:

```fea
initial hot_start {
    thermal.temperature {
        target = geometry.housing.region.body
        uniform = 423.15 K
    }
}

load ambient_cooling {
    thermal.convection {
        target = geometry.housing.region.exposed_faces
        ambient_temperature = 293.15 K
        coefficient = 18 W/m^2/K
    }
}

load_case cooldown { include load.ambient_cooling }
```

Select `model.initial.hot_start` through the transient procedure's `initial_conditions` field. The
initial state contributes stored energy; convection leaves wall temperature unknown. Verify at each
accepted time that applied heat plus boundary heat equals stored-energy change to the procedure's
tolerance. Repeat with smaller time steps around the fastest initial decay and refine the boundary
layer until surface heat flow and the temperatures of interest are stable.

## How do I constrain a thermally periodic representative volume element?

Pair opposite faces with `thermal.periodic` for zero macroscopic gradient or
`thermal.affine_jump` for a prescribed temperature difference. Supply the exact translation or
rotation, and constrain one physical temperature reference if the fluctuation field retains an
additive nullspace. Verify mean heat flux, face-pair residual, and effective conductivity against
cell size and mesh refinement.

## What thermal constraint results should I request?

Request prescribed-temperature reaction heat flow, mean/minimum/maximum temperature on each target,
tie or periodic residual, pair coverage, and global energy balance. For affine jumps, also request
the volume-averaged gradient and heat flux. A temperature contour cannot reveal whether a fixed wall
supplied an unrealistic amount of heat.

## How do I choose between fixed temperature, convection, heat flux, and insulation?

Use `thermal.temperature` only for a boundary held at known temperature by a sufficiently strong
reservoir or controller. Use `thermal.convection` when heat flow depends on unknown wall temperature
and known bulk-fluid temperature. Use `thermal.heat_flux` for known power per area. Insulation is the
natural zero-flux condition and normally needs no explicit declaration unless naming it improves
review or the procedure requires complete boundary classification.

These are different mathematical conditions. Replacing convection with a fixed ambient temperature
removes the film resistance and usually underpredicts wall temperature; replacing a known heat flux
with fixed temperature changes the energy input.

## How do I model a cold plate, heat sink, or temperature-controlled mounting surface?

Choose the boundary at the level represented by the geometry. If the cold plate is omitted and its
interface temperature is measured, prescribe `thermal.temperature`. If coolant-to-wall resistance
matters, model the wall and use convection or conjugate heat transfer. If a thermal interface
material separates two solids, use a conductance/resistance interface rather than one shared
temperature constraint.

Report reaction heat flow at controlled-temperature boundaries. That reaction is the cooling power
the ideal reservoir must supply; compare it with the real cold plate or controller capacity.

## How do I represent a measured boundary-temperature history?

Attach a temperature signal to `thermal.temperature` and state interpolation and extrapolation
explicitly. Sampling cadence must resolve the fastest thermal input relevant to the model.

```fea
signal measured_wall : temperature over time {
    samples = [
        { at = 0 s, value = 293.15 K },
        { at = 60 s, value = 323.15 K },
        { at = 300 s, value = 318.15 K },
    ]
    interpolation = linear
    extrapolation = reject
}

constraint instrumented_wall {
    thermal.temperature {
        target = geometry.test_article.region.instrumented_wall
        value = signal.measured_wall
    }
}
```

Do not extend a short measurement implicitly. Retain source file identity, units, timestamps,
sensor-to-boundary interpretation, and the boundary reaction heat flow.

## How do I avoid an overconstrained thermal assembly?

Each connected conducting component needs enough boundary physics to establish temperature, but
overlapping fixed temperatures, ties, and finite-conductance interfaces can assign two owners to the
same thermal DOFs. Use exactly one relation across each physical interface. A tie represents zero
temperature jump; conductance represents a finite jump; two independent temperatures represent two
reservoirs, not contact.

Inspect connected components after contact/interface resolution, then review prescribed-DOF overlap,
pair coverage, reaction heat flow, and the global energy balance.
