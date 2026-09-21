# Constraints, boundaries, connectors, and interfaces

Constraints state what motion or field values are restricted. They are model objects, not meshing
options and not solver guesses. Every target is a typed geometry region, so RunMat can reject a
solid used where a face is required, a stale persistent id, or an ambiguous selector before solve.

| Domain | Guide |
| --- | --- |
| Supports, prescribed motion, coupling, ties, equations, periodicity | [Structural constraints](./structural) |
| Temperature, thermal ties, jumps, and periodicity | [Thermal constraints](./thermal) |
| Electric references, conductors, current constraints, gauges, and periodicity | [Electromagnetic constraints](./electromagnetic) |
| Pressure and periodic acoustic conditions | [Acoustic constraints](./acoustic) |
| Velocity, walls, pressure reference, and periodic flow | [Fluid constraints](./fluid) |
| Springs, dashpots, and foundations | [Connectors](./connectors) |
| Contact and multiphysics transfer | [Interfaces](./interfaces) |
| Displacement, velocity, temperature, pressure, and field state at the initial instant | [Initial conditions](./initial-conditions) |

The guides distinguish essential constraints from natural boundaries. Pressure, heat flux,
convection, impedance, radiation, magnetic insulation, and outlet pressure are documented with
[loads and boundary conditions](../loads/), because they contribute flux or traction rather than
directly prescribing the primary unknown.

## Common form

```fea
model supported_part {
    constraint support {
        structural.fixed { target = geometry.part.region.mounting_faces }
    }
}
```

The declaration name (`support`) is stable model identity. The kind (`structural.fixed`) selects a
closed typed contract. `target` resolves a named region from the bound geometry. An analysis may
activate, continue, or modify model objects explicitly; omission never means “let the solver choose.”

## Which boundary condition should I use?

Start with the field being solved and the physical statement at the boundary:

| Physical statement | Use | Do not substitute |
| --- | --- | --- |
| A displacement or rotation component is known | `structural.prescribed` | An arbitrary stiff spring |
| A surface cannot move | `structural.fixed` | Zero force |
| A temperature is controlled | `thermal.temperature` | Very large convection coefficient |
| Heat transfer to an ambient is known | `thermal.convection` load | Fixed temperature |
| Fluid sticks to a stationary wall | `fluid.no_slip_wall` | Zero pressure |
| Fluid pressure needs a gauge datum | `fluid.pressure_reference` | Outlet pressure on every outlet |
| Two surfaces are permanently kinematically bonded | structural/thermal `tie` | Contact |
| Two surfaces may open or slide | contact interface | Tie |
| Repeating boundaries are geometrically mapped | family-specific `periodic` | Coincident-node assumptions |
| Motion or fields transfer between physics | typed multiphysics interface | Duplicated manual histories |

Essential constraints prescribe the primary unknown. Natural boundaries prescribe its conjugate
flux or traction and are documented under [loads](../loads/). Connectors add constitutive relations;
interfaces transfer state between two regions. Keeping these categories separate makes assembly,
validation, and preview behavior predictable.

## How do region targets connect constraints to CAD?

Define a named, typed geometry region from persistent CAD identity or an explicit geometric
selection, then reference it as `geometry.<geometry-name>.region.<region-name>`. The compiler checks
the region's entity dimension and cardinality before meshing; the admission path records how it maps
to solver entities after meshing.

```fea
geometry bracket {
    file = "bracket.step"
    units = mm

    region mounting_faces : face {
        select tag "mount"
        expect one_or_more
    }
}

model supported_bracket {
    constraint mounting_support {
        structural.fixed {
            target = geometry.bracket.region.mounting_faces
        }
    }
}
```

If a CAD revision removes or splits the tagged region contrary to its cardinality contract, checking
fails. RunMat does not silently choose a similarly located face.

## How do I constrain only one direction?

Use a component-wise structural prescribed constraint in an explicit frame. A fixed support removes
all admitted structural degrees of freedom and is too strong when a roller, slider, or bearing only
blocks selected motion. See [structural constraints](./structural) for component masks, local frames,
normal displacement, symmetry, and couplings.

## How do I connect nonmatching meshes?

Use a typed tie or interface and name both sides. Pairing tolerance controls geometric matching; it
does not create compliance. Choose structural tie for kinematic continuity, thermal tie for equal
temperature, finite thermal conductance/resistance for a temperature jump, and contact when opening
or sliding is possible.

## How do I model cyclic symmetry or a repeating cell?

Use the periodic constraint for the solved field family and state the complete translation or
rotation that maps primary to secondary. Thermal affine jump adds a macroscopic temperature
difference. Acoustic and magnetic phase-shifted periodicity add complex Bloch phase. Periodicity is
not inferred from matching face shapes.

## How do I detect an underconstrained model?

RunMat checks field nullspaces before solve where the formulation exposes them: rigid-body modes in
structural statics, the constant-temperature mode in pure-Neumann steady conduction, pressure gauge
freedom in incompressible flow, electric-potential reference freedom, and magnetic gauge freedom.
Add the physical reference the model is missing; do not add an arbitrary numerical anchor merely to
make the matrix nonsingular.

## How do I avoid conflicting or redundant constraints?

Review overlaps by resolved solver DOF, not declaration name. Two region names can map to the same
face or node set. RunMat diagnoses incompatible values and records redundant relations; the family
guides explain valid combinations. In particular, do not combine fixed and nonzero prescribed
motion on the same DOF, tie and contact on the same pair, or several independent gauge references
without a physical reason.

## What should I inspect before trusting a constrained solve?

Check resolved target coverage, removed/free DOF counts, pair projection coverage, reference/gauge
status, reaction resultants, and any constraint residual. For contact, also inspect active/open/slip
state and penetration. These facts belong in result evidence so a later run can show exactly which
constraint realization was solved.
