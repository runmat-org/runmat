# Constraints

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

## Find a constraint by modeling task

| I need to model… | Start here |
| --- | --- |
| Fixed, pinned, roller, sliding, simple, symmetry, or prescribed-motion support | [Structural constraints](./structural) |
| Bolt/weld idealization, rigid/distributing coupling, nonmatching tie, or equation | [Structural couplings and ties](./structural#coupling-and-ties) |
| Spring, dashpot, elastic support, or compliant mount | [Connectors](./connectors) |
| Opening, sliding, friction, bonded contact, or finite interface stiffness | [Interfaces](./interfaces) |
| Controlled temperature, adiabatic wall, thermal contact, or periodic temperature | [Thermal constraints](./thermal) |
| Ground, floating/equipotential conductor, current terminal, magnetic gauge, or sector | [Electromagnetic constraints](./electromagnetic) |
| Pressure-release surface, datum, acoustic periodic cell, or Bloch phase | [Acoustic constraints](./acoustic) |
| Inlet velocity, stationary/moving/slip wall, pressure datum, or periodic passage | [Fluid constraints](./fluid) |
| Initial displacement, velocity, temperature, pressure, or restart state | [Initial conditions](./initial-conditions) |

These are physical modeling choices, not alternative solver stabilization knobs. Each linked guide
states the equation being imposed, exact `.fea` form, target topology, common conflicts, and evidence
to inspect.

The guides distinguish essential constraints from natural boundaries. Pressure, heat flux,
convection, impedance, radiation, magnetic insulation, and outlet pressure are documented with
[loads and boundary conditions](../loads/), because they contribute flux or traction rather than
directly prescribing the primary unknown.

## Complete essential-constraint catalog

The `.fea` language accepts the following closed set. Family pages provide complete field syntax,
equations, examples, conflicts, and verification evidence. If a kind is absent from this table,
`runmat fea check` rejects it; RunMat does not reinterpret an unknown constraint as a generic
penalty.

| Family | Constraint kinds | Physical role |
| --- | --- | --- |
| Structural supports | `structural.fixed`, `structural.symmetry`, `structural.antisymmetry` | Remove the support/symmetry DOFs defined by the formulation |
| Structural prescribed state | `structural.prescribed`, `structural.prescribed_motion`, `structural.normal_displacement` | Prescribe selected translations, rotations, motion, or normal displacement |
| Structural relations | `structural.rigid_coupling`, `structural.distributing_coupling`, `structural.tie`, `structural.periodic`, `structural.equation` | Relate DOFs across points or paired regions |
| Thermal | `thermal.temperature`, `thermal.periodic`, `thermal.tie`, `thermal.affine_jump` | Prescribe temperature or relate temperature across boundaries |
| Electrical | `electrical.potential`, `electrical.equipotential`, `electrical.floating_conductor`, `electrical.integral_current`, `electrical.potential_reference`, `electrical.periodic` | Prescribe or reference electric potential and terminal current |
| Magnetic | `magnetic.vector_potential`, `magnetic.vector_potential_ground`, `magnetic.coulomb_gauge`, `magnetic.periodic`, `magnetic.phase_shifted_periodic` | Prescribe potential, fix gauge freedom, or relate periodic fields |
| Acoustic | `acoustic.pressure`, `acoustic.periodic`, `acoustic.phase_shifted_periodic` | Prescribe pressure or relate periodic acoustic fields |
| Fluid | `fluid.velocity`, `fluid.moving_wall`, `fluid.no_slip_wall`, `fluid.slip_wall`, `fluid.symmetry`, `fluid.pressure_reference`, `fluid.periodic` | Prescribe velocity/wall kinematics, gauge pressure, or periodic flow |

The target must be a named geometry region. Boundary constraints reject solid and cell targets,
except `magnetic.coulomb_gauge`, which may act on the admitted gauge domain. Paired constraints
require distinct primary and secondary regions plus their mapping/tolerance contract. Scalar values
retain their physical quantity: temperature cannot be used as a generic scalar, and pressure cannot
stand in for an acoustic or fluid velocity condition.

## Which constraint families can I execute today?

All cataloged constraints are part of the typed source contract and are available to `runmat fea
check`. Native whole-study execution currently admits structural procedure graphs; thermal,
electromagnetic, acoustic, fluid, and coupled procedure graphs remain checkable definitions rather
than admitted end-to-end solves. A parser or semantic-check success proves that names, quantities,
targets, and relations are coherent. It does not prove that the requested formulation has an
execution, recovery, and publication path.

Each family guide states the relevant limitation next to its examples. The authoritative summary is
[FEA capability status](../status). Validation claims require the evidence described in
[verification and validation](../validation), including formulation-specific references and
fail-closed negative cases.

## Complete connector and interface catalog

These declarations are related to constraints but assemble different mathematical objects:

| Category | Closed kinds |
| --- | --- |
| Connector | `axial_spring`, `axial_dashpot`, `isotropic_ground_spring`, `surface_normal_foundation` |
| Structural contact | `contact.frictionless`, `contact.coulomb` |
| Fluid–structure transfer | `fluid_structure.no_slip` |
| Conjugate heat transfer | `conjugate_heat_transfer.perfect`, `conjugate_heat_transfer.conductance`, `conjugate_heat_transfer.resistance` |

See [connectors](./connectors) and [interfaces](./interfaces). A directional point spring,
six-DOF bushing, cohesive law,
nonlinear gap spring, radiation interface, or porous jump is not implied by these names and must not
be approximated by supplying dimensionally incompatible coefficients.

## How do I prescribe a value that varies with time or frequency?

Where a constraint field admits `PrescribedValue`, bind a typed signal rather than writing an
expression inside the constraint. The signal owns domain, interpolation, extrapolation, and
function provenance; the constraint owns target and physical meaning.

```fea
signal platen_motion : displacement over time {
    samples = [
        { at = 0 s, value = 0 mm },
        { at = 1 s, value = 2 mm },
        { at = 2 s, value = 2 mm },
    ]
    interpolation = linear
    extrapolation = reject
}

model compression_fixture {
    constraint moving_platen {
        structural.prescribed {
            target = geometry.specimen.region.top_platen
            frame = global
            uz = signal.platen_motion
        }
    }
}
```

The analysis domain must be covered by the signal domain. A frequency-domain constraint also needs
the admitted complex amplitude/phase representation; a real magnitude is not silently promoted to
a zero-phase complex boundary.

## How do I constrain an assembly without hiding its load path?

Give every physical support, tie, connector, and interface its own stable name and the smallest
region that represents its footprint. Do not place one fixed constraint on all mounting faces merely
because it removes rigid modes. For an assembly, inspect reactions by support, connector endpoint
forces, interface transfer, and the moment balance about a stated origin. This preserves the load
path and makes a support-assumption comparison possible.

## How do I model a boundary condition that RunMat does not list?

Do not reuse the nearest-looking declaration. Write down the field, conjugate flux, equation,
units, target topology, time/frequency dependence, and whether the relation is essential, natural,
constitutive, unilateral, or coupled. If no catalog entry has that contract, the condition is not
currently first class. The family guides identify common unsupported cases and the evidence a future
typed implementation would require.

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

## How do constraints interact with load cases and analysis steps?

Constraints are stable named model objects. The study sequence determines when admitted constraints
are active or continued; load cases determine external actions. A preload or retained-state procedure
must state which constraints remain active downstream. Omission does not mean the solver may choose
a convenient support.

## How do I parameterize a prescribed value?

Bind the value to a compatible typed input or signal where that constraint kind admits one. Keep
length, angle, temperature, pressure, velocity, or electromagnetic quantity types intact. A sweep
varies the typed input and produces distinct result cases; it does not perform textual replacement in
the constraint block.

## How do I choose primary and secondary sides?

Choose the direction that matches the declared geometric transform, projection policy, and physical
reporting convention. Primary/secondary is retained in evidence and can affect normal sign,
projection, phase, and contact reporting. Reversing the pair is not merely cosmetic when the law is
directional.

## What happens to constraints when the mesh changes?

Constraints resolve from persistent CAD regions on each admitted mesh. Ties, periodic pairs, and
interfaces repeat their pairing/projection and report coverage/error. RunMat does not preserve stale
node numbers from a prior mesh. Compare target measure, projected entity count, rank, and residual
between mesh revisions.

## What is the difference between a constraint, connector, interface, and load?

These declarations assemble different mathematical objects:

| Declaration | Mathematical role | Typical question |
| --- | --- | --- |
| Constraint | Prescribes a field value or exact relation, $\mathsf B\mathbf x=\mathbf g$ | “Which motion or field value is forbidden or known?” |
| Connector | Adds a constitutive relation such as $\mathbf f=k\Delta\mathbf u$ | “What compliant or dissipative attachment connects these regions?” |
| Interface | Transfers fields or enforces unilateral/coupled behavior between two domains | “How do these bodies or physics communicate?” |
| Natural-boundary load | Adds traction, flux, impedance, or an outlet operator | “What conjugate action acts on this boundary?” |

This distinction controls result interpretation. A fixed constraint produces a reaction; a spring
stores energy; contact changes active status; pressure contributes external work. Choosing one only
because it makes the matrix solvable changes the model.

## How do I write a complete constrained model fragment?

Declare geometry regions by their physical role, then place constraints inside the model that owns
them. Loads remain separately named so activation is reviewable:

```fea
geometry bracket {
    file = "bracket.step"
    units = mm
    region body : solid { select tag "body"; expect exactly_one }
    region mount : face { select tag "mount"; expect one_or_more }
    region pad : face { select tag "load_pad"; expect exactly_one }
}

model supported_bracket {
    assign material.steel to geometry.bracket.region.body

    constraint mounting_fixture {
        structural.fixed { target = geometry.bracket.region.mount }
    }

    load service_force {
        force on geometry.bracket.region.pad = [0, -4, 0] kN
    }
    load_case service { include load.service_force }
}
```

`mounting_fixture` removes admitted structural degrees of freedom only on `mount`. The load case
does not activate or infer the support. In a multi-step study, the analysis sequence states whether
that named constraint continues into each downstream procedure.

## How do I review constraint rank and reactions?

For linear equality constraints, inspect the resolved relation matrix rank, removed degrees of
freedom, redundant rows, and residual $\|\mathsf B\mathbf x-\mathbf g\|$. For supports, integrate
reaction force and moment by named constraint. For ties and periodic pairs, inspect matched measure,
unmatched entities, projection distance, and pair residual. For contact, rank alone is insufficient:
inspect active/open/stick/slip state, penetration, and contact work.

## What should fail during `runmat fea check`?

Checking should reject an incompatible topology role, missing or ambiguous region, duplicate
identity, invalid quantity or frame, incomplete pair transform, self-pair, nonphysical coefficient,
conflicting ownership, and a procedure that cannot consume the constraint kind. Geometry-dependent
pair coverage and operator nullspaces are resolved later, but they still fail closed before a result
is admitted. Solver regularization is not a substitute for a missing physical reference.
