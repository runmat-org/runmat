# Contact and interfaces

Interfaces pair two typed geometry regions and state how fields cross their common boundary.

Structural contact enforces nonpenetration $g_n\ge0$, compressive contact traction $t_n\le0$, and
complementarity $g_nt_n=0$; Coulomb contact additionally bounds tangential traction by
$\|\mathbf t_t\|\le\mu|t_n|$. Perfect CHT enforces temperature and flux continuity, while finite
conductance uses $q_n=h_c(T_p-T_s)$. FSI no-slip enforces velocity continuity and traction balance.

| Kind | Required fields | Use |
| --- | --- | --- |
| `contact.frictionless` | `primary`, `secondary` | Structural contact without tangential friction |
| `contact.coulomb` | `primary`, `secondary`, `coefficient` | Contact with Coulomb friction |
| `fluid_structure.no_slip` | `primary`, `secondary` | Transfer traction and wall motion |
| `conjugate_heat_transfer.perfect` | `primary`, `secondary` | Perfect thermal transfer |
| `conjugate_heat_transfer.conductance` | pair plus `coefficient` | Finite conductance law |
| `conjugate_heat_transfer.resistance` | pair plus `resistance` | Finite contact resistance law |

The table is the complete current authored member set. Interface targets are CAD region references;
the source language does not accept mesh IDs. Unlike `structural.tie` and `thermal.tie`, interface
declarations have no `tolerance`, search-distance, penalty, enforcement, stabilization, or pairing-
policy member. Pair realization and its evidence belong to the admitted formulation. If the two
regions cannot be paired under that contract, change the geometry/regions or use a formulation that
exposes the needed policy—do not add undocumented members.

```fea
interface bonded_heat_path {
    conjugate_heat_transfer.conductance {
        primary = geometry.assembly.region.hot_side
        secondary = geometry.assembly.region.cold_side
        coefficient = 5000 W/m^2/K
    }
}
```

## How do I model frictionless structural contact?

```fea
interface bearing_contact {
    contact.frictionless {
        primary = geometry.assembly.region.pin_surface
        secondary = geometry.assembly.region.bore_surface
    }
}
```

Frictionless contact transfers compressive normal traction, permits tangential slip, and permits
opening. Primary/secondary orientation affects search/projection and must be reviewed. Use a tie when
the surfaces are permanently bonded; use normal-displacement constraint only when unilateral contact
status is intentionally excluded.

## How do I model Coulomb friction contact?

```fea
interface frictional_joint {
    contact.coulomb {
        primary = geometry.assembly.region.pad
        secondary = geometry.assembly.region.rail
        coefficient = 0.25
    }
}
```

The coefficient is dimensionless and nonnegative. Coulomb friction limits tangential traction by
$\mu|t_n|$ but does not by itself define adhesion, wear, velocity dependence, or regularization.
Inspect penetration, active-set/slip status, contact work, and enforcement evidence produced by the
formulation. Penalty or multiplier selection is not authored in this interface block.

## How do I couple a no-slip fluid boundary to a moving structure?

```fea
interface wet_wall {
    fluid_structure.no_slip {
        primary = geometry.duct.region.fluid_wall
        secondary = geometry.duct.region.structural_wall
    }
}
```

This transfers fluid traction to the structure and structural wall motion to the fluid. The
interface must cover corresponding physical surfaces and the FSI solve must converge both traction
and displacement transfer; a `fluid.moving_wall` constraint would instead prescribe motion without
two-way feedback.

## How do I model perfect thermal contact between fluid and solid?

```fea
interface perfect_heat_transfer {
    conjugate_heat_transfer.perfect {
        primary = geometry.cooler.region.solid_wall
        secondary = geometry.cooler.region.fluid_wall
    }
}
```

Perfect transfer enforces temperature continuity and equal/opposite normal heat flux. Use it only
when interfacial resistance is negligible at the accuracy required.

## How do I specify thermal interface conductance?

```fea
interface finite_conductance {
    conjugate_heat_transfer.conductance {
        primary = geometry.cooler.region.solid_wall
        secondary = geometry.cooler.region.fluid_wall
        coefficient = 5000 W/m^2/K
    }
}
```

This uses $q_n=h_c(T_p-T_s)$. The coefficient carries $\mathrm{W/(m^2 K)}$ and should be supported
by contact/coating data over the actual pressure and temperature range.

## How do I specify thermal contact resistance instead?

```fea
interface finite_resistance {
    conjugate_heat_transfer.resistance {
        primary = geometry.assembly.region.hot_face
        secondary = geometry.assembly.region.cold_face
        resistance = 2e-4 m^2*K/W
    }
}
```

Resistance $R''$ is the reciprocal of conductance when both describe the same linear interface:
$q_n=(T_p-T_s)/R''$. Use one representation, not both, and verify area normalization.

## Should I use contact or a tie?

Use a tie when the surfaces remain bonded and share compatible kinematics for the whole analysis.
Use contact when they may open, close, or slide. A tie is linear and bilateral; contact is generally
state-dependent and unilateral. Applying both to the same pair suppresses the contact behavior and
must be rejected as conflicting ownership.

## How do I model bonded contact with finite stiffness?

The closed interface set currently provides structural tie and frictionless/Coulomb contact, but no
cohesive-zone, adhesive, or finite-normal/tangential-stiffness interface. Do not tune contact penalty
to stand in for a physical adhesive law. A complete cohesive interface needs traction-separation,
damage/fracture-energy, mixed-mode, thickness, and state-retention semantics.

## How do I model thermal contact that changes with pressure?

The current conductance/resistance interface is a declared coefficient. Pressure-dependent thermal
contact requires a coupled constitutive law evaluated from structural contact pressure and gap. Do
not manually copy one pressure snapshot into several unrelated conductance values unless the staged
approximation and its validity range are explicit.

## How do I verify contact pairing and interface conservation?

Inspect primary/secondary covered area, unmatched entities, projection distance, normal orientation,
and pair weights. Structural contact should report penetration, normal/tangential reaction, open/
stick/slip state, and contact work. Thermal and FSI transfer should report equal-and-opposite
integrated flux/traction to their numerical tolerance.

## How do I select primary and secondary contact surfaces?

Choose the pairing direction deliberately. The secondary side supplies the surface against which
the primary side is searched/projected under the admitted contact formulation. Prefer a side whose
geometry and discretization make the projection well conditioned, then verify coverage rather than
relying on a “fine mesh versus coarse mesh” rule. Reversing the pair can change normals, projection
error, and reported reactions even when the physical bodies are unchanged.

```fea
interface pin_in_bore {
    contact.frictionless {
        primary = geometry.joint.region.pin_contact
        secondary = geometry.joint.region.bore_contact
    }
}
```

The two regions must name the physical mating patches. Selecting a complete cylindrical face when
only a partial arc can contact changes both search and resultant interpretation.

## How do I model a bonded structural interface?

Use `structural.tie` in the owning model when the surfaces remain kinematically bonded for the whole
analysis. It is a bilateral constraint, not an `interface` declaration:

```fea
constraint bonded_joint {
    structural.tie {
        primary = geometry.assembly.region.part_a_bond
        secondary = geometry.assembly.region.part_b_bond
        tolerance = 10 um
    }
}
```

Use contact instead when opening or slip is part of the physics. Use a first-class cohesive law
when finite interface stiffness or damage is required; the tie cannot represent adhesive compliance.

## How do I model a fluid–structure boundary in RunMat?

Declare one `fluid_structure.no_slip` interface and use it from a
`fluid_structure_interaction` procedure. The interface owns the paired CAD regions; the procedure
owns the coupling and convergence policy. Do not add an independent prescribed wall velocity on the
same fluid face unless the study intentionally overrides structural motion. At every accepted
coupling step, compare transferred fluid traction with the structural interface load and compare
wall velocity/displacement after mapping.

## How do I model conjugate heat transfer between a fluid and solid?

Use `conjugate_heat_transfer.perfect` when temperature is continuous. Use `.conductance` or
`.resistance` when the model intentionally retains a temperature jump. The interface is then
consumed by a `conjugate_heat_transfer` procedure; two unrelated thermal boundary loads do not
constitute a coupled interface. Verify equal-and-opposite integrated heat flow and report the
temperature jump for finite conductance/resistance.

## Why does an interface leave unmatched faces or nodes?

Check that both named regions cover the same physical patch, the primary-to-secondary orientation
is correct, and assembly transforms were applied. This interface syntax has no authorable tolerance
or pairing policy. Inspect unmatched area and maximum projection distance in physical units before
changing mesh size. If the admitted realization cannot pair the surfaces, it must fail explicitly;
there is no source-level tolerance knob to widen.

## What results establish interface convergence?

For contact, inspect penetration, complementarity residual, normal reaction, tangential traction,
stick/slip/open status, and contact work. For CHT, inspect paired temperature/flux residual and
integrated energy conservation. For FSI, inspect displacement/velocity transfer, traction balance,
and coupling residual by iteration. Report these against the named interface, not only as one global
solver norm.

## How do I model bolted or clamped contact between parts?

Contact defines possible separation and slip; it does not create clamp load. Supply bolt preload or
resolved fastener loading through a supported structural representation, then use frictionless or
Coulomb contact on the mating surfaces. If preload is outside the current closed load set, do not
substitute a tuned friction coefficient or interference without documenting the approximation.

Check normal contact pressure, open/stick/slip area, bolt or clamp resultant, friction work, and
reaction balance through the preload and service steps. Mesh refinement should preserve total
contact force while resolving pressure gradients.

For a clamped lap joint whose preload is supplied by a separately supported bolt representation,
the mating interface itself remains explicit:

```fea
interface faying_surface {
    contact.coulomb {
        primary = geometry.joint.region.upper_faying_face
        secondary = geometry.joint.region.lower_faying_face
        coefficient = 0.18
    }
}
```

This declaration does not create bolt preload, initial closure, interference, or contact
stabilization. The current interface block also has no search tolerance. Before accepting service
stress, require pair coverage evidence, compressive normal resultant consistent with preload,
stick/slip/open area by load step, tangential/normal reaction balance, and contact-work convergence.
Refine both surfaces independently; total transferred force should converge even when local edge
pressure remains singular.

## How do I model a bonded joint, adhesive layer, or weld?

Use `structural.tie` only when relative motion is negligible and failure is out of scope. Model a
resolved adhesive solid when thickness and constitutive response matter. A cohesive interface is
required for traction-separation and damage; the current tie/contact pair must not be relabeled as
cohesive behavior.

For a tie, retain pair coverage, transfer residual, and interface resultants. For a resolved layer,
retain thickness, material axes, mesh through thickness, peel/shear stress, and energy convergence.

## How do I model thermal contact in a bolted assembly?

Use conductance when the interface law is $q_n=h_c\Delta T$ or resistance when
$q_n=\Delta T/R''$. Choose the coefficient from surface finish, pressure, interstitial material, and
temperature data. If conductance must evolve with solved contact pressure, use an admitted coupled
law; one constant value is an explicit approximation.

Report interface temperature jump, integrated heat flow on both sides, unmatched area, and global
energy balance. A shared temperature contour can hide an accidentally perfect tie.

## How do I verify fluid–structure and conjugate-heat-transfer mapping?

For FSI, compare integrated fluid traction with equal-and-opposite structural load, plus mapped wall
velocity/displacement and coupling residual. For CHT, compare integrated heat flux on both sides,
temperature continuity or the declared jump, and energy storage. Perform these checks by named
interface and iteration/time step, not only globally.

Repeat the transfer check after independent mesh refinement on each side. Conservation must survive
nonmatching discretizations; nearest-node agreement is not a substitute for resultant-preserving
projection.

Primary and secondary are part of the contract. Use boundary-face regions for the two sides; volume
regions and incidental geometric proximity do not create an interface. Semantic admission checks
family/topology roles, realization checks coverage and projection, and execution evidence records
the resolved interface. Never rely on geometric proximity to create contact or transfer
automatically.

Further reading: [conjugate heat transfer](../physics/conjugate-heat-transfer),
[fluid–structure interaction](../physics/fluid-structure-interaction), and [meshing](../meshing).
