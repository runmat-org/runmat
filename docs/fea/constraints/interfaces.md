# Contact and multiphysics interfaces

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

```fea
interface bonded_heat_path {
    conjugate_heat_transfer.conductance {
        primary = geometry.assembly.region.hot_side
        secondary = geometry.assembly.region.cold_side
        coefficient = 5000 W/(m^2*K)
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
Inspect penetration, active-set/slip status, contact work, and penalty/multiplier evidence.

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
        coefficient = 5000 W/(m^2*K)
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
to stand in for a physical adhesive law. A production cohesive interface needs traction-separation,
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

Primary and secondary are part of the contract. The compiler checks topology roles; realization
checks coverage and projection; execution evidence records the resolved interface. Never rely on
geometric proximity to create contact or transfer automatically.

Further reading: [conjugate heat transfer](../physics/conjugate-heat-transfer),
[fluid–structure interaction](../physics/fluid-structure-interaction), and [meshing](../meshing).
