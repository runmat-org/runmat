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

## How do I choose spring and dashpot coefficients for a target natural frequency?

For a one-DOF estimate, $\omega_n=\sqrt{k/m}$ and damping ratio
$\zeta=c/(2\sqrt{km})$. Use the effective participating mass and connector direction of the actual
model, then verify the assembled modal result. This estimate does not replace eigenanalysis when the
attachment excites several structural modes.

## Why does a connector make the model singular?

A translational connector only restrains the relative motion represented by its admitted basis. It
may leave rigid rotation or orthogonal translations free. Inspect connector target coverage and the
remaining rigid-body modes; add only the physical constraints the assembly really has.

Use springs for elastic compliance, dashpots for rate-dependent force, and a foundation for a
distributed support law. Use [structural constraints](./structural) for kinematic restrictions and
[interfaces](./interfaces) for contact or multiphysics transfer.
