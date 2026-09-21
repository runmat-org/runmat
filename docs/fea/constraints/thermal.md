# Thermal constraints

Thermal constraints operate on the nodal temperature vector. A prescribed value partitions the
conductivity system just as an essential displacement partitions structural equilibrium. Paired
constraints have the forms

$$T_p-T_s=0\quad\text{(tie/periodic)},\qquad
T_p-T_s=\Delta T\quad\text{(affine jump)}.$$

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `thermal.temperature` | `target`, `value` | Prescribe temperature on a region |
| `thermal.tie` | `primary`, `secondary`, optional `tolerance` | Enforce temperature continuity across paired regions |
| `thermal.periodic` | pair plus optional periodic transform | Enforce periodic temperature across mapped regions |
| `thermal.affine_jump` | periodic pair/transform plus `jump` | Enforce a declared temperature difference |

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
rejected.

Prescribed temperature is an essential condition. Convection and heat flux are natural boundaries
and belong in [thermal loads](../loads/thermal). Check that the chosen combination admits heat flow
and does not impose incompatible temperatures on the same DOFs.

For steady pure-Neumann conduction, adding one physical temperature reference or a positive
convection boundary removes the constant-temperature nullspace. RunMat rejects an underconstrained
connected component rather than choosing an arbitrary mean temperature.
