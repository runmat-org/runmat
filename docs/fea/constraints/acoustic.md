# Acoustic constraints

Acoustic pressure is the primary scalar unknown. Essential pressure imposes $p=\bar p$; periodic
pairs impose $p_s=p_p\circ\mathcal T^{-1}$; phase-shifted periodicity imposes
$p_s=e^{i\phi}p_p\circ\mathcal T^{-1}$.

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `acoustic.pressure` | `target`, `value` | Prescribe acoustic pressure |
| `acoustic.periodic` | periodic pair/transform | Pair periodic acoustic regions |
| `acoustic.phase_shifted_periodic` | periodic pair/transform plus `phase` | Apply a complex phase shift |

```fea
constraint pressure_reference {
    acoustic.pressure {
        target = geometry.cavity.region.reference_face
        value = 0 Pa
    }
}
```

## How do I prescribe acoustic pressure or establish a pressure reference?

`acoustic.pressure` fixes $p=\bar p$ on the target. A zero value can establish the datum required by
a pressure formulation; a nonzero value describes a known boundary pressure:

```fea
constraint driven_pressure {
    acoustic.pressure {
        target = geometry.duct.region.inlet
        value = 2 Pa
    }
}
```

For frequency-dependent drive, prefer an `acoustic_pressure` signal/excitation so domain and complex
amplitude are explicit. Prescribed pressure is not the same as a radiation or impedance boundary.

## How do I model a periodic acoustic unit cell?

```fea
constraint repeating_duct {
    acoustic.periodic {
        primary = geometry.cell.region.x_min
        secondary = geometry.cell.region.x_max
        translation = [25 mm, 0 mm, 0 mm]
        tolerance = 1 um
    }
}
```

This enforces equal pressure after mapping. The geometry pair must cover corresponding faces and
their normals/orientations must be consistent with the formulation.

## How do I model Bloch or phase-shifted acoustic periodicity?

```fea
constraint bloch_pair {
    acoustic.phase_shifted_periodic {
        primary = geometry.cell.region.x_min
        secondary = geometry.cell.region.x_max
        translation = [25 mm, 0 mm, 0 mm]
        phase = 45 deg
    }
}
```

The relation is $p_s=e^{i\phi}p_p$. It requires a complex harmonic solution space and is used for
wave propagation/band-structure cells. Phase direction follows the primary-to-secondary map; reverse
the pair only if the sign convention changes with it.

Rigid walls, radiation boundaries, and impedance are natural boundaries rather than essential
constraints; see [acoustic loads and boundaries](../loads/acoustic). A periodic pair may specify a
translation or complete rotation and an optional geometric tolerance. The phase-shifted form also
requires `phase`, making Bloch/Floquet intent explicit rather than embedding it in solver flags.

Further reading: [acoustic studies](../physics/acoustic) and [geometry regions](../geometry).
