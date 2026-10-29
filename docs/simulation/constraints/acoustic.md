# Acoustic constraints

Acoustic pressure is the primary scalar unknown. Essential pressure imposes $p=\bar p$; periodic
pairs impose $p_s=p_p\circ\mathcal T^{-1}$; phase-shifted periodicity imposes
$p_s=e^{i\phi}p_p\circ\mathcal T^{-1}$.

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `acoustic.pressure` | `target`, `value` | Prescribe acoustic pressure |
| `acoustic.periodic` | periodic pair/transform | Pair periodic acoustic regions |
| `acoustic.phase_shifted_periodic` | periodic pair/transform plus `phase` | Apply a complex phase shift |

This is the complete essential acoustic constraint set. `acoustic.pressure` accepts a scalar
pressure constant or compatible scalar pressure signal; it has no direction or frame. Periodic pair
members are CAD region references, not mesh node sets. Rigid-wall, impedance, and radiation forms
are `load` declarations because they contribute natural boundary operators; writing them as
`constraint` kinds is rejected.

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

## How do I set the acoustic pressure datum without driving the model?

Use a zero `acoustic.pressure` constraint only on a region that is physically held at the reference
pressure. The constraint removes the pressure degree of freedom on every node projected from that
region:

```fea
constraint vent_pressure_reference {
    acoustic.pressure {
        target = geometry.enclosure.region.vent
        value = 0 Pa
    }
}
```

Do not add an arbitrary zero-pressure face to a sealed cavity merely to make its matrix nonsingular.
That turns the selected face into a pressure-release boundary and changes the modes. A formulation
that needs a scalar datum for an otherwise valid sealed problem must provide that gauge explicitly;
it is not represented by hiding a physical boundary condition in solver settings.

## How do I model a pressure-release or soft acoustic boundary?

A pressure-release boundary has $p=0$, so it is exactly a zero `acoustic.pressure` constraint:

```fea
constraint free_surface {
    acoustic.pressure {
        target = geometry.tank.region.liquid_surface
        value = 0 Pa
    }
}
```

This is appropriate for an ideal pressure-release surface, not for a generic opening. A duct opening
that radiates into an exterior domain needs `acoustic.radiation`; a lined wall needs
`acoustic.impedance`. Those natural conditions are documented under
[acoustic loads and boundaries](../loads/acoustic).

## How do I prescribe a complex harmonic pressure?

The constraint value is a typed pressure value. Use it for a real constant datum or a known real
boundary value. A harmonic pressure with frequency-dependent magnitude or phase belongs in a
pressure signal and `acoustic_pressure` excitation so the frequency domain and complex convention
remain explicit:

```fea
signal inlet_pressure : pressure over frequency {
    samples = [
        { at = 100 Hz, value = 1.0 Pa },
        { at = 500 Hz, value = 2.5 Pa },
        { at = 1 kHz, value = 1.2 Pa },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation driven_inlet {
    acoustic_pressure {
        target = geometry.duct.region.inlet
        signal = signal.inlet_pressure
    }
}
```

Use the excitation in the analysis load case. Do not encode phase by placing degrees in a pressure
field; pressure and phase are different quantities. This example provides real-valued spectral
samples. Do not infer an arbitrary complex-amplitude literal syntax from it; use only complex signal
forms admitted by the signal reference for the selected procedure.

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
their normals/orientations must be consistent with the formulation. Omitting `tolerance` requests
matching pairing; a supplied tolerance has length units and requests geometric-tolerance pairing.
The transform may be absent, one translation, or a complete rotation (`rotation_origin`,
`rotation_axis`, `rotation_angle`), never a translation/rotation mixture.

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

## How do I model a straight periodic duct with no phase shift?

Use `acoustic.periodic`, not a zero-degree phase-shifted declaration. The ordinary periodic kind
states the intended real equality directly and works for steady or transient formulations that
admit periodic acoustic pressure:

```fea
constraint repeating_segment {
    acoustic.periodic {
        primary = geometry.duct_cell.region.inlet_cut
        secondary = geometry.duct_cell.region.outlet_cut
        translation = [100 mm, 0 mm, 0 mm]
        tolerance = 2 um
    }
}
```

The translation maps primary coordinates to secondary coordinates. It is not a flow direction or a
propagation vector.

## How do I model acoustic cyclic symmetry around an axis?

Use a rotational transform on the periodic pair [not yet in the current release](pending:acoustic.periodic.rotational). The axis, angle, and paired sector faces must
describe one complete primary-to-secondary map. For a traveling circumferential wave, use
`acoustic.phase_shifted_periodic` and supply the sector phase as well. Do not constrain both faces
independently after pairing them: that over-specifies the same pressure degrees of freedom.

## How do I choose the Bloch phase for a wave vector?

For translation $\mathbf L$ and wave vector $\mathbf k$, the phase is
$\phi=\mathbf k\cdot\mathbf L$. Supply that phase to the phase-shifted periodic constraint using the
same primary-to-secondary direction as the geometric transform. A band-structure sweep should sweep
the typed phase input and retain the wave-vector/sector convention in study evidence; it should not
rewrite region pairing for each case.

## Why does an acoustic periodic constraint fail to pair its faces?

Check the failure in this order:

1. Both regions must resolve to boundary faces, not solids or derived mesh-only groups.
2. The transform must map the complete primary face to the secondary face.
3. The faces must have compatible topology and discretization for the admitted pairing policy.
4. `tolerance` must describe CAD/mesh correspondence error, not bridge a genuinely different shape.
5. Primary and secondary must not overlap after mapping unless the periodic formulation explicitly
   admits that topology.

Increasing tolerance is not a general repair. A large tolerance can pair the wrong nodes and give a
smooth but physically incorrect pressure field.

## Why is my acoustic model overconstrained?

The usual causes are a prescribed pressure on a face that is also in a periodic pair, independent
pressure constraints on both members of a periodic pair, or a pressure constraint combined with an
incompatible pressure excitation on the same degrees of freedom. Keep one owner for each essential
pressure condition. Radiation, impedance, normal velocity, and rigid-wall declarations are natural
boundary operators and must be checked for compatibility separately.

## What should I inspect after applying acoustic constraints?

Inspect the resolved CAD regions, projected node counts, unpaired periodic entities, maximum pairing
distance, constraint residual, and reaction/acoustic-flux balance exposed by the procedure. For
phase-shifted periodicity, also verify the complex phase sign with a known propagation direction.
Successful parsing proves the declaration is typed; it does not prove the selected faces represent
the intended physical boundary.

Rigid walls, radiation boundaries, and impedance are natural boundaries rather than essential
constraints; see [acoustic loads and boundaries](../loads/acoustic). A periodic pair may specify a
translation or complete rotation and an optional geometric tolerance. The phase-shifted form also
requires `phase`, making Bloch/Floquet intent explicit rather than embedding it in solver flags.

## Should I use a rigid wall, pressure-release boundary, or impedance?

A rigid wall imposes zero normal particle velocity and belongs to the natural acoustic boundary set.
A pressure-release surface uses `acoustic.pressure` with zero perturbation pressure. An impedance
boundary relates pressure to normal velocity and can absorb or reflect energy. They are different
physical limits; applying pressure zero and rigid wall to the same face overconstrains the boundary.

## How do I model an acoustic symmetry plane?

Choose the field parity implied by the acoustic mode. A hard symmetry plane corresponds to zero
normal velocity; an antisymmetric or pressure-release plane corresponds to zero pressure. Use the
matching typed boundary or constraint rather than a generic label, and verify the mirrored mode
against a full-domain reference when possible.

## How do I set up a periodic waveguide cell?

Name both cell-end faces, declare their translation, and use `acoustic.periodic` for zero phase or
`acoustic.phase_shifted_periodic` for a Bloch phase. Keep wall termination and material definitions
separate. Sweep phase or frequency through typed study inputs, and request pair residual plus complex
power flux so a sign error is visible.

## What acoustic constraint results should I request?

Request constrained pressure values, periodic pair coverage and residual, complex pressure and
normal velocity on paired faces, and acoustic power flux. For phase-shifted pairs, compare amplitude
and phase after applying the declared mapping. A magnitude-only contour can hide an incorrect Bloch
phase or reversed face normal.

## How do I choose a rigid, pressure-release, impedance, or radiating boundary?

- Use `acoustic.rigid_wall` when normal velocity is zero.
- Use `acoustic.pressure` with zero pressure for an ideal pressure-release surface.
- Use `acoustic.impedance` when pressure and normal velocity follow a known local relation.
- Use `acoustic.radiation` to truncate an exterior domain with the admitted radiation model.

These choices produce different reflection magnitude and phase. A rigid wall is not an open
boundary, and zero pressure is not a generic far field. Compare reflected acoustic power for the
frequency and incidence range that matters.

## How do I constrain a closed acoustic cavity?

Apply rigid or impedance conditions to the physical walls, then add the source as pressure or normal
velocity on its own patch. A purely rigid cavity at zero frequency has a constant-pressure
nullspace; the selected harmonic/transient formulation must own its datum treatment rather than
adding a pressure constraint that changes finite-frequency modes.

Check cavity volume, wall coverage, source area, modal frequencies, and energy balance. A missing
wall region can look like an unintended pressure-release opening.

## How do I constrain an acoustic exterior domain?

Surround the scatterer/source with a fluid region and place a radiation boundary far enough away for
the chosen local approximation, or use a supported absorbing-layer formulation. Apply rigid,
impedance, or coupled structural conditions on the actual body. Do not apply radiation directly to
the vibrating body surface.

Refine both mesh and truncation distance. Inspect reflected power and compare near-field pressure and
radiated power across domain sizes; a smooth contour does not establish nonreflection.

## How do I author a driven duct with rigid walls and a radiating outlet?

Keep the essential/natural roles separate: the source is an excitation, rigid walls and radiation
are loads, and no pressure constraint is placed on the radiating face.

```fea
signal source_level : pressure over frequency {
    samples = [
        { at = 100 Hz, value = 1 Pa },
        { at = 1 kHz, value = 1 Pa },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation inlet_source {
    acoustic_pressure {
        target = geometry.duct.region.inlet
        signal = signal.source_level
    }
}

load duct_wall {
    acoustic.rigid_wall { target = geometry.duct.region.wall }
}

load open_end {
    acoustic.radiation { target = geometry.duct.region.outlet }
}

load_case driven {
    include excitation.inlet_source
    include load.duct_wall
    include load.open_end
}
```

Verify that inlet, wall, and outlet regions have no unintended overlap and cover the intended
boundary. Across a frequency sweep, report complex pressure and normal velocity at both ends,
incident/reflected or net acoustic power under the chosen convention, and global power balance.
Repeat with a longer exterior/truncation distance or a better radiation model; unchanged near-field
response and radiated power are stronger evidence than a visually smooth pressure contour.

## How do I use acoustic symmetry without removing the mode I need?

Choose the mirrored field parity. A rigid symmetry plane imposes zero normal velocity; a
pressure-release plane imposes zero pressure. Which one is correct depends on the acoustic mode and
source symmetry. For antisymmetric excitation, a rigid plane can suppress the intended solution.

Validate a reduced model against the full domain or known modal parity, comparing frequency,
pressure phase, and power after applying the symmetry scale factor.

Further reading: [acoustic studies](../physics/acoustic) and [geometry regions](../geometry).
