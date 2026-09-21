# Thermal loads and boundaries

Thermal loads enter the heat equation as volume and boundary integrals. Convection adds both a
boundary operator and source:

$$
\mathbf q_Q=\int_\Omega\mathsf N^{\mathsf T}Q\,d\Omega,
\quad
\mathbf q_{flux}=\int_\Gamma\mathsf N^{\mathsf T}\bar q\,d\Gamma,
\quad
\mathsf K_h=\int_\Gamma\mathsf N^{\mathsf T}h\mathsf N\,d\Gamma,
\quad
\mathbf q_h=\int_\Gamma\mathsf N^{\mathsf T}hT_\infty\,d\Gamma.
$$

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `heat_generation` | $\mathrm{W/m^3}$ | Volumetric source in solid/cell regions |
| `heat_flux` | $\mathrm{W/m^2}$ | Signal-driven surface flux |
| `temperature_field` | K | Spatial or varying imposed field |
| `ambient_temperature` | K | Ambient-temperature excitation for a compatible formulation |
| `thermal.heat_flux` | `target`, `value` | Constant natural surface flux |
| `thermal.convection` | `target`, `ambient_temperature`, `coefficient` | Robin boundary $q=h(T-T_\infty)$ |

```fea
load chip_loss {
    heat_generation on geometry.module.region.die = 8 MW/m^3
}

load cooled_surface {
    thermal.convection {
        target = geometry.module.region.finned_faces
        ambient_temperature = 298.15 K
        coefficient = 35 W/(m^2*K)
    }
}
```

## How do I apply volumetric heat generation?

```fea
load winding_loss {
    heat_generation on geometry.motor.region.winding = 3.5 MW/m^3
}
```

This targets a solid/cell volume and preserves source intensity. Integrated power is
$P=\int_\Omega Q\,d\Omega$; compare it with the expected electrical, chemical, or mechanical loss.
For temperature-dependent or spatially nonuniform loss, use a compatible signal/function or coupled
physics state rather than one averaged constant.

## How do I apply a heat flux to a surface?

The direct form creates a constant excitation:

```fea
load laser_flux {
    heat_flux on geometry.part.region.illuminated_face = 25 kW/m^2
}
```

The explicit natural-boundary form is:

```fea
load imposed_flux {
    thermal.heat_flux {
        target = geometry.part.region.hot_face
        value = 25 kW/m^2
    }
}
```

Both carry power per area. Check the outward-normal sign convention and integrated power over the
resolved area. A total wattage must be converted through known area or represented by a resultant-
preserving contract; it is not numerically equal to heat flux.

## How do I prescribe a temperature field as a load?

```fea
load operating_temperature {
    temperature_field on geometry.bracket.region.body = 373.15 K
}
```

`temperature_field` is useful when temperature drives thermomechanical response or supplies an
initial/known field without solving heat transfer in that step. It targets solver nodes/regions and
conflicting overlapping fields are rejected. Use `thermal.temperature` when the thermal equation is
being solved and the temperature is an essential boundary condition.

## How do I provide ambient temperature as a varying excitation?

```fea
signal ambient_cycle : temperature over time {
    samples = [
        { at = 0 s, value = 293.15 K },
        { at = 600 s, value = 323.15 K },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation chamber_temperature {
    ambient_temperature {
        target = geometry.part.region.exterior_faces
        signal = signal.ambient_cycle
    }
}
```

Ambient temperature needs a compatible convection/radiation formulation to transfer heat. It is not
by itself the same as prescribing the part surface to that temperature.

## How do I model convection to air or coolant?

```fea
load forced_convection {
    thermal.convection {
        target = geometry.part.region.cooled_faces
        ambient_temperature = 298.15 K
        coefficient = 80 W/(m^2*K)
    }
}
```

This applies $q=h(T-T_\infty)$ with constant $h$ and $T_\infty$. Split regions when different flow
zones need different coefficients. Do not use convection as an undocumented surrogate for radiation
or contact resistance.

## How do I create a time-varying power or heat-flux cycle?

Use a typed `heat_generation` or `heat_flux` signal over time, then apply it with a matching
excitation. Interpolation and extrapolation are mandatory because the solver may sample between
authored points. Signal/function source identity is retained in the result and Desktop preview.

```fea
signal power_cycle : heat_generation_density over time {
    samples = [
        { at = 0 s, value = 0 W/m^3 },
        { at = 30 s, value = 2 MW/m^3 },
        { at = 90 s, value = 2 MW/m^3 },
        { at = 120 s, value = 0 W/m^3 },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation cycling_loss {
    heat_generation {
        target = geometry.module.region.die
        signal = signal.power_cycle
    }
}
```

## How do I convert total component power into heat generation?

If a component dissipates total power $P$ uniformly in resolved volume $V$, use $Q=P/V$. Prefer a
resultant-preserving power contract when available; otherwise compute the intensity from the admitted
CAD volume and record that dependency. Do not divide by a nominal volume that differs from the
meshed region. Verify $\int_\Omega Q\,d\Omega=P$ in result evidence.

## How do I model an insulated or adiabatic wall?

For the standard conduction weak form, an omitted natural boundary has zero normal heat flux. Keep
the face out of heat-flux, convection, and interface declarations and document the intended
adiabatic assumption in the study. Do not add a zero-temperature constraint; that is an isothermal
sink, not insulation.

## How do I model thermal radiation?

The current closed load set does not include a surface-radiation law
$q=\epsilon\sigma(T^4-T_{sur}^4)$. Do not approximate radiation with a constant convection
coefficient unless that linearization and its operating temperature are part of the model intent.
An admitted radiation boundary needs emissivity, surrounding/radiation temperature, view-factor or
enclosure semantics, nonlinear tangent behavior, and energy-balance evidence.

## How do I model heat transfer across a contact interface?

Use `conjugate_heat_transfer.perfect`, `.conductance`, or `.resistance` between the two named
surfaces. A heat flux applies a known boundary source; it does not conserve equal-and-opposite energy
between independently solved bodies. See [contact and multiphysics interfaces](../constraints/interfaces).

For varying heat input, declare a `heat_generation` or `heat_flux` signal over time and apply it
with an excitation of the same kind. Surface flux and volume generation are deliberately different
dimensions and target roles. A temperature constraint fixes the primary field; convection and flux
do not.

Check integrated power, exterior area, sign convention, temperature units, and energy balance.
Further reading: [thermal constraints](../constraints/thermal), [heat-transfer studies](../physics/thermal),
and [electrothermal analysis](../physics/electrothermal).
