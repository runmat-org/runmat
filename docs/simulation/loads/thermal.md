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

The typed source checker validates the declarations on this page. Supported standalone steady
thermal studies can run through `runmat run` on a native host; the study's formulation,
boundaries and resource plan must pass admission. Authoring a boundary does not by itself make
its execution available. In particular, surface radiation remains explicitly rejected by the
canonical thermal procedure, as described below.

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `heat_generation` | $\mathrm{W/m^3}$ | Volumetric source in a solid region or explicit whole-model excitation |
| `heat_flux` | $\mathrm{W/m^2}$ | Signal-driven surface flux |
| `temperature_field` | K | Uniform imposed temperature input on a solid region |
| `ambient_temperature` | K | Ambient-temperature excitation for a compatible formulation |
| `thermal.heat_flux` | `target`, `value` | Constant natural surface flux |
| `thermal.convection` | `target`, `ambient_temperature`, `coefficient` | Robin boundary $q=h(T-T_\infty)$ |
| `thermal.radiation` | `target`, `ambient_temperature`, `emissivity` | Authored surface-to-ambient gray radiation; thermal procedure execution remains rejected |

`heat_generation` targets a solid or, in explicit excitation form, `whole_model`.
`temperature_field` targets a solid. Heat flux, ambient temperature, and the natural boundary
blocks target faces. Every implemented thermal quantity here is scalar, so normal orientation—not
an authored direction vector—sets the sign of a surface flux.

```fea
load chip_loss {
    heat_generation on geometry.module.region.die = 8000000 W/m^3
}

load cooled_surface {
    thermal.convection {
        target = geometry.module.region.finned_faces
        ambient_temperature = 298.15 K
        coefficient = 35 W/m^2/K
    }
}
```

## Surface-to-ambient radiation admission

```fea
load radiating_surface {
    thermal.radiation {
        target = geometry.module.region.finned_faces
        ambient_temperature = 298.15 K
        emissivity = 0.8
    }
}
```

The authored contract uses nonnegative Kelvin and dimensionless emissivity in `[0, 1]`.
The CPU face-form library integrates outward loss
`epsilon * sigma * (T^4 - T_ambient^4)` and its exact consistent tangent. This describes
exchange with a prescribed ambient, not enclosure/view-factor or participating-medium radiation.
The canonical thermal procedure currently **rejects** this boundary explicitly: nonlinear
steady/transient closure, resource admission, whole-solve heat accounting and execution/restart
qualification are not yet implemented. Authorability and local face evidence do not admit a solve.
See [radiation qualification](../qualification/thermal-radiation.md).

## How do I apply volumetric heat generation?

```fea
load winding_loss {
    heat_generation on geometry.motor.region.winding = 3500000 W/m^3
}
```

This targets a solid region and preserves source intensity. Integrated power is
$P=\int_\Omega Q\,d\Omega$; compare it with the expected electrical, chemical, or mechanical loss.
For temperature-dependent or spatially nonuniform loss, use a compatible signal/function or coupled
physics state rather than one averaged constant.

## How do I apply electronic, battery, or chemical heat generation?

Convert the governing loss model to a volumetric source only when its spatial support is known. For
uniform loss in a cell or die, use `heat_generation` on that exact solid region. For state-dependent
loss, generate the source from a typed RunMat function or transfer it from the electrical/chemical
procedure so units, parameter values, and integrated power remain in evidence.

## How do I apply heat to a thin layer or surface heater?

Use surface `heat_flux` when thickness is intentionally collapsed. Use `heat_generation` in the
actual solid when through-thickness gradients or layer heat capacity matter. Converting a layer
source $Q$ to surface flux requires $q''=Qt$ with modeled thickness $t$; record that reduction.

## How do I split total power among several components?

Create one region-specific generation load per component. If fractions $f_i$ are known, use
$Q_i=f_iP/V_i$ and verify $\sum_i\int_{V_i}Q_i\,dV=P$. Do not apply the full assembly power to each
solid or divide by the total bounding-box volume.

## How do I apply a heat flux to a surface?

The direct form creates a constant excitation:

```fea
load laser_flux {
    heat_flux on geometry.part.region.illuminated_face = 25000 W/m^2
}
```

The explicit natural-boundary form is:

```fea
load imposed_flux {
    thermal.heat_flux {
        target = geometry.part.region.hot_face
        value = 25000 W/m^2
    }
}
```

Both carry power per area. Check the outward-normal sign convention and integrated power over the
resolved area. A total wattage must be converted through known area or represented by a resultant-
preserving contract; it is not numerically equal to heat flux.

## How do I apply a known total heater power to a face?

For uniform distribution over resolved area $A$, set $q''=P/A$ and verify the integrated boundary
power in evidence. If the heater footprint or spatial distribution matters, define that exact CAD
region and use its field. A constant flux on a changing/deformed area needs an explicit reference-
versus-current configuration convention.

## How do I model a laser or moving heat source?

A constant `heat_flux` models a stationary uniform spot only. A moving Gaussian source needs beam
power, absorptivity, radius, path, coordinate frame, and time dependence in a typed spatial function.
The current `.fea` signal domain is time, frequency, study parameter, or constant; it does not expose
the coordinates of each target point to a source function. A moving spatial source is therefore not
first class. Do not approximate it by turning an entire face on and off unless that is the intended
reduced model and its deposited-power integral is verified.

## Why is my heat-flux sign reversed?

Flux sign follows the resolved outward normal and formulation convention. Preview the CAD face,
check its normal, and compare integrated boundary power with the expected input. Reversing a face
selector or negating a value are not interchangeable if several bodies share an interface.

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
        coefficient = 80 W/m^2/K
    }
}
```

This applies $q=h(T-T_\infty)$ with constant $h$ and $T_\infty$. Split regions when different flow
zones need different coefficients. Do not use convection as an undocumented surrogate for radiation
or contact resistance.

## How do I choose a convection coefficient?

Use a correlation, test result, or coupled flow solution valid for the geometry, fluid, regime, and
temperature range. Record whether $h$ is local or area-averaged. A single coefficient on all faces
can hide stagnant zones and fins with very different flow. Run a sensitivity range when $h$ is an
uncertain engineering input.

## How do I model natural versus forced convection?

The `thermal.convection` syntax is the same once $h$ and $T_\infty$ are known; their derivation is
different. For natural convection, $h$ depends on orientation, gravity, geometry, and temperature
difference. For forced convection, it depends on flow speed and regime. Solve conjugate heat
transfer when the fluid field and local coefficient must be predicted rather than prescribed.

## How do I apply different ambient temperatures to different faces?

Split the CAD into governed regions and declare one convection load per environment. Do not select a
parent region and then try to override a subset with a second load: overlapping Robin operators can
double-count heat transfer unless the formulation explicitly defines precedence.

## How do I create a time-varying power or heat-flux cycle?

Use a typed `heat_generation` or `heat_flux` signal over time, then apply it with a matching
excitation. Interpolation and extrapolation are mandatory because the solver may sample between
authored points. Signal/function source identity is retained in the result and Desktop preview.

```fea
signal power_cycle : heat_generation_density over time {
    samples = [
        { at = 0 s, value = 0 W/m^3 },
        { at = 30 s, value = 2000000 W/m^3 },
        { at = 90 s, value = 2000000 W/m^3 },
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

## How do I apply a measured transient surface heat flux?

Use `heat_flux` rather than `heat_generation` when the measurement is power per unit area. The
signal is scalar; its sign is resolved against the selected face's outward normal:

```fea
signal measured_flux : heat_flux over time {
    samples = [
        { at = 0 s, value = 0 W/m^2 },
        { at = 2 s, value = 12000 W/m^2 },
        { at = 8 s, value = 4000 W/m^2 },
        { at = 10 s, value = 0 W/m^2 },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation measured_surface_heating {
    heat_flux {
        target = geometry.calorimeter.region.heater_footprint
        signal = signal.measured_flux
    }
}
```

Verify the delivered history
$P(t)=\int_{\Gamma_q}q''(t)\,d\Gamma$. If the measurement is total watts, divide by the admitted
footprint area only for a justified uniform-flux approximation.

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

## How do I linearize radiation near an operating temperature?

If a documented reduced model is acceptable, derive
$h_r\approx4\epsilon\sigma T_m^3$ about stated mean absolute temperature $T_m$ and use convection
with $h_r$ and the radiation-environment temperature. Record the linearization temperature and run
a sensitivity check. This approximation is not a first-class radiation solve and becomes inaccurate
over wide temperature ranges.

## How do I model heat transfer across a contact interface?

Use `conjugate_heat_transfer.perfect`, `.conductance`, or `.resistance` between the two named
surfaces. A heat flux applies a known boundary source; it does not conserve equal-and-opposite energy
between independently solved bodies. See [contact and multiphysics interfaces](../constraints/interfaces).

## How do I model heat-sink contact or thermal interface material?

Use a conductance/resistance interface when the layer is reduced to a boundary law. Model the layer
as a solid with conductivity, density, and heat capacity when its thickness gradients or transient
storage matter. Do not represent contact resistance by lowering bulk conductivity in unrelated
parts.

## How do I model Joule heating from an electrical solve?

Transfer the retained electromagnetic loss field through an electrothermal procedure. The transfer
must conserve integrated electrical loss and record topology mapping. Use manually authored
`heat_generation` only for a deliberate one-way reduced model with the source calculation retained
as evidence.

## How do I model plastic-work, friction, or damping heat?

These are coupled dissipation sources. Use a procedure/interface that transfers the correct fraction
of mechanical work into thermal power. A constant guessed heat flux is acceptable only as an
explicit engineering approximation. State partition factors, contact side, and time dependence.

## Why does the thermal energy balance not close?

Check volume and surface units, source-region volume, convection area, boundary normal signs,
interface double counting, transient stored energy, and whether imported coupled losses preserve
their resultant. In steady state, total source power should equal total outward boundary/interface
power within tolerance. In transient analysis, include the rate of stored thermal energy.

## What should I inspect after applying thermal loads?

Inspect resolved volumes/areas, integrated power by source, convection/radiation sink power, contact
transfer on both sides, peak and mean temperatures, and global energy residual. For signal-driven
loads, verify time-domain coverage and interpolation at every solver step.

For varying heat input, declare a `heat_generation` or `heat_flux` signal over time and apply it
with an excitation of the same kind. Surface flux and volume generation are deliberately different
dimensions and target roles. A temperature constraint fixes the primary field; convection and flux
do not.

Check integrated power, exterior area, sign convention, temperature units, and energy balance.

## How do I model a chip, PCB, or battery heat source in RunMat?

Create a solid region for each dissipating component, assign its thermal material, and apply
`heat_generation` using power divided by that region's admitted volume. Keep sources separate when
their powers or duty cycles differ:

```fea
load processor_loss {
    heat_generation on geometry.board.region.processor_die = 18000000 W/m^3
}

load regulator_loss {
    heat_generation on geometry.board.region.regulator_die = 6000000 W/m^3
}

load_case full_power {
    include load.processor_loss
    include load.regulator_loss
}
```

Verify the integrated watts for each component. Applying total board power to a bounding-box region
changes source density and spreads heat through material that does not generate it.

## How do I model a prescribed heater power and convection together?

Declare the heater and cooling boundaries as separate loads and include them in the same case. The
heater adds energy; convection removes or adds energy according to the current wall temperature:

```fea
load cartridge_heater {
    heat_generation on geometry.assembly.region.heater = 4000000 W/m^3
}

load ambient_cooling {
    thermal.convection {
        target = geometry.assembly.region.exposed_faces
        ambient_temperature = 293.15 K
        coefficient = 12 W/m^2/K
    }
}
```

Do not subtract an estimated convective wattage from the heater input; the Robin boundary computes
heat exchange from the solved wall temperature.

## How do I choose between temperature, heat flux, and convection boundary conditions?

Use `thermal.temperature` when a controller or reservoir truly fixes surface temperature. Use
`thermal.heat_flux` when surface power density is known independently of surface temperature. Use
`thermal.convection` when heat flow is proportional to the difference between unknown wall
temperature and known ambient. These produce different thermal resistances and cannot be selected
only to obtain a desired peak temperature.

## How do I check a thermal load mesh refinement study?

Across meshes, compare admitted source volume or area, integrated source power, boundary heat flow,
stored-energy change, and temperature norms. Thin layers and sharp source footprints can require
local refinement even when total energy closes. A stable peak temperature with a drifting energy
balance is not convergence; neither is energy closure with an unresolved local gradient at a
material or contact interface.

## How do I model a thermal soak followed by cooldown?

Use an ambient-temperature signal for the chamber or fluid history and a convection boundary for
the heat-transfer law. Do not prescribe the part temperature unless the test fixture actually
controls the part surface to that temperature.

```fea
signal chamber_cycle : temperature over time {
    samples = [
        { at = 0 s, value = 293.15 K },
        { at = 1800 s, value = 393.15 K },
        { at = 5400 s, value = 393.15 K },
        { at = 7200 s, value = 293.15 K },
    ]
    interpolation = linear
    extrapolation = hold_endpoint
}

excitation chamber_air {
    ambient_temperature {
        target = geometry.part.region.exposed_faces
        signal = signal.chamber_cycle
    }
}

```

The selected transient formulation must admit `ambient_temperature` as the varying film ambient and
must provide the corresponding heat-transfer coefficient. The direct `thermal.convection` boundary
currently accepts a constant `ambient_temperature`; do not write a signal reference there and assume
it was consumed. The thermal time constant depends on heat capacity, conduction, exposed area, and
film coefficient. Check energy stored during heat-up, energy rejected during cooldown, and time-step
convergence at the ramps.

## How do I model heat from a chip, winding, battery cell, or resistor?

Start from the known power definition. Use `heat_generation` on the actual dissipating volume when
loss is volumetric, `thermal.heat_flux` for a thin resolved heater, or an electrothermal procedure
when current distribution must determine $q'''=\mathbf J\cdot\mathbf E$. If a component dissipates
total power $P$ uniformly in selected volume $V$, the source density is $q'''=P/V$.

Do not apply $P$ as though it were $q'''$: watts and watts per cubic meter are different typed
quantities. After projection, verify that $\int_V q'''\,dV=P$ and that the source did not spread into
non-dissipating encapsulant, substrate, or coolant.

## How do I model a prescribed temperature map from test or another solver?

Use `temperature_field` only when the temperature field is an input to a consuming procedure, such
as thermal-strain loading. A steady or transient thermal solve normally solves temperature; fixing
the same field as a load would remove that unknown. Bind imported data to coordinates, units,
interpolation, out-of-domain behavior, and source identity, then retain projection error and
coverage.

For thermomechanical work, map the thermal result through the typed state edge so the structural
procedure can prove which mesh, time/frequency case, and temperature reference produced thermal
strain. Avoid exporting anonymous nodal CSV data and reattaching it by row order.

The current `.fea` grammar does not yet accept an imported topology field for `temperature_field`.
It accepts a uniform scalar signal on a solid region. The mapping contract described above is the
required future boundary, not syntax that can be authored today.

## How do I represent a heater controlled to constant power or constant temperature?

Constant power is a source (`heat_generation` or `thermal.heat_flux`). Constant temperature is an
essential constraint (`thermal.temperature`). A thermostatic controller is neither: it is a
feedback law that switches or modulates power from measured temperature. Model that controller with
an admitted function/coupling contract; do not alternate between power and fixed temperature to
force the desired curve.

For every control representation, report commanded power, delivered power, sensor temperature,
switching or saturation state, and global energy balance.

## How do I verify a complete steady thermal model?

Compute each contribution in watts rather than comparing temperature contours:

$$
P_{volume}=\int_\Omega Q\,d\Omega,\qquad
P_{flux}=\int_{\Gamma_q}\bar q\,d\Gamma,\qquad
P_{conv}=\int_{\Gamma_h}h(T-T_\infty)\,d\Gamma.
$$

At steady state, signed source and sink power must close, including every interface transfer. Then
refine the mesh and compare integrated powers, mean temperature, and the temperature at engineering
probe locations. A peak at a point source or sharp material corner may not converge; qualify it by
distance or averaged region. For transient work, add the stored-energy rate
$d\int_\Omega \rho c_pT\,d\Omega/dt$ and repeat with a smaller time step.

Further reading: [thermal constraints](../constraints/thermal),
[heat-transfer studies](../physics/thermal), and
[electrothermal analysis](../physics/electrothermal).
