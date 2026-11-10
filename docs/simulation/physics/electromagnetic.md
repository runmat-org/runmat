# Electromagnetic

Electromagnetic analysis predicts electric potential, current flow, magnetic vector potential,
flux density, field intensity, and frequency-dependent loss. RunMat uses typed quantities for volts,
amperes, coulombs, current density, and vector potential; it never passes them to a solver as
unitless numbers.

Four electromagnetic analyses run from a `.fea` study with `runmat run`:

| Analysis | Selected by | Unknown | Elements |
| --- | --- | --- | --- |
| Steady current flow | `electromagnetic { equation = steady_conduction }` + `electromagnetic static {}` | electric potential $V$ | Tet4 / Tet10 nodal |
| Electrostatics | `electromagnetic { equation = electrostatics }` + `electromagnetic static {}` | electric potential $V$ | Tet4 / Tet10 nodal |
| Magnetostatics | `electromagnetic { equation = magnetostatics }` + `electromagnetic static {}` | magnetic vector potential $\mathbf A$ | lowest-order Nédélec edges |
| Time-harmonic fields | `electromagnetic harmonic { frequencies = [...] }` | electric field phasor $\mathbf E$ | lowest-order Nédélec edges |

A fifth analysis computes the impedance, delay and coupling of board traces from a 2D
cross-section of the stackup: `electromagnetic line_parameters {}`. See
[trace impedance](./trace-impedance.md).

The loss of a steady-current or harmonic analysis can heat a thermal analysis of the same parts
(see [How does the loss heat the part?](#how-does-the-loss-heat-the-part) and
[electrothermal analysis](./electrothermal)).

## What equations are solved?

**Steady current flow and electrostatics.** $\nabla\cdot(a\nabla V)=-q$ with $a=\sigma$ (conduction)
or $a=\varepsilon$ (electrostatics), $\mathbf E=-\nabla V$, $\mathbf J=\sigma\mathbf E$ or
$\mathbf D=\varepsilon\mathbf E$. Prescribed potentials are essential conditions; an injected terminal
current is a natural condition with uniform normal current density over the terminal faces; every
other face is insulated. Results: nodal potential, element electric field, current density or
displacement, and Joule power density $\sigma|\nabla V|^2$ (or energy density). The Joule integral
equals $\sum V_iI_i$ over the terminals (checked on every solve).

**Magnetostatics.** $\nabla\times(\mu^{-1}\nabla\times\mathbf A)=\mathbf J_s$ with tangential
$\mathbf A=0$ on grounded faces (`magnetic.vector_potential_ground`, the far field or a flux wall)
and magnetic insulation, $\mathbf n\times\mathbf H=0$, on every other exterior face (a symmetry
plane of the current). A weak Coulomb gauge (`magnetic.coulomb_gauge`) fixes the gradient part of
$\mathbf A$ in the whole field: in air, iron and conductors alike. Results:
vector potential, flux density $\mathbf B=\nabla\times\mathbf A$ and field intensity, at integration
points and as element averages.

**Time-harmonic fields.** For each authored frequency, with $e^{i\omega t}$ phasors (peak values),

$$
\nabla\times\left(\mu^{-1}\nabla\times\mathbf E\right)
+ \left(i\omega\sigma - \omega^2\varepsilon\right)\mathbf E
= -i\omega\mathbf J_s,
$$

the full Maxwell equation in the temporal gauge ($\mathbf E=-i\omega\mathbf A$), including
displacement current. Conductivity, permittivity and permeability are per material; zero
conductivity is allowed, so air needs no artificial loss. Results at integration points and as
element averages: $\mathbf E$, $\mathbf B=i\nabla\times\mathbf E/\omega$, $\mathbf H$,
$\mathbf D$, conduction current density, and the cycle-averaged Joule loss density
$\tfrac12\sigma|\mathbf E|^2$.

Boundary conditions for harmonic fields:

- `magnetic.vector_potential_ground` (or `electrical.perfect_conductor`): tangential
  $\mathbf E=0$, the far field of an air region or a perfectly conducting wall.
- every other exterior face: $\mathbf n\times\mathbf H=0$ (a symmetry plane of the current).

Sources for harmonic fields:

- `magnetic_volume_current_density` on a body: an impressed (stranded) current density.
- a **driven solid conductor**: `electric_current on <terminal face> = I` with `electrical.potential
  { value = 0 V }` on its return terminal (current drive), or two `electrical.potential` terminals
  (voltage drive). Inside the conductor the field is $\mathbf E-U\nabla v_0$, where $v_0$ is the
  conductor's electrokinetic unit potential and $U$ the terminal voltage (an unknown for current
  drive). Skin and proximity effect in the driven conductor are solved, not assumed. Terminals that
  lie on a grounded face return their current through it; terminals inside the model (a gap-fed
  coil) are joined by the discrete potential flow of the terminal current through the electrically
  transparent medium, so no artificial charge builds up across the gap. A medium is electrically
  transparent at a frequency when its skin depth exceeds the model size
  ($\omega\mu\sigma L^2<1$, $L$ the model diagonal).

Each terminal drive reports its voltage, current and complex power $\tfrac12 U\bar I$; the Joule
loss in the conductors equals the real power delivered, and the complex Poynting balance and weak
charge continuity are checked on every frequency.

Every authored frequency is an independent solve. A study with several frequencies runs one case
per frequency (`frequency=<f>Hz`), in parallel, and each case publishes its own fields.

The harmonic and magnetostatic systems are factored with a sparse direct solver (LU with partial
pivoting); the residual of the original equations is measured after the solve.

## How do I describe a harmonic electromagnetic study?

```fea
model hardening_cell {
    assign material.steel_4140 to geometry.cell.region.gear
    assign material.copper to geometry.cell.region.coil
    assign material.air to geometry.cell.region.air
    constraint coil_return {
        electrical.potential { target = geometry.cell.region.coil_out; value = 0 V }
    }
    constraint far_field {
        magnetic.vector_potential_ground { target = geometry.cell.region.outer_boundary }
    }
    load coil_drive {
        electric_current on geometry.cell.region.coil_in = 2000 A
    }
}

study gear_hardening {
    geometry = geometry.cell
    mesh = mesh.cell
    model = model.hardening_cell
    analysis field {
        electromagnetic harmonic { frequencies = [3, 10, 30] kHz }
        retain loss
    }
    analysis heating after analysis.field {
        thermal transient {
            start_time = 0 s
            end_time = 3 s
            time_step = 0.05 s
            heat_source = analysis.field.state.loss
        }
    }
}
```

No `electromagnetic { equation = ... }` block is needed for a harmonic analysis: the analysis's
frequencies define it. A `harmonic_maxwell` model domain with a `reference_frequency` is also
accepted; the analysis frequencies, not the reference, set what is solved.

## How does the loss heat the part?

`heat_source = analysis.<harmonic>.state.<name>` in a `thermal steady` or `thermal transient`
analysis binds the harmonic analysis's retained state as the thermal heat source. For each frequency
the cycle-averaged Joule loss density of every element, $\tfrac1{V_e}\int_e\tfrac12\sigma|\mathbf
E|^2dV$, becomes the element heat generation of the thermal analysis on the same mesh, so the total
heat equals the electromagnetic loss. The run publishes the harmonic case and then the heating case
(`frequency=<f>Hz/field`, `frequency=<f>Hz/heating`).

For steady current flow with temperature-dependent conductivity, use an
[electrothermal analysis](./electrothermal), which iterates current and temperature.

## How do I apply voltage, current, charge, or current density?

- `electrical.potential` prescribes a terminal potential (static and harmonic).
- `electric_current` injects a total current through terminal faces (uniform normal density).
- `electric_charge_density` is a volume charge for electrostatics.
- `magnetic_volume_current_density` is an impressed current density in a body (magnetostatics and
  harmonic fields).
- `magnetic_vector_potential` holds exterior faces at a known tangential vector potential
  (magnetostatics), for example an imposed external field; those faces count as grounded.

Use direct `load` syntax for constant data and `signal` plus `excitation` for frequency- or
time-dependent data.

## What electrical and magnetic references are required?

- Steady current flow and electrostatics: at least one prescribed potential on each connected
  conductor.
- Magnetostatics: `magnetic.vector_potential_ground` on one connected set of exterior faces
  (normally the far field; include the ends of a conductor whose current leaves the model, since
  current cannot cross an insulated face), and one `magnetic.coulomb_gauge`.
- Harmonic fields: a grounded far field (`magnetic.vector_potential_ground`). A Coulomb gauge is not
  used: conductivity and permittivity fix the field.

RunMat reports a missing reference instead of adding one.

## What material properties does electromagnetic analysis need?

Electrical conductivity, relative permittivity and relative permeability per material. A
`resistive_heating_coefficient` $\alpha$ defines $\sigma(T)=\sigma_{ref}/(1+\alpha(T-T_{ref}))$;
standalone electromagnetic analyses use $\sigma_{ref}$, electrothermal analyses the solved
temperature.

## How accurate are the results?

See [electromagnetic validation](../validation/electromagnetic) for the analytic checks (wire
resistance, parallel-plate and coaxial capacitance, conductor field, skin-effect impedance).
Resolve the skin depth $\delta=\sqrt{2/(\omega\mu\sigma)}$ with several elements where loss matters.

See [electromagnetic constraints](../constraints/electromagnetic),
[electromagnetic loads](../loads/electromagnetic), [electrothermal analysis](./electrothermal),
[typed units](../language/types-units-and-expressions), and [current status](../validation/status.md).
