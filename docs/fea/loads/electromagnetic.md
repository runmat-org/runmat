# Electromagnetic loads and boundaries

Current density appears as source $\mathbf J_s$ in the electromagnetic weak form. Total current is
its surface integral $I=\int_\Gamma\mathbf J\cdot\mathbf n\,d\Gamma$; total charge is
$Q=\int_\Omega\rho_q\,d\Omega$. RunMat keeps these measures distinct during projection.

| Kind | Quantity | Projection/boundary meaning |
| --- | --- | --- |
| `terminal_voltage` | V | Electric-potential signal on a terminal |
| `electric_current` | A | Preserve total current |
| `electric_charge` | C | Preserve total charge |
| `current_density` | $\mathrm{A/m^2}$ | Preserve source intensity |
| `magnetic_vector_potential` | $\mathrm{Wb/m}$ | Vector-potential signal |
| `magnetic.insulation` | no scalar value | Natural magnetic boundary |

```fea
load terminal_current {
    electric_current on geometry.busbar.region.input = 100 A
}

load insulated_exterior {
    magnetic.insulation {
        target = geometry.motor.region.outer_boundary
    }
}
```

## How do I drive a terminal with voltage?

Use `terminal_voltage` when voltage is an applied excitation rather than a permanent essential
constraint:

```fea
load dc_terminal {
    terminal_voltage on geometry.device.region.high_terminal = 48 V
}
```

For a frequency/time sweep use a potential signal and explicit excitation. The model still needs a
reference terminal. Do not prescribe voltage and incompatible integral current on the same terminal.

## How do I apply a voltage difference between two electrodes?

Drive one terminal and establish the other as the electrical reference. For a constant prescribed
potential, use the corresponding electrical constraints; for a swept or time-varying drive, use a
terminal-voltage excitation and retain one reference:

```fea
load high_side_drive {
    terminal_voltage on geometry.device.region.high_terminal = 48 V
}

constraint return_reference {
    electrical.potential_reference {
        target = geometry.device.region.return_terminal
    }
}
```

Do not independently prescribe both voltage and total current unless the formulation defines one as
a measured/control quantity rather than a second boundary condition.

## How do I apply an AC voltage waveform?

Use an electric-potential signal over time for transient drive or over frequency for harmonic
amplitude/phase, then bind it with `terminal_voltage`. A frequency sweep is not a sampled time
waveform. Preserve RMS/peak and complex phase conventions in the study.

## How do I apply a total electric current?

```fea
load coil_current {
    electric_current on geometry.motor.region.coil_terminal = 12 A
}
```

This is a resultant-preserving source. RunMat projects it to canonical source entities and records
source-region coverage/conservation. Use current density when the local distribution is known.

## How do I model a return current or multiple terminals?

Declare sourced and returned terminal resultants with consistent signs only when both are physically
prescribed. Otherwise drive the known terminal and let conservation determine the return through the
referenced conductor network. Check that the algebraic sum of terminal currents is zero for a steady
closed system, accounting for displacement or charge accumulation in transient models.

## How do I apply current to several parallel conductors?

Use one current load per conductor when the split is known. If the split should follow electrical
resistance, prescribe the common voltage/terminal relation and solve it rather than assigning equal
currents. A total current on a parent region must not be duplicated on every child region.

## How do I apply electric charge?

```fea
load electrode_charge {
    electric_charge on geometry.capacitor.region.electrode = 2e-6 C
}
```

Charge preserves the declared total coulombs over the target. Verify whether the intended
formulation expects surface or volume charge and that the selected region role is compatible; the
typed admission path must reject an unsupported target rather than reinterpret it.

## How do I model an electrostatic electrode with fixed charge?

Use `electric_charge` only when total free charge is the controlled quantity. Use electrical
potential when a voltage source controls the electrode. A floating conductor with known net charge
requires the floating/equipotential constraint plus a compatible charge law; setting every node to
a potential would remove the unknown electrode voltage.

## How do I apply current density?

```fea
load distributed_current {
    current_density on geometry.busbar.region.input_face = 5e6 A/m^2
}
```

Current density preserves amperes per square metre. Integrated current follows the resolved face
area and normal direction. Check CAD units, face orientation, and the integrated resultant.

## How do I import measured or computed current-density data?

Use a typed topology field whose coordinate frame, units, mesh/CAD association, interpolation, and
resultant error are retained. Do not paste a vector table without its source topology. Verify
$\int_\Gamma\mathbf J\cdot\mathbf n\,d\Gamma$ against the expected terminal current.

## How do I apply magnetic vector potential as an excitation?

```fea
load imposed_potential {
    magnetic_vector_potential on geometry.motor.region.outer_boundary = 1e-6 T*m
}
```

Use this only when vector-potential data is physically known and supported by the formulation. A
gauge controls nonuniqueness; it is not a source. Magnetic insulation is a natural boundary; it is
not zero vector potential.

## How do I apply magnetic insulation?

```fea
load insulated_boundary {
    magnetic.insulation {
        target = geometry.motor.region.exterior
    }
}
```

The boundary contributes no scalar value. In the experimental edge-graph realization, result
evidence distinguishes canonical boundary projection from ordinal/global-count/unmapped fallback.
Require canonical localization for geometry-sensitive conclusions.

## How do I define a frequency-dependent coil current?

```fea
signal drive_current : electric_current over frequency {
    samples = [
        { at = 50 Hz, value = 12 A },
        { at = 400 Hz, value = 8 A },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation coil_drive {
    electric_current {
        target = geometry.motor.region.coil_terminal
        signal = signal.drive_current
    }
}
```

Frequency samples in the analysis and signal must cover the same requested domain. Complex source
shape/phase must be explicit when required; a real amplitude does not silently define phase.

## How do I drive a three-phase winding?

Give each phase its own signal/excitation and preserve the $120^\circ$ phase offsets and winding
orientation. Include the three excitations in one load case. Do not encode phase by changing only
current signs at all frequencies; a negative real amplitude represents $180^\circ$, not
$\pm120^\circ$.

## How do I model PWM or pulsed electrical drive?

Use a time-domain voltage/current signal with switching edges resolved by the transient procedure,
or an explicitly derived harmonic representation. State extrapolation and switching convention. An
under-resolved square wave can alias loss and force predictions; a frequency sweep of magnitudes does
not preserve harmonic phase by itself.

## Should I drive a conductor by voltage, total current, or current density?

Use voltage when a circuit fixes terminal potential, total current when the integrated conductor
current is known, and current density when the spatial distribution is known. Voltage needs a
reference potential; total current requires a compatible return path; current density requires a
correctly oriented surface or volume role. Specifying more than one independently on the same
terminal can overconstrain the electrical problem.

## How do I model a stranded coil?

Use an `electric_current` excitation on the admitted coil-terminal or coil-region contract when the
total ampere-turn source is represented by that formulation. Do not multiply current by turns unless
the formulation explicitly expects ampere-turns rather than conductor current. Winding direction,
turn count, fill factor, return path, and phase are distinct physical facts and must not be inferred
from a region name.

## How do I model a permanent magnet?

The current closed load set does not expose magnetization or remanent-flux-density as a load.
Magnet behavior belongs in a magnetic material capability with orientation, recoil permeability,
remanence, and temperature dependence. Do not emulate a permanent magnet with an arbitrary vector-
potential boundary value.

## How do I represent skin effect or eddy-current drive?

Apply the harmonic terminal voltage/current/current-density source and select an admitted
frequency-domain electromagnetic formulation whose material and discretization resolve conductive
loss. Skin effect is a solution consequence, not a separate boundary condition. Verify element size
against skin depth $\delta=\sqrt{2/(\omega\mu\sigma)}$ and inspect current-density and loss fields.

## How do I model an open magnetic boundary?

`magnetic.insulation` is not a universal open boundary; it imposes the admitted zero-flux/tangential
relation and can distort an exterior field. Model sufficient surrounding air and perform domain-size
sensitivity, or use a formulation with an infinite-element/open-boundary contract. Do not place
insulation tightly around a device and call the resulting field unbounded.

## How do I apply an external uniform magnetic field?

A uniform background field needs a typed incident/background-field or consistent vector-potential
boundary representation. The current closed load set does not provide a direct magnetic-flux-density
load. Do not insert teslas into `magnetic_vector_potential`, whose dimension and gauge differ.

## How do I obtain electromagnetic force for structural analysis?

Use a retained Maxwell-stress/Lorentz-force field from the electromagnetic procedure and transfer it
through a typed electromechanical coupling. The mapping must conserve force and moment. Applying a
visually estimated structural pressure discards field direction, fringe effects, and provenance.

## How do I couple Joule heating into a thermal analysis?

Use retained electromagnetic loss/power state as the typed source for an electrothermal or staged
thermal analysis. Do not manually re-enter a visually estimated heat-generation value. The transfer
must record mesh/topology mapping and conserve integrated electrical loss.

Total current and charge preserve their resultants during projection; current density preserves
intensity. Voltage and vector-potential excitations carry their physical quantity through the signal
contract. Pair these loads with an explicit electrical potential reference or magnetic gauge; an
unreferenced field must fail rather than receive a hidden numerical anchor.

## Why is the electrical or magnetic model singular?

Typical causes are a missing electric reference, a disconnected conductor without a
floating-conductor contract, a missing vector-potential gauge, or boundaries that leave a field
nullspace. Add the physical reference or gauge declared by the formulation; do not pin an arbitrary
node outside the study definition.

## What should I inspect after applying electromagnetic loads?

Inspect resolved terminal area and orientation, integrated current and charge conservation,
reference and return paths, gauge status, source phase/amplitude convention, skin-depth mesh
adequacy, stored-energy and loss balance, and transferred force or heat resultants.

Further reading: [electromagnetic constraints](../constraints/electromagnetic),
[electromagnetic studies](../physics/electromagnetic), and [electrothermal analysis](../physics/electrothermal).
