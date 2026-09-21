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

## How do I apply a total electric current?

```fea
load coil_current {
    electric_current on geometry.motor.region.coil_terminal = 12 A
}
```

This is a resultant-preserving source. RunMat projects it to canonical source entities and records
source-region coverage/conservation. Use current density when the local distribution is known.

## How do I apply electric charge?

```fea
load electrode_charge {
    electric_charge on geometry.capacitor.region.electrode = 2e-6 C
}
```

Charge preserves the declared total coulombs over the target. Verify whether the intended
formulation expects surface or volume charge and that the selected region role is compatible; the
typed admission path must reject an unsupported target rather than reinterpret it.

## How do I apply current density?

```fea
load distributed_current {
    current_density on geometry.busbar.region.input_face = 5e6 A/m^2
}
```

Current density preserves amperes per square metre. Integrated current follows the resolved face
area and normal direction. Check CAD units, face orientation, and the integrated resultant.

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

## How do I couple Joule heating into a thermal analysis?

Use retained electromagnetic loss/power state as the typed source for an electrothermal or staged
thermal analysis. Do not manually re-enter a visually estimated heat-generation value. The transfer
must record mesh/topology mapping and conserve integrated electrical loss.

Total current and charge preserve their resultants during projection; current density preserves
intensity. Voltage and vector-potential excitations carry their physical quantity through the signal
contract. Pair these loads with an explicit electrical potential reference or magnetic gauge; an
unreferenced field must fail rather than receive a hidden numerical anchor.

Further reading: [electromagnetic constraints](../constraints/electromagnetic),
[electromagnetic studies](../physics/electromagnetic), and [electrothermal analysis](../physics/electrothermal).
