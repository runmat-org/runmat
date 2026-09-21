# Acoustic loads and boundaries

Acoustic pressure and normal velocity are conjugate boundary variables. A rigid wall imposes
$v_n=0$; an impedance boundary imposes $p=Zv_n$; a radiation boundary approximates an outgoing-wave
relation. They change the Helmholtz boundary operator in different ways.

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `acoustic_pressure` | Pa | Pressure signal on an acoustic boundary |
| `normal_velocity` | m/s | Drive a boundary through normal velocity |
| `acoustic.rigid_wall` | `target` | Zero normal velocity |
| `acoustic.radiation` | `target` | Outgoing-wave truncation |
| `acoustic.impedance` | `target`, `impedance` | $p=Zv_n$ |

```fea
load driver_velocity {
    normal_velocity on geometry.cavity.region.driver = 0.01 m/s
}

load absorbing_end {
    acoustic.impedance {
        target = geometry.duct.region.outlet
        impedance = 415 Pa*s/m
    }
}
```

## How do I apply an acoustic pressure source?

```fea
signal source_pressure : pressure over frequency {
    samples = [
        { at = 100 Hz, value = 1 Pa },
        { at = 1000 Hz, value = 2 Pa },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation speaker_pressure {
    acoustic_pressure {
        target = geometry.cavity.region.driver
        signal = signal.source_pressure
    }
}
```

The signal owns frequency/amplitude data; the excitation owns the boundary target. If phase matters,
use the complex signal shape supported by the procedure instead of treating magnitude as a complete
complex pressure.

## How do I drive an acoustic boundary with normal velocity?

```fea
load piston_velocity {
    normal_velocity on geometry.cavity.region.piston = 0.01 m/s
}
```

Normal velocity models piston/diaphragm motion without prescribing pressure. Its sign follows the
resolved outward normal. For coupled structural motion, use FSI/acoustic-structure transfer rather
than manually duplicating a velocity trace.

## How do I model a rigid acoustic wall?

```fea
load enclosure_wall {
    acoustic.rigid_wall { target = geometry.cavity.region.walls }
}
```

A rigid wall sets normal velocity to zero. It reflects acoustic energy; it is not absorbing. Use it
only where wall compliance and loss are negligible for the frequency range of interest.

## How do I model an open or radiating acoustic boundary?

```fea
load open_end {
    acoustic.radiation { target = geometry.duct.region.opening }
}
```

Radiation approximates outgoing waves at a truncated domain. Place the boundary where its admitted
approximation is valid and test domain-size sensitivity. It is not interchangeable with a measured
termination impedance or a perfectly matched layer.

## How do I apply an acoustic impedance boundary?

```fea
load liner {
    acoustic.impedance {
        target = geometry.duct.region.lined_wall
        impedance = 500 Pa*s/m
    }
}
```

Specific acoustic impedance relates pressure to normal velocity. The current constant field is real;
frequency-dependent complex liners need a compatible signal/material formulation and cannot be
encoded by dropping their phase. Check whether the value is specific impedance rather than total
mechanical impedance.

## How do I model a loudspeaker or vibrating diaphragm?

Use `normal_velocity` when diaphragm velocity is known, or an `acoustic_pressure` excitation when
the boundary pressure is known. If diaphragm motion must be solved from structural dynamics, use an
admitted acoustic-structure/FSI transfer instead of imposing the same guessed motion on both
physics. The sign of normal velocity follows the acoustic domain's outward normal.

## How do I model an anechoic termination or open end?

Use `acoustic.radiation` only when its outgoing-wave approximation is appropriate at the truncation
surface. Use `acoustic.impedance` for a measured or designed termination relation. A rigid wall
reflects energy and is the opposite of an anechoic termination. Check reflection sensitivity by
moving the truncation boundary or comparing against a known duct mode.

## How do I apply a sound-pressure level in decibels?

The typed load quantity is pressure in Pa, not dB. Convert an RMS sound-pressure level using
$p_{rms}=p_{ref}10^{L_p/20}$ with the correct reference pressure for the medium, then state whether
the harmonic solver expects RMS, peak, or complex amplitude. Do not enter the dB number as pascals.

## How do I prescribe acoustic power?

The current closed load set has pressure and normal-velocity drives, not a direct total acoustic-
power declaration. Converting watts to pressure requires radiation area, impedance, mode shape, and
amplitude convention. Author one of the supported boundary variables only when those assumptions
are known; otherwise a first-class resultant acoustic-power source is required.

Use an excitation with a pressure or velocity signal for transient or harmonic drive. A rigid wall,
radiation boundary, and impedance boundary describe different physical terminations; none should be
selected from geometry names automatically.

Further reading: [acoustic constraints](../constraints/acoustic),
[acoustic studies](../physics/acoustic), and [fluid–structure interaction](../physics/fluid-structure-interaction).
