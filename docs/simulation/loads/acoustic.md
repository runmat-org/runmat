# Acoustic loads

Acoustic pressure and normal velocity are conjugate boundary variables. A rigid wall imposes
$v_n=0$; an impedance boundary imposes $p=Zv_n$; a radiation boundary approximates an outgoing-wave
relation. They change the Helmholtz boundary operator in different ways.

The native typed executor supports standalone canonical acoustic harmonic studies;
see the retained [authored duct evidence](../qualification/acoustic-right-lu.md). It does not cover
every acoustic declaration and is not a general exterior-wave solver.

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `acoustic_pressure` | Pa | Pressure signal on an acoustic boundary |
| `incident_pressure` | Pa | Uniform normally incoming plane-wave pressure, not total pressure |
| `normal_velocity` | m/s | Drive a boundary through normal velocity |
| `volume_velocity` | m³/s | Total outward volume flow over a finite exterior patch |
| `acoustic.rigid_wall` | `target` | Zero normal velocity |
| `acoustic.radiation` | `target` | Outgoing-wave truncation |
| `acoustic.impedance` | `target`, `impedance` | $p=Zv_n$ |

Every implemented acoustic action and boundary on this page targets a face region. Pressure and
normal velocity are scalar normal quantities; they do not accept a separate global direction.

## How do I inject incoming pressure without fixing the reflected pressure?

```fea
excitation incoming_wave {
    incident_pressure {
        target = geometry.duct.region.inlet
        signal = signal.phased_pressure
    }
}
```

The signal supplies **incoming** pressure. Total pressure remains an unknown, allowing a reflected
wave to leave through the same face. For peak phasors with `exp(+i omega t)`, the operator uses
`v_n = Y (p - 2 p_inc)`, where `Y = 1/(rho*c)` comes from the unique exterior owning fluid.
Reported input power is net incoming power (incoming minus reflected); it is not twice incident
power from Robin source-work bookkeeping. The source has the same shared scalar pressure-signal
and projection-provenance contract as other excitations, without private amplitude/phase arrays.

This bounded implementation requires coplanar equally oriented exterior faces in one homogeneous
medium, uniform distribution, fixed frame and no separate direction. Conflicting sources,
natural boundaries, pressure prescriptions and periodic pairing reject. Curved, oblique and
spatially varying incidence are not supported by this action. Do not add a separate impedance
boundary on its target: the incoming action already includes the matching outgoing operator.

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

## How do I apply acoustic pressure with phase?

Use complex pressure samples. A negative real scalar represents a $180^\circ$ phase change; other
phases require both components:

```fea
signal phased_pressure : pressure over frequency {
    samples = [
        { at = 250 Hz, value = { real = 1 Pa, imaginary = 0 Pa } },
        { at = 500 Hz, value = { real = 0 Pa, imaginary = -1 Pa } },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation phased_driver {
    acoustic_pressure {
        target = geometry.cavity.region.driver
        signal = signal.phased_pressure
    }
}
```

The second sample has phase $-90^\circ$ under the declared harmonic convention. Preserve that
convention when comparing source power, pressure, and particle velocity.

## How do I apply a transient pressure pulse?

The signal language can represent pressure over `time`, but the current typed procedure set has no
acoustic transient procedure to consume it. The following declaration is typeable signal/excitation
data, not a runnable acoustic analysis today:

```fea
signal pressure_pulse : pressure over time {
    samples = [
        { at = 0 ms, value = 0 Pa },
        { at = 1 ms, value = 20 Pa },
        { at = 3 ms, value = 0 Pa },
    ]
    interpolation = linear
    extrapolation = zero
}

excitation pulsed_source {
    acoustic_pressure {
        target = geometry.cavity.region.source_face
        signal = signal.pressure_pulse
    }
}
```

A future transient procedure must resolve the pulse bandwidth. A short rise time can require a much
smaller time step and finer acoustic mesh than its total duration suggests.

## How do I drive an acoustic boundary with normal velocity?

```fea
load piston_velocity {
    normal_velocity on geometry.cavity.region.piston = 0.01 m/s
}
```

Normal velocity models piston/diaphragm motion without prescribing pressure. Its sign follows the
resolved outward normal. For coupled structural motion, use FSI/acoustic-structure transfer rather
than manually duplicating a velocity trace.

## How do I prescribe total volume flow rather than speed?

```fea
signal total_flow : volume_flow_rate over constant { value = -0.01 m^3/s }

excitation volume_driver {
    volume_velocity {
        target = geometry.cavity.region.piston
        signal = signal.total_flow
    }
}
```

`volume_velocity` uses the shared scalar `volume_flow_rate` signal. The total is
distributed once over the complete selected exterior patch as `v_n=Q/A`, not once
per face. Its sign is outward-positive; a negative value injects flow into the
domain. m/s and kg/s signals are different quantities and reject. There is no
separate direction, follower motion or moment projection. Complex frequency
samples retain both components through the ordinary shared signal contract.

This bounded CPU implementation is under qualification; see the
[volume-flow evidence and remaining gates](../qualification/acoustic-volume-velocity.md).
It is a finite boundary source, not an interior monopole or an arbitrary spatial
distribution. Do not overlap another acoustic source, pressure prescription,
rigid/impedance boundary or paired periodic relation on its patch.

## How do I apply a frequency-dependent piston velocity?

Use a velocity signal over frequency and an explicit normal-velocity excitation:

```fea
signal piston_spectrum : velocity over frequency {
    samples = [
        { at = 50 Hz, value = 0.2 mm/s },
        { at = 500 Hz, value = 1.0 mm/s },
        { at = 2 kHz, value = 0.3 mm/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation piston_drive {
    normal_velocity {
        target = geometry.cavity.region.piston
        signal = signal.piston_spectrum
    }
}
```

The scalar is resolved along the acoustic boundary normal. Reverse geometry orientation or the
signal sign deliberately; do not add an arbitrary global direction to a normal-velocity condition.

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

Specific acoustic impedance relates pressure to normal velocity, $p=Zv_n$, with $v_n$ along the
outward normal. A quantity is a constant real (resistive) impedance; a complex or
frequency-dependent impedance comes from a `table` (below). Check that the value is specific
impedance rather than total mechanical impedance.

## How do I model a resistive acoustic liner?

Use `acoustic.impedance` with the measured or derived *specific* impedance in `Pa*s/m`. For a purely
resistive liner, a positive real value dissipates energy under the formulation's normal/sign
convention. Apply it only to the lined faces:

```fea
load resistive_liner {
    acoustic.impedance {
        target = geometry.duct.region.liner_faces
        impedance = 650 Pa*s/m
    }
}
```

Do not multiply specific impedance by area before entering it. Mechanical impedance in `N*s/m` and
surface-specific acoustic impedance are not interchangeable.

## How do I model a frequency-dependent complex impedance?

Declare a top-level `table` of impedance over frequency and reference it from the boundary. A
measured liner is a list of resistance and reactance samples; harmonic solves interpolate them
linearly in frequency and hold the end values beyond the first and last sample:

```fea
table liner_data : impedance over frequency {
    samples = [
        { at = 125 Hz, resistance = 420 Pa*s/m, reactance = -900 Pa*s/m },
        { at = 250 Hz, resistance = 430 Pa*s/m, reactance = -380 Pa*s/m },
        { at = 500 Hz, resistance = 440 Pa*s/m, reactance = -120 Pa*s/m },
    ]
}

model lined_duct {
    load liner {
        acoustic.impedance {
            target = geometry.duct.region.liner_faces
            impedance = table.liner_data
        }
    }
}
```

A rigidly backed porous layer (mineral wool, foam, fibrous ceiling tiles) is described by its flow
resistivity and thickness with the Delany–Bazley model, which gives the layer's characteristic
impedance and wavenumber in the adjacent air; the surface impedance is
$Z=-iZ_c\cot(k_cd)$:

```fea
table ceiling_tile : impedance over frequency {
    model = delany_bazley
    flow_resistivity = 20000 Pa*s/m^2
    thickness = 50 mm
}
```

Reactance follows the $e^{+i\omega t}$ convention of the harmonic solver: a stiffness-like (spring)
surface has negative reactance and a mass-like surface positive reactance. Resistance must be
nonnegative. Impedance tables apply to harmonic studies; modal and transient studies take rigid
walls, pressure release and constant impedance.

## How do I model a perfectly matched layer (PML)?

`acoustic.radiation` is a boundary approximation, not a PML. A PML requires a transformed exterior
volume, thickness and coordinate-stretch law. The current closed load set has no first-class PML
object. Extend the acoustic formulation with a typed PML domain before claiming PML behavior; do not
rename a radiation boundary.

## How do I model a loudspeaker or vibrating diaphragm?

Use `normal_velocity` when diaphragm velocity is known, or an `acoustic_pressure` excitation when
the boundary pressure is known. If diaphragm motion must be solved from structural dynamics, use an
admitted acoustic-structure/FSI transfer instead of imposing the same guessed motion on both
physics. The sign of normal velocity follows the acoustic domain's outward normal.

## How do I model structure-borne sound from a vibrating panel?

Use a coupled acoustic-structure interface so structural normal velocity is transferred to the
acoustic boundary and acoustic pressure returns traction to the structure when two-way coupling is
admitted. A copied `normal_velocity` trace is a one-way imposed approximation; if used, document the
source run and topology mapping as evidence rather than presenting it as coupled response.

The current interface catalog exposes a fluid-structure interface, not a dedicated
acoustic-structure interface declaration. Do not present that interface as acoustic coupling
without a procedure that explicitly binds structural and acoustic states.

## How do I model a monopole or point acoustic source?

The current public load set is boundary-based and does not expose a point-monopole or volume source
strength. Smearing a point source over an arbitrary tiny face changes its units and mesh dependence.
A first-class source needs a typed volume-velocity/source-strength quantity, location or region,
regularization, and power-balance evidence.

## How do I model an incident plane wave?

An incident-field/scattered-field decomposition needs propagation direction, amplitude, phase,
medium properties, and a compatible truncation boundary. It is not equivalent to uniform pressure
on every exterior face. The current closed declarations do not expose an incident-plane-wave
object; use an admitted formulation with that typed contract or author the physically equivalent
boundary data only when it is known on the entire truncation surface.

## How do I model diffuse-field or random acoustic excitation?

A diffuse field is a spatially correlated stochastic field, not one scalar pressure history. It
requires cross-spectral density/coherence and directional sampling. Do not attach the same random
signal independently to every face. Use a stochastic acoustic procedure only when it preserves the
correlation model and reports the convention used.

## How do I model an anechoic termination or open end?

Use `acoustic.radiation` only when its outgoing-wave approximation is appropriate at the truncation
surface. Use `acoustic.impedance` for a measured or designed termination relation. A rigid wall
reflects energy and is the opposite of an anechoic termination. Check reflection sensitivity by
moving the truncation boundary or comparing against a known duct mode.

## How do I apply a sound-pressure level in decibels?

The typed load quantity is pressure in Pa, not dB. Convert an RMS sound-pressure level using
$p_{rms}=p_{ref}10^{L_p/20}$ with the correct reference pressure for the medium, then state whether
the harmonic solver expects RMS, peak, or complex amplitude. Do not enter the dB number as pascals.

For air, $p_{ref}$ is commonly $20\,\mu\text{Pa}$; other media and standards can use a different
reference. A level alone still does not state phase or spatial coherence. Record amplitude convention
and reference pressure in the study inputs/evidence.

## How do I combine several speakers or acoustic sources?

Give each independently controlled source its own signal and excitation, include them in an explicit
load case, and preserve their relative phases. Adding scalar magnitudes loses interference. For
uncorrelated random sources, combine response spectra according to the stochastic procedure rather
than assigning arbitrary deterministic phase.

## Why is the acoustic source producing the wrong sign or phase?

Inspect the resolved face normal, source sign, harmonic $e^{+i\omega t}$ or $e^{-i\omega t}$
convention, and primary-to-secondary orientation of any periodic boundary. A normal-velocity sign
error can look like a 180-degree phase error. Verify against a simple duct or cavity solution before
interpreting a complex assembly.

## What should I inspect after applying acoustic loads?

Inspect resolved target area, normal orientation, input pressure/velocity convention, achieved
boundary resultant or power, reflected flux at truncations, and global acoustic energy balance.
For frequency sweeps, inspect interpolation coverage and confirm no solver sample relied on rejected
or unintended extrapolation.

## How do I prescribe acoustic power?

The current closed load set has pressure and normal-velocity drives, not a direct total acoustic-
power declaration. Converting watts to pressure requires radiation area, impedance, mode shape, and
amplitude convention. Author one of the supported boundary variables only when those assumptions
are known; otherwise a first-class resultant acoustic-power source is required.

Use an excitation with a pressure or velocity signal for transient or harmonic drive. A rigid wall,
radiation boundary, and impedance boundary describe different physical terminations; none should be
selected from geometry names automatically.

## How do I convert sound-pressure level to an acoustic pressure input?

For a stated reference pressure $p_{ref}$ and RMS convention,
$p_{rms}=p_{ref}10^{L_p/20}$. Convert to peak only if the harmonic procedure expects peak amplitude.
Then apply the pressure with its phase as an `acoustic_pressure` signal. State whether the level is
free-field, diffuse-field, incident, or total pressure; the same decibel value does not define the
boundary wave decomposition.

## How do I model a duct inlet and anechoic outlet?

Drive the inlet with `normal_velocity` or `acoustic_pressure` and terminate the outlet with
`acoustic.radiation` or a characterized `acoustic.impedance`. Keep wall regions under
`acoustic.rigid_wall` unless wall compliance is modeled through an FSI interface. The outlet must be
far enough from strong evanescent content for the selected termination law to be valid.

```fea
load duct_inlet {
    normal_velocity on geometry.duct.region.inlet = 0.02 m/s
}

load open_end {
    acoustic.radiation {
        target = geometry.duct.region.outlet
    }
}
```

## How do I model a cavity resonance test?

Use rigid or impedance wall boundaries matching the test article, drive a small known pressure or
normal velocity on the physical source patch, and sweep frequency with sufficient resolution around
expected modes. Request complex pressure at microphone or probe locations and wall/source power. A
uniform pressure constraint on the entire cavity suppresses rather than excites many cavity modes.

## How do I verify acoustic energy balance?

Compare time-averaged input power at driven boundaries with radiated, absorbed, and dissipated power.
Inspect reflected flux at truncations and phase at probes, not only pressure magnitude. Mesh and
frequency refinement must resolve wavelength and boundary layers required by the formulation; a
stable-looking SPL plot with nonclosing power balance is not sufficient evidence.

## How do I choose between pressure and normal-velocity excitation?

Use `acoustic_pressure` when boundary pressure is the controlled source. Use `normal_velocity` when
the normal motion of a piston, diaphragm, or wall is known. Pressure is effort-like and normal
velocity is flow-like; prescribing both on the same boundary generally overconstrains the acoustic
port.

For a source face $\Gamma$, time-averaged complex power is
$\bar P=\tfrac12\operatorname{Re}\int_\Gamma p\,v_n^*\,d\Gamma$. Request both conjugate quantities
from the solution even when only one is prescribed, then check sign and delivered power.

## How do I drive a speaker diaphragm from measured motion?

Convert the measured normal surface velocity into a typed velocity signal, retaining whether the
measurement is peak, RMS, or complex amplitude. Apply it only to the radiating diaphragm region.

```fea
signal diaphragm_velocity : velocity over frequency {
    samples = [
        { at = 100 Hz, value = { real = 0.01 m/s, imaginary = 0 m/s } },
        { at = 1000 Hz, value = { real = 0.004 m/s, imaginary = -0.003 m/s } },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation speaker_motion {
    normal_velocity {
        target = geometry.enclosure.region.diaphragm
        signal = signal.diaphragm_velocity
    }
}
```

Confirm face-normal orientation and compare integrated volume velocity
$Q=\int_\Gamma v_n\,d\Gamma$ with the measurement. A single uniform value is an approximation when
the diaphragm bends. The current `.fea` excitation grammar does not accept a spatial complex
velocity field, so measured nonuniform motion needs a future topology-field contract that preserves
coordinates and phase.

## How do I model a muffler, silencer, or lined duct?

Drive the inlet with pressure or normal velocity, apply `acoustic.impedance` to the lined faces, and
terminate the outlet with `acoustic.radiation` or a validated impedance. Keep rigid casing walls as
`acoustic.rigid_wall`. Compute incident and transmitted power or a two-port quantity; pressure level
at one point is not insertion loss.

Check wavelength resolution, liner impedance convention, reflected power at the outlet, and energy
absorbed by the liner. If mean flow materially changes propagation, use a formulation that owns that
coupling rather than modifying sound speed by hand.

## How do I model two coherent acoustic sources?

Create one excitation and complex signal per independently controlled source. Their phases determine
interference. A 180-degree phase difference is represented by the complex signal, not by reversing
the target face normal. Include both excitations in the same case and inspect complex pressure and
power before converting to SPL.

For incoherent or partially coherent sources, deterministic phasor addition is not valid. The study
needs cross-spectral density or an admitted random-acoustic contract; do not approximate
incoherence by choosing arbitrary phases.

## How do I verify a one-dimensional duct source and termination?

Use a straight, constant-area duct with a source at one end and the chosen radiation or impedance
condition at the other. Request complex pressure and normal velocity on at least two sections. Check
the time-averaged power

$$
\bar P=\tfrac12\operatorname{Re}\int_\Gamma p v_n^*\,d\Gamma
$$

and compare the measured reflection coefficient with the analytic termination value. Repeat with a
finer mesh and a longer duct or moved truncation plane. Pressure magnitude alone cannot distinguish
a correct traveling wave from a standing wave produced by the wrong normal, impedance, or phasor
sign convention.

Further reading: [acoustic constraints](../constraints/acoustic),
[acoustic studies](../physics/acoustic), and
[fluid–structure interaction](../physics/fluid-structure-interaction).
