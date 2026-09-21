# Acoustic FEA in RunMat: cavity modes, harmonic pressure, and boundary impedance

Acoustic analysis predicts pressure modes and pressure response in a fluid domain. The physical
model needs density and sound speed, a pressure/velocity source, and boundary behavior such as a
rigid wall, radiation condition, or impedance. RunMat does not infer those conditions from CAD face
names or from an enclosed-looking volume.

The `.fea` language checks static, modal, transient, harmonic, and nonlinear acoustic procedure
contracts. The numerical library contains an experimental **reduced acoustic Helmholtz graph**
component, but the public typed `.fea` executor does not connect acoustic studies end to end. It is
not a tetrahedral acoustic FE formulation and must not be presented as one.

## What acoustic equation is represented?

For small-amplitude harmonic pressure $\hat p$ in a homogeneous medium,

$$
\nabla^2\hat p+k^2\hat p=-\hat q,\qquad
k=\frac{\omega}{c},
$$

with pressure, normal-velocity, impedance, rigid-wall, radiation, or periodic boundary conditions.
An impedance boundary has the frequency-domain form

$$
\hat p=Z\hat v_n.
$$

The implemented reduced component constructs a bounded structured graph from the analysis model,
uses nearest-neighbor Laplacian weights $1/h_d^2$, adds $k^2$ and damping/boundary diagonal terms,
maps pressure sources onto graph nodes, and solves the complex Helmholtz system. Its admitted
realization is named `ReducedAcousticHelmholtz`; source evidence distinguishes a hashed-target graph
projection from a synthetic fallback. This component is useful for pipeline integration and
diagnostics, not a production claim for arbitrary CAD acoustics.

## Which acoustic procedures are defined?

| Procedure | Required domain/state | Current numerical status |
| --- | --- | --- |
| `acoustic static` | Static pressure state | Typed contract; reduced component is harmonic, not static |
| `acoustic modal` | Positive mode count | Typed contract only |
| `acoustic transient` | Time samples and compatible initial pressure | Typed contract only |
| `acoustic harmonic` | Frequency samples and complex pressure | Experimental reduced Helmholtz graph component |
| `acoustic nonlinear` | Load-factor domain | Typed contract only |

No unsupported procedure is approximated by silently selecting the reduced harmonic component.

## How do I describe an acoustic cavity modal study?

```fea
model cavity_model {
    assign material.air to geometry.cavity.region.fluid
    constraint pressure_reference {
        acoustic.pressure {
            target = geometry.cavity.region.reference_face
            value = 0 Pa
        }
    }
}

study cavity_modes {
    geometry = geometry.cavity
    mesh = mesh.cavity
    model = model.cavity_model
    formulation = experimental
    analysis modes { acoustic modal { modes = 20 } }
}
```

For a production acoustic eigenproblem, the expected discrete system is
$\mathsf K_p\boldsymbol\phi_i=\omega_i^2\mathsf M_p\boldsymbol\phi_i$. The typed contract carries
mode count and acoustic state, but no native public modal realization is currently admitted.

## How do I apply an acoustic pressure or speaker velocity?

```fea
signal drive : velocity over frequency {
    samples = [
        { at = 100 Hz, value = 0.005 m/s },
        { at = 1000 Hz, value = 0.010 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation speaker {
    normal_velocity {
        target = geometry.cavity.region.driver
        signal = signal.drive
    }
}
```

Use `acoustic_pressure` for a pressure drive and `normal_velocity` for a boundary velocity. The
signal quantity, frequency domain, interpolation, and complex/real shape must match the excitation.
Static direct forms are available when one constant value is sufficient.

## How do I model rigid, radiating, or impedance boundaries?

```fea
load enclosure {
    acoustic.rigid_wall { target = geometry.duct.region.walls }
}

load termination {
    acoustic.impedance {
        target = geometry.duct.region.outlet
        impedance = 415 Pa*s/m
    }
}
```

- `acoustic.rigid_wall` imposes zero normal velocity.
- `acoustic.radiation` represents an outgoing-wave truncation.
- `acoustic.impedance` relates pressure and normal velocity through the declared specific impedance.
- `acoustic.pressure` prescribes the primary pressure unknown.
- periodic and phase-shifted periodic constraints pair repeating acoustic boundaries.

These are different physical models. A radiation boundary is not a generic replacement for a
measured termination impedance, and a pressure reference is not an absorbing boundary.

## Why are my acoustic resonances or sound levels wrong?

Investigate:

- wrong geometry units or sound speed;
- a missing pressure reference for a formulation that needs one;
- rigid-wall assumptions on openings that should radiate;
- source area/direction or impedance units that do not match the experiment;
- insufficient spatial resolution relative to wavelength $\lambda=c/f$;
- frequency samples that skip narrow resonances;
- damping or boundary loss added without physical basis;
- reduced/synthetic realization evidence where full CAD-backed FE accuracy is required.

Pressure amplitude, RMS pressure, and sound-pressure level are not interchangeable. When SPL is
reported, its reference pressure and amplitude convention must be part of the output definition:

$$
L_p=20\log_{10}\!\left(\frac{p_{rms}}{p_{ref}}\right).
$$

See [acoustic constraints](../constraints/acoustic), [acoustic loads](../loads/acoustic),
[fluid–structure interaction](./fluid-structure-interaction), [outputs](../language/outputs-fields-and-observations),
and [current status](../status).
