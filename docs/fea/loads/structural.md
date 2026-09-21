# Structural loads

Structural loads contribute to the weak-form right-hand side:

$$
\mathbf f_e=\int_{\Omega_e}\mathsf N^{\mathsf T}\mathbf b\,d\Omega
+\int_{\Gamma_e}\mathsf N^{\mathsf T}\bar{\mathbf t}\,d\Gamma.
$$

Pressure uses $\bar{\mathbf t}=-p\mathbf n$. A resultant force/moment is projected so its integrated
resultant is conserved; traction, pressure, and body-force density preserve intensity.

| Kind | Quantity and typical target | Projection invariant |
| --- | --- | --- |
| `force` | Resultant force; compatible point/edge/face target | $\sum\mathbf f_i=\mathbf F$ |
| `moment` | Resultant moment on a compatible target | resultant moment |
| `traction` | Force per area on faces | traction intensity |
| `pressure` | Normal pressure on faces | scalar pressure intensity |
| `body_force` | Force per volume on solid/cell regions | body-force density |
| `structural.pressure` | Constant natural pressure boundary on faces | pressure intensity |

```fea
load bearing_force {
    force on geometry.bracket.region.bore = [0, -12, 0] kN
}

load gravity_like_body_force {
    body_force on geometry.bracket.region.body = [0, -77.0, 0] kN/m^3
}

load vessel_pressure {
    structural.pressure {
        target = geometry.vessel.region.inner_wall
        value = 2 MPa
    }
}
```

## How do I apply a total force to a face, edge, or point?

```fea
load actuator_force {
    force on geometry.bracket.region.actuator_face = [0, -12, 0] kN
}
```

`force` is a resultant. RunMat projects it to compatible solver entities while preserving
$\sum_i\mathbf f_i=\mathbf F$ within the recorded tolerance. The selected region and projection
measure determine the distribution; use traction when a known local stress distribution matters.

For a time-varying scalar amplitude, use a force signal plus direction:

```fea
signal actuator_history : force over time {
    samples = [
        { at = 0 s, value = 0 N },
        { at = 0.2 s, value = 12 kN },
        { at = 1 s, value = 12 kN },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation actuator {
    force {
        target = geometry.bracket.region.actuator_face
        signal = signal.actuator_history
        direction = global([0, -1, 0])
        kinematics = fixed_frame
    }
}
```

## How do I apply a torque or moment?

```fea
load shaft_torque {
    moment on geometry.shaft.region.end_face = [0, 0, 250] N*m
}
```

`moment` preserves the resultant moment during projection. Confirm the frame, sign convention, and
whether a pure moment is physically transferred by the selected fixture. A force offset from an axis
and a pure applied moment are different load models.

## How do I apply traction to a surface?

```fea
load shear_traction {
    traction on geometry.panel.region.loaded_face = [2e6, 0, 0] Pa
}
```

Traction is force per area and preserves local intensity. Its integrated resultant changes with
selected area. Use it for known distributed shear/normal vectors; use pressure for a scalar load
normal to the current/reference surface according to the declared kinematics.

## How do I apply pressure to a structural surface?

For a constant natural boundary:

```fea
load internal_pressure {
    structural.pressure {
        target = geometry.vessel.region.inner_wall
        value = 2 MPa
    }
}
```

For a signal-driven pressure use `pressure on ...` or an explicit `pressure` excitation. Pressure
acts along the resolved surface normal. Verify normal orientation and whether `fixed_frame` or
`follower` kinematics matches the intended nonlinear formulation.

## How do I apply gravity or another body force?

```fea
load vertical_body_force {
    body_force on geometry.bracket.region.body = [0, -77.0, 0] kN/m^3
}
```

`body_force` carries force per volume and targets solid/cell regions. For gravity, the density-
dependent value is $\mathbf b=\rho\mathbf g$; if density varies by material region, author the
appropriate typed gravity/body-force model rather than using one averaged density.

## When should I use `pressure` versus `structural.pressure`?

Both represent structural surface pressure. `structural.pressure` is the explicit constant natural-
boundary block with `target` and `value`. The direct `pressure on region = value` form creates a
constant signal/excitation and participates naturally in load cases; an explicit excitation is used
for varying signals, direction/kinematics metadata, and step/case activation. The resolved physics
must be the same for equivalent constant inputs, and evidence should show the exact source path.

## How do I model a bearing load on a hole?

If only the resultant is known, apply `force` to a named bore-face region and let the admitted
projection preserve the total force. If the bearing-pressure distribution is known, apply traction
or pressure over the physically loaded arc instead. Do not select the full cylindrical face when
only one side carries compression: define the loaded patch in CAD or as an explicit governed region.

```fea
load pin_resultant {
    force on geometry.bracket.region.loaded_bore_patch = [0, -12, 0] kN
}
```

Review the resolved force and moment about the hole axis. A uniform full-bore pressure can cancel
its own resultant and is not equivalent to a pin bearing load.

## How do I apply hydrostatic pressure?

Hydrostatic pressure varies with position. Represent it with a pressure signal/function over the
spatial field when the selected formulation admits spatial functions; a constant
`structural.pressure` is valid only when the pressure variation is negligible. Keep density,
gravity, free-surface location, and pressure datum explicit in the RunMat function so the field is
reproducible and unit checked.

## How do I apply a follower pressure in a nonlinear analysis?

Use an explicit pressure excitation with follower kinematics, not a constant direct load whose
direction remains in the reference frame. Follower pressure updates $-p\mathbf n$ with the deformed
surface and contributes the appropriate load stiffness only when the nonlinear formulation admits
it. The analysis must reject follower kinematics for a formulation that cannot assemble that term.

## How do I model centrifugal loading from a spinning rotor?

For constant angular speed, the continuum source is
$\mathbf b=\rho\,\boldsymbol\omega\times(\boldsymbol\omega\times\mathbf r)$.
Use a spatial body-force field when that field contract is admitted, with an explicit axis, origin,
frame, density source, and angular speed. A uniform `body_force` vector is not centrifugal loading.
For prestressed modal or harmonic response, solve the rotational preload first and consume its
retained structural state in the downstream procedure.

## How do I apply base acceleration or earthquake motion?

Use `structural.prescribed_motion` on the support region with a typed acceleration signal. This is a
kinematic constraint, not a body-force load. Modal response-spectrum and random-response procedures
consume their dedicated spectrum/PSD inputs; do not convert those inputs into an arbitrary force
history. See [structural constraints](../constraints/structural#how-do-i-prescribe-base-acceleration-or-support-motion)
and [structural analysis](../physics/structural).

## How do I represent bolt preload?

RunMat's `structural preload` procedure carries a solved prestress state forward, but this page's
closed load set does not currently declare a first-class bolt-section pretension object. Do not
replace it with an unexplained nodal force or thermal trick. Model a physically justified contact,
interference, pressure, or applied-force setup only when it represents the assembly process, and
retain the resulting preload state. A dedicated bolt pretension contract should be added before
claiming section-cut preload semantics.

A vector-valued direct load carries its magnitude and direction in one typed value. An explicit
excitation may instead name a scalar signal and a normalized `global([x,y,z])` direction. Force and
moment projection preserve resultants; traction and pressure preserve intensity. Follower
kinematics are available only where the formulation can update direction with deformation.

Group alternatives or combinations explicitly:

```fea
load_case operating {
    include load.bearing_force
    include load.vessel_pressure
}
```

Check the resolved resultant, moment, target area/volume, sign convention, active analysis steps,
and reaction balance. Further reading: [structural constraints](../constraints/structural) and
[structural studies](../physics/structural).
