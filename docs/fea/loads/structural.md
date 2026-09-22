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

## How do I distribute a known force over a face?

Apply `force` to the face when the total resultant is known and RunMat should distribute it through
the admitted projection. Apply `traction` when force per area is known. These are not interchangeable:
doubling target area leaves a resultant force unchanged but doubles the integrated force from an
unchanged traction.

## How do I apply a force at a remote point?

Create a reference-point coupling from the attachment region to the physical load application point,
then apply the resultant through that admitted reference-point target. Choose rigid coupling only if
the attachment truly moves rigidly; use distributing coupling when the face may deform. Applying the
same force directly to the face omits the moment $\mathbf r\times\mathbf F$ created by the offset.

## How do I ramp, pulse, or cycle a force?

Use a `signal` over time and an explicit force excitation, as in `actuator_history` above. For a
repeating waveform, state the full time interval and extrapolation behavior; do not assume the
signal repeats after its last sample. A fatigue cycle count is an analysis/procedure input, not a
force unit.

## How do I apply a sinusoidal harmonic force?

Use a force signal over frequency with the amplitude and phase convention admitted by the harmonic
procedure. Keep direction in the excitation and amplitude in the signal. A frequency sweep describes
many harmonic solves; it is not a time-history sine wave sampled at the sweep frequencies.

## How do I apply an impact load?

Use a time-domain force or pressure signal with the measured pulse shape. Preserve impulse
$\int F(t)\,dt$, peak convention, rise time, and application region. A single static peak force does
not reproduce dynamic amplification, while an unrealistically sharp sample transition introduces
high-frequency content that must be resolved by the mesh and time integrator.

## How do I apply a torque or moment?

```fea
load shaft_torque {
    moment on geometry.shaft.region.end_face = [0, 0, 250] N*m
}
```

`moment` preserves the resultant moment during projection. Confirm the frame, sign convention, and
whether a pure moment is physically transferred by the selected fixture. A force offset from an axis
and a pure applied moment are different load models.

## How do I apply combined force and torque?

Declare a force and a moment as separate named loads on the compatible attachment target and include
both in one load case. This preserves each resultant and makes their origins reviewable:

```fea
load shaft_thrust {
    force on geometry.shaft.region.end_face = [0, 0, 8] kN
}

load shaft_drive_torque {
    moment on geometry.shaft.region.end_face = [0, 0, 250] N*m
}

load_case drive {
    include load.shaft_thrust
    include load.shaft_drive_torque
}
```

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

## How do I apply wind or aerodynamic pressure without running CFD?

Apply measured or code-derived pressure/traction to named exterior regions and document the source,
sign, spatial distribution, and load combination. Uniform `structural.pressure` is suitable only
when the engineering approximation is uniform. If pressure varies spatially, use an admitted field
mapping; do not replace it with one unexplained average. This is a structural load model, not a CFD
solution.

## How do I map pressure from a fluid result onto a structure?

Use a typed fluid-structure interface so topology mapping, traction conservation, time/frequency
sampling, and provenance are retained. Copying nodal values between unrelated meshes outside the
interface loses the conservation evidence needed to trust the load transfer.

## How do I apply gravity or another body force?

```fea
load vertical_body_force {
    body_force on geometry.bracket.region.body = [0, -77.0, 0] kN/m^3
}
```

`body_force` carries force per volume and targets solid/cell regions. For gravity, the density-
dependent value is $\mathbf b=\rho\mathbf g$; if density varies by material region, author the
appropriate typed gravity/body-force model rather than using one averaged density.

## How do I apply self-weight to a multi-material assembly?

Self-weight is $\rho_r\mathbf g$ in each material region $r$. Use a gravity/body-force realization
that reads the assigned material density, or declare region-specific body-force densities. A single
average value preserves neither local stress nor total weight when densities differ. Verify
$\int_\Omega \rho\mathbf g\,d\Omega$ against the expected assembly weight.

## How do I apply inertial acceleration to the whole model?

For a known nonrotating frame acceleration $\mathbf a$, the equivalent body force is
$\mathbf b=-\rho\mathbf a$. Prefer a typed inertial/base-motion contract when dynamic relative
motion matters. A body-force approximation is appropriate only when its frame, density source, and
quasi-static assumption are explicit.

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

## How do I model an unbalanced rotating mass?

An imbalance $m_e e$ rotating at angular speed $\omega$ produces force amplitude
$F=m_e e\omega^2$ with two quadrature components. Use a harmonic force excitation with the correct
phase relationship and attachment point, or a rotor-dynamics formulation that owns the rotating
frame. A static centrifugal body force represents mean rotational stress, not synchronous imbalance.

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

## How do I apply thermal expansion as a structural load?

Thermal strain is a constitutive input,
$\boldsymbol\varepsilon_{th}=\boldsymbol\alpha(T-T_{ref})$, not a surface force. Assign thermal
expansion properties by material region and transfer a temperature field through a thermomechanical
procedure. Converting temperature change into an arbitrary nodal force is valid only for a separately
derived reduced model and should not be presented as continuum thermal expansion.

## How do I combine dead, live, pressure, and accidental load cases?

Name each physical load once, build explicit `load_case` declarations, and use study inputs/sweeps
for factors or alternatives. Keep design-code combination factors separate from raw physical loads
so evidence can report both. Do not duplicate a pressure or force under several names merely to
change its factor.

## Why do reactions not balance the applied force?

Check dynamic inertia, contact/interface forces, body-load volume, omitted model fractions,
reference-point moments, and projection evidence before treating it as solver error. In static
equilibrium, resolved external resultants and reactions should balance within tolerance. In a
transient solve, the balance also contains mass times acceleration and damping forces.

## Why is a concentrated load causing a stress singularity?

A point force, sharp re-entrant corner, or abrupt constraint can produce a nonconvergent local
stress. Model the real contact/loaded area or coupling when it is known, and assess converged
resultants or stresses away from the singularity. Mesh refinement alone cannot make an ideal
continuum singularity finite.

## What should I inspect after applying structural loads?

Inspect resolved target topology and measure, projected force and moment error, pressure/traction
normal, active load case and step, material-weight total for body loads, and global reaction balance.
For signal-driven loads, verify domain coverage, interpolation, extrapolation, amplitude convention,
and frame/follower kinematics.

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
and reaction balance.

## How do I model snow, occupancy, or other area design loads?

Convert the design action into the physical pressure or traction acting on the modeled surface,
including tributary-area and direction conventions. Use `structural.pressure` for a constant normal
intensity or `traction` for a vector intensity:

```fea
load roof_snow {
    structural.pressure {
        target = geometry.roof.region.loaded_panels
        value = 1.2 kPa
    }
}
```

Do not enter a total building force as pascals or apply the full area load to every repeated panel.
For code combinations, keep the characteristic load and combination factor separately traceable.

## How do I apply a load in a local coordinate system?

Direct vector loads are expressed in the vector's declared frame. For signal-driven scalar force,
keep amplitude in the signal and direction in the excitation. Use an admitted local frame when the
load follows a shaft, beam, or fixture axis; otherwise use `global([x, y, z])`. The checker must
reject a frame that is unavailable at the target. Review the transformed vector and resultant in
global coordinates before interpreting reactions.

## How do I model a line load on a beam or edge?

Use a compatible edge or beam target and the load kind whose dimensions match the known quantity. A
resultant `force` preserves total newtons over the selected edge; a distributed line load requires a
force-per-length contract. Do not enter `N/m` into `traction`, which is force per area. When no
first-class line-intensity kind is admitted, use resolved beam loads or a justified resultant rather
than silently changing dimensions.

## How do I confirm structural load mesh convergence?

Track both the load invariant and the response. Across refinements, compare resolved target measure,
integrated force and moment, reaction balance, displacement and energy norms, and stresses away from
singular load/support edges. A resultant load should retain total force while its nodal vector
changes; a pressure should retain intensity while its total follows the admitted area. If either
invariant drifts, the projection or region selection changed and the response comparison is not a
clean convergence study.

## How do I load a pressure vessel correctly?

Apply `structural.pressure` to every wetted face, with the sign checked against the outward CAD
normal. A cut model also needs the axial resultant that the omitted end cap would have carried. For
a closed circular vessel with internal pressure $p$ and internal radius $r_i$, that resultant is
$F_z=p\pi r_i^2$. Apply it through a physically justified end-face traction, coupling, or modeled end
cap; pressure on the cylindrical wall alone produces hoop loading but omits the closed-end load path.

```fea
load internal_pressure {
    structural.pressure {
        target = geometry.vessel.region.wetted_faces
        value = 2.5 MPa
    }
}

load omitted_end_cap {
    force on geometry.vessel.region.cut_end = [0, 0, 314159] N
}
```

For a symmetry sector, scale only resultants applied to the cut representation; pressure remains an
intensity. Check integrated pressure force, end-cap force, reactions, membrane stress away from
discontinuities, and sensitivity to end-condition idealization.

## How do I apply a harmonic load with amplitude and phase?

Represent each independently phased component with its own frequency-domain signal. The structural
harmonic procedure consumes complex forcing; a real scalar without an explicit convention means
zero phase, not an unknown phase. Do not replace two out-of-phase shakers with the algebraic sum of
their amplitudes.

```fea
signal shaker_a : force over frequency {
    samples = [
        { at = 100 Hz, value = { real = 100 N, imaginary = 0 N } },
        { at = 500 Hz, value = { real = 100 N, imaginary = 0 N } },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation shaker_a_drive {
    force {
        target = geometry.bracket.region.shaker_a
        signal = signal.shaker_a
        direction = global([0, 0, 1])
        kinematics = fixed_frame
    }
}
```

Use a second signal for the second phase. Inspect complex reaction balance and phase at response
probes; magnitude alone cannot reveal a reversed phase or coordinate axis.

## How do I run a base-shake, response-spectrum, or random-vibration study?

These inputs act through the constrained support, not as equivalent point forces:

- For a time-history base shake, use `structural.prescribed_motion` with an acceleration signal.
- For modal response spectrum, supply the spectrum and direction to the response-spectrum procedure;
  retain modal participation, missing-mass, and combination evidence.
- For random response, supply the admitted acceleration or force PSD and its normalization; retain
  frequency integration and modal truncation evidence.

A PSD has squared quantity per hertz (for example $(m/s^2)^2/Hz$); an SRS or response spectrum is a
peak response versus natural frequency and damping. They are not interchangeable load tables. See
[structural studies](../physics/structural) for the procedure declarations and use this page for the
underlying force, pressure, and support-motion semantics.

## How do I load a hole, pin, clevis, or lug without creating a point singularity?

Prefer resolved contact when the pressure distribution matters. For a reduced model, distribute the
known resultant over the bearing arc or couple the hole surface to a reference point and load that
point. Preserve both force and moment about the chosen reference. A uniform pressure over the full
cylindrical hole generally creates the wrong direction and may cancel its own resultant.

Archive the selected arc, reference point, coupling weights, integrated force/moment, and stress
exclusion distance. Compare the reduced representation with a contact model when local bearing,
fretting, or peak stress drives the decision.

Further reading: [structural constraints](../constraints/structural) and
[structural studies](../physics/structural).
