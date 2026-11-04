# Fluid loads and boundary inputs

A positive `mass_flow` rate enters the fluid. With the outward domain normal,
$\dot m_{\mathrm{in}}=-\int_\Gamma\rho\mathbf u\cdot\mathbf n_{\mathrm{out}}\,d\Gamma$.
Boundary-balance reports use the outward sign, so the report for a positive inlet source is negative.
Inlet velocity and wall motion prescribe kinematics; outlet pressure supplies the natural pressure trace.

The canonical Tet4 solver realizes a signed total mass rate as uniform normal speed on the exact
selected faces, using their total area and the admitted density. It checks the realized total against
the source's projection tolerance. A complete velocity or pressure law on the same faces is an
unsupported overlap; a pressure reference supplies a gauge rather than another face law.

`runmat run` realizes every load on this page in steady and transient flow, standalone or coupled to
heat transfer or a structure. `inlet_velocity` and `wall_motion` prescribe the face velocity from a
vector signal, or from a scalar speed history with a unit `direction`; wall motion is tangential,
and wall motion normal to the wall comes from a coupled structure. See the
[signed-source witness](../qualification/fluid-mass-flow-native.md) for mass flow.

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `inlet_velocity` | m/s | Apply an inlet-velocity signal |
| `mass_flow` | kg/s | Preserve total mass-flow rate |
| `wall_motion` | m/s | Drive wall velocity |
| `fluid.outlet_pressure` | `target`, `value` in Pa | Natural outlet pressure |

All implemented fluid actions and boundaries on this page target face regions. `mass_flow` and
outlet pressure are scalar; inlet velocity and wall motion are directed vectors, either supplied as
a vector signal or as a scalar signal plus normalized `global([x, y, z])` direction.

```fea
load flow_rate {
    mass_flow on geometry.channel.region.inlet = 0.2 kg/s
}

load outlet {
    fluid.outlet_pressure {
        target = geometry.channel.region.outlet
        value = 0 Pa
    }
}
```

## How do I apply a time-varying inlet velocity?

```fea
signal startup_profile : velocity over time {
    samples = [
        { at = 0 s, value = 0 m/s },
        { at = 0.5 s, value = 1.5 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation inlet_ramp {
    inlet_velocity {
        target = geometry.channel.region.inlet
        signal = signal.startup_profile
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}
```

This separates scalar speed history from direction and target. Use `fluid.velocity` constraint for a
constant prescribed vector. A flat inlet profile and a developed profile can yield different
pressure drop and entrance behavior; the current contract must not imply a profile it did not author.

## How do I prescribe a steady uniform inlet velocity?

Use a constant `inlet_velocity` action when the complete global vector is known:

```fea
load steady_inlet {
    inlet_velocity on geometry.channel.region.inlet = [2, 0, 0] m/s
}
```

The current `fluid.velocity` essential-constraint contract carries a scalar `Velocity`; it does not
encode a complete three-component direction. Use `inlet_velocity` for the directed inlet vector or
a signal-driven magnitude/direction. Do not apply both declarations to the same target.

## How do I apply a parabolic or fully developed inlet profile?

A scalar `inlet_velocity` excitation does not silently invent a parabolic profile. Define the
spatial profile through a topology field only after that field contract is implemented. The current
`.fea` signal function is sampled over time, frequency, or a study parameter; it is not called with
boundary coordinates. Therefore a generated `.m` signal cannot currently author a parabolic inlet
profile. Extend the formulation with a typed developed-profile or topology-field declaration.
For a circular laminar pipe, the analytic profile is
$u(r)=2\bar u[1-(r/R)^2]$; its mean, maximum, radius, axis, and coordinate frame must all remain
explicit.

## How do I specify turbulence intensity and length scale at an inlet?

These are transported-model boundary data, not velocity units. The current closed load set does not
expose a turbulence-inlet object. A turbulence formulation must add typed intensity/kinetic-energy,
length-scale or dissipation data with admissible ranges. Do not hide turbulence quantities in a
velocity signal or rely on undocumented solver defaults.

## How do I specify total mass flow?

```fea
load commanded_flow {
    mass_flow on geometry.channel.region.inlet = 0.2 kg/s
}
```

Mass flow preserves $\dot m$, not velocity. A realization must use density and resolved area to
construct a compatible boundary field and report the achieved resultant. Check sign/orientation and
do not separately prescribe an incompatible complete velocity field on the same inlet.

## How do I split a known mass flow across several inlets?

Create one `mass_flow` load per inlet with an explicit signed share, then include all of them in the
same load case. The shares should sum to the intended total. Do not apply the full total to every
inlet and expect RunMat to divide it by area automatically.

```fea
load branch_a_flow {
    mass_flow on geometry.manifold.region.inlet_a = 0.12 kg/s
}

load branch_b_flow {
    mass_flow on geometry.manifold.region.inlet_b = 0.08 kg/s
}
```

For a solved flow split, prescribe the physically known upstream/downstream conditions instead and
let the network resistance determine each branch rate.

## How do I drive a moving fluid wall?

```fea
signal belt_speed : velocity over time {
    samples = [
        { at = 0 s, value = 0 m/s },
        { at = 1 s, value = 2 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation moving_belt {
    wall_motion {
        target = geometry.channel.region.belt
        signal = signal.belt_speed
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}
```

This is prescribed kinematics. Use `fluid_structure.no_slip` when wall motion must come from and feed
back into a structural solution.

## How do I set up a Couette-flow moving-wall check?

Use a channel of height $H$, a stationary no-slip lower wall, and a constant upper-wall velocity.
The direct `wall_motion` action carries the global vector:

```fea
constraint stationary_wall {
    fluid.no_slip_wall {
        target = geometry.channel.region.lower_wall
    }
}

load moving_wall {
    wall_motion on geometry.channel.region.upper_wall = [1, 0, 0] m/s
}

load_case couette {
    include load.moving_wall
}
```

With no pressure gradient and a Newtonian fluid, the developed solution is
$u_x(y)=Uy/H$ and the wall shear is $\tau=\mu U/H$. Compare the velocity profile, wall shear, and
mass balance after refining cells across the gap. This separates moving-wall semantics from inlet
or pressure-boundary behavior.

## How do I model a rotating wall, impeller, or cylinder?

A single `wall_motion` direction describes translational velocity, not rigid rotation over a
surface. Rotating-wall kinematics require axis, origin, angular speed and the spatial law
$\mathbf u=\boldsymbol\omega\times(\mathbf x-\mathbf x_0)$. The current source language does not
expose a coordinate-dependent wall field or rotating-domain load. Do not assign one tangential
vector to an entire curved rotor.

## How do I model suction or blowing through a wall?

Treat the porous patch as an inlet/outlet boundary with a signed normal velocity or conserved mass
flow. Confirm normal orientation and exclude it from a no-slip wall region. A porous-medium pressure
drop is a separate constitutive domain law and cannot be inferred from boundary suction alone.

## How do I set outlet pressure?

```fea
load atmospheric_outlet {
    fluid.outlet_pressure {
        target = geometry.channel.region.outlet
        value = 0 Pa
    }
}
```

The value follows the study's pressure convention (typically gauge when a zero reference is used).
Outlet pressure is a natural boundary; `fluid.pressure_reference` fixes the gauge. Keep the two roles
distinct and avoid conflicting constraints on recirculating/inflow portions of an outlet.

## How do I set absolute pressure versus gauge pressure?

State the convention in study inputs and material/state definitions. A value of `0 Pa` commonly
means zero gauge pressure; it is not zero absolute pressure. Compressible density and thermodynamic
properties usually require absolute pressure even when the momentum unknown is stored relative to a
gauge. Do not mix conventions across inlet, outlet, and material models.

## How do I model several pressure outlets?

Declare one outlet-pressure load per region. Equal values permit the solution to determine flow
split; different values impose the intended pressure difference. Inspect signed flux at every outlet
because one may become an inlet. Do not add a redundant pressure-reference constraint to every
pressure outlet unless the formulation explicitly requires it.

## Should I prescribe inlet velocity or mass flow?

Use `inlet_velocity` when the velocity profile or speed history is known. Use `mass_flow` when the
integrated mass rate is known and the formulation is responsible for constructing a compatible
profile. For incompressible flow with constant density, volume flow is $\dot V=\dot m/\rho$; do not
enter volume-per-time units into `mass_flow`.

## How do I model a pump, fan, or pressure rise?

The current closed load set provides inlet velocity, mass flow, wall motion, and outlet pressure. It
does not yet define a first-class pump/fan curve or internal pressure-jump object. Do not hide a
pump curve in an unrelated outlet pressure. A complete contract needs upstream/downstream regions,
flow-vs-pressure data, rotation/control state where applicable, interpolation/extrapolation policy,
and power-balance evidence.

## How do I model backflow at a pressure outlet?

An outlet-pressure boundary supplies traction, not a one-way valve. If the solved field reverses,
the admitted formulation must define its backflow stabilization and transported-state values.
Inspect signed flux on each outlet; do not clamp reverse flow by also prescribing velocity on the
same face.

## How do I apply a volumetric flow rate?

The current typed resultant is mass flow. Convert a known volumetric rate only with a declared
density valid for the modeled state: $\dot m=\rho\dot V$. For compressible or variable-density flow,
a single conversion may be invalid; a dedicated volumetric-flow contract would need reference-state
semantics.

## How do I model buoyancy or natural convection?

Buoyancy requires gravity plus a density/temperature relation in a coupled thermal-fluid
formulation. It is not represented by an inlet velocity. Use a conjugate/natural-convection model
that admits the Boussinesq or compressible body-force law and declares its reference temperature,
thermal expansion coefficient, and gravity vector.

## How do I model wind loading on a structure?

Solve the fluid traction and transfer it through an admitted fluid-structure interface when the
flow field matters. A guessed uniform structural pressure is a separate engineering approximation,
documented under [structural pressure loads](./structural#how-do-i-apply-pressure-to-a-structural-surface).
Do not label a pressure-only structural model as CFD or two-way FSI.

## How do I model a periodic pressure drop through a channel cell?

Use `fluid.periodic` for the mapped velocity/pressure fluctuation and a separate formulation input
for the macroscopic pressure gradient or conserved flow. The current load set has no first-class
periodic pressure-jump declaration. Do not place unrelated outlet-pressure boundaries on faces that
are also periodic.

## Why is the inlet flow direction reversed?

Check the geometry face normal, the declared global direction, and the sign of mass flow. A positive
mass-flow convention can be relative to the resolved outward normal, while a velocity vector is in
its declared frame. Preview the annotation on the exact CAD face and verify signed integrated flux
before interpreting downstream results.

## Why does a pressure outlet become an inlet?

Pressure is not a one-way constraint. Recirculation or an adverse pressure field can produce
backflow. Inspect transported quantities and stabilization supplied for backflow, move the boundary
farther from the recirculation region if physically appropriate, or model the connected downstream
domain. Do not add a contradictory velocity clamp merely to force the desired sign.

## What should I inspect after applying fluid loads?

Inspect resolved face area and orientation, integrated mass flux, imbalance across all boundaries,
wall-relative velocity, pressure convention, outlet backflow, and momentum/energy residuals. For
transient signals, verify that every solver time lies inside the authored interpolation/extrapolation
contract.

An inlet-velocity excitation preserves a velocity field; mass flow preserves a resultant. Wall
motion is distinct from a structural displacement unless an FSI interface transfers the motion.
Avoid combining incompatible prescribed velocity and pressure data on one boundary.

## How do I convert volumetric flow rate to a RunMat inlet condition?

The closed inlet resultant is `mass_flow`, so convert volumetric rate using density consistent with
the fluid model: $\dot m=\rho Q$. For a uniform velocity approximation on admitted area $A$,
$\bar u=Q/A$. Choose one representation; do not prescribe both on the same inlet. Compressible or
strongly variable-density flow requires the formulation to own the state-dependent conversion
rather than using one nominal density silently.

## How do I model a fan curve or pump curve?

A fan or pump curve relates pressure rise to flow and is not a fixed outlet pressure. The current
load set does not expose a first-class pressure-jump-versus-flow device. Use a resolved rotating-
machine domain or an admitted typed reduced component when available. For one operating-point
approximation, state the derived pressure/flow condition and its limited validity; do not present it
as a complete fan curve.

## How do I model an opening where flow may enter or leave?

Use an outlet or open-boundary formulation that owns backflow state and transported quantities. A
prescribed velocity fixes direction and cannot represent bidirectional exchange; a pressure outlet
may admit backflow but needs compatible temperature, species, and turbulence data from the
formulation. Place the boundary where the reduced opening law is physically appropriate and inspect
signed flux.

## How do I verify inlet and outlet mass conservation?

Integrate signed mass flux over every inlet, outlet, porous transfer, and moving boundary. For a
steady closed control volume, the algebraic sum must close within tolerance. For transient or
compressible work, include storage. Compare each boundary's resolved area, normal, and resultant
after remeshing; a plausible velocity field does not compensate for a nonclosing mass balance.

## How do I choose a well-posed inlet and outlet combination?

For a single-path incompressible domain, a common setup is one known inlet velocity or mass flow,
wall conditions on every impermeable face, one outlet pressure, and one pressure datum if the
formulation requires it. Do not prescribe both complete velocity and pressure on the same open face.
For several outlets, pressure conditions let the resistance determine flow split; fixed outlet flows
instead prescribe that split.

Before solving, classify every boundary face as inlet, outlet/opening, wall, symmetry, periodic, or
coupled interface. Unclassified faces must not acquire a solver default silently. After solving,
report signed mass flux for each named region and the pressure-reference owner.

## How do I model pipe or duct flow from a known flow rate?

Use `mass_flow` when mass rate is known. For a constant-density fluid, convert a specified volume
rate with $\dot m=\rho\dot V$. If using a uniform velocity approximation, compute
$\bar u=\dot V/A$ from the resolved inlet area and record the assumption.

```fea
load duct_flow {
    mass_flow on geometry.duct.region.inlet = 0.36 kg/s
}

load duct_outlet {
    fluid.outlet_pressure {
        target = geometry.duct.region.outlet
        value = 0 Pa
    }
}
```

Compare the solved pressure drop with an analytic or correlation-based estimate in its valid
Reynolds-number and roughness range. Entrance length, bends, and developing profiles can make a
fully developed reference inapplicable.

## How do I model external aerodynamics around a vehicle or component?

Use a sufficiently remote outer fluid domain, prescribe freestream velocity at the inlet/far-field
portion, pressure/open behavior downstream, and no-slip or moving-wall behavior on the body. Side
and top boundaries may use symmetry only when that approximation is physically intended. The
current boundary set does not turn a small enclosure into a far field automatically.

Report blockage ratio, boundary distances, drag/lift from integrated traction, wake development,
mass balance, and sensitivity to domain extent. To transfer aerodynamic loading into structural
FEA, use the governed fluid-to-structure state edge and preserve total force and moment.

## How do I model a manifold with several inlets and outlets?

Name each port separately. Apply only the quantities known from the surrounding system: prescribed
flows at controlled branches, pressures at reservoir-connected branches, or a network/device law
when neither is fixed. Signed flows must close across all ports, including any porous or moving
boundary contribution.

Avoid one broad `ports` region for different conditions. Separate identities make reversed flow,
incorrect normals, and unintended flow split visible in the evidence rather than hidden in a single
net resultant.

## How do I verify a laminar pipe-flow load case?

Use a straight circular pipe and compare with Hagen–Poiseuille flow in its valid regime:

$$
\Delta p=\frac{128\mu L\dot V}{\pi D^4},\qquad
u(r)=2\bar u\left(1-\frac{r^2}{R^2}\right).
$$

Prescribe either the admitted mass flow or inlet velocity and one outlet pressure; do not prescribe
both flow resultants. Confirm Reynolds number, entrance length, signed mass balance, pressure drop,
and wall shear. Refine the mesh radially and axially and move the measurement sections away from the
inlet. If a uniform inlet is used, compare only after the profile has developed.

## What does RunMat preserve when the fluid mesh changes?

`mass_flow` preserves the signed integrated kilograms per second. `inlet_velocity` and
`wall_motion` preserve the authored velocity field, so their integrated flow changes with resolved
area and density. `fluid.outlet_pressure` preserves pressure intensity, not outlet flow. After every
geometry or mesh revision, compare region identity, area, normal orientation, each port resultant,
and the global continuity residual before comparing pressure or force outputs.

Further reading: [fluid constraints](../constraints/fluid), [fluid studies](../physics/fluid),
[conjugate heat transfer](../physics/conjugate-heat-transfer), and
[fluid–structure interaction](../physics/fluid-structure-interaction).
