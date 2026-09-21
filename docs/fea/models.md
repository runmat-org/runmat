# Models

A model owns physical data attached to typed geometry: materials, sections, layups, orientations, essential constraints, natural boundaries, applied actions, signals, initial conditions, interfaces, and load cases. A model does not choose a solver procedure or execution location; a study does that.

## Material Is Separate From Assignment

```fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.30
    }
}

model bracket {
    assign material.steel to geometry.part.region.body
}
```

Defining `steel` does not apply it everywhere. The assignment names the exact geometry region that owns the constitutive law. For a tetrahedral volume model that target is a solid or cell region—not one of its boundary faces. Edge and face material ownership is used for beam and shell formulations together with the required section, layup, and orientation properties.

Before assembly, RunMat projects every assignment through persistent geometry-to-mesh lineage into one immutable element property map. It rejects gaps, overlaps, wrong topology, incompatible capabilities, stale geometry/mesh revisions, and competing homogeneous/composite ownership. Solver implementations consume that compiled map and cannot choose a fallback material independently.

## Material Capabilities

Material declarations may provide the capabilities required by the selected physics:

- mass density;
- linear elasticity;
- Rayleigh damping;
- thermal reference data, conductivity, heat capacity, expansion, and temperature dependence;
- acoustic density/sound-speed damping data;
- electrical and magnetic properties;
- typed plasticity parameters.

The checker validates physical dimensions and admissible ranges. A study cannot satisfy a mechanical formulation with only thermal conductivity, or invent a missing density for dynamics.

AP242 material/property candidates can be adopted explicitly from imported geometry. The declaration identifies each property to use, its evidence environment, and conflict policy; source confidence remains provenance and never scales a coefficient.

## Constraints And Natural Boundaries

```fea
model supported {
    constraint fixed_support {
        structural.fixed { target = geometry.part.region.support }
    }
    load pressure {
        structural.pressure {
            target = geometry.part.region.pressure_face
            value = 250 kPa
        }
    }
}
```

Constraint and boundary kinds are closed, family-specific typed variants. Structural fixed/symmetry/antisymmetry, prescribed structural motion, thermal temperature/tie/flux/convection, electrical potential/current/reference, magnetic vector-potential/gauge, acoustic pressure/impedance/radiation, and fluid wall/inlet/outlet/periodic conditions retain their exact target topology and physical units.

Relational constraints—tie, periodic, rigid/distributing coupling, equations, and phase-shifted periodic forms—name both sides and the constrained degrees of freedom explicitly. Unsupported or contradictory combinations fail during checking rather than becoming solver conventions.

## Loads, Signals, And Cases

A constant applied action can use concise syntax:

```fea
load downward {
    force on geometry.part.region.pad = [0, -1, 0] kN
}
load_case service { include load.downward }
```

Time-, frequency-, or parameter-dependent action uses a quantity-typed signal:

```fea
signal pulse : force over time {
    samples = [
        { at = 0 s, value = 0 N },
        { at = 0.01 s, value = 1 kN },
        { at = 0.02 s, value = 0 N },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation impact {
    force {
        target = geometry.part.region.pad
        signal = signal.pulse
        direction = global([0, -1, 0])
    }
}
load_case impact { include excitation.impact }
```

Signals own physical quantity, domain, interpolation, extrapolation, discontinuity, and frame rules. Standard time waveforms include step, ramp, sine, half-sine, triangular, and rectangular pulses. Pure imported `.m` functions may generate time histories or spectral data through the bounded materialization contract described in [RunMat function integration](./language/runmat-function-integration).

## Sections, Layups, And Orientations

Beam models use explicit general, rectangular, circular, or tubular sections. Shell models use a homogeneous section or an ordered laminate layup with typed plies. Orientations may be global Cartesian, explicit Cartesian, cylindrical, or spherical.

These are separate definitions and assignments. A material assignment does not imply a section, thickness, laminate stacking direction, or material frame. Duplicate owners, degenerate frames, empty layups, invalid section geometry, and unsupported offsets fail before assembly.

## Initial Conditions And Spectral Environments

Models can declare typed structural displacement/velocity/angular-velocity, thermal temperature, electrical potential, magnetic vector-potential, acoustic pressure, and fluid pressure/velocity initial state. Sources are uniform values, signal samples, or admitted field artifacts; the consuming procedure names the state explicitly.

Response-spectrum and random-excitation declarations are also model data. They own axes, ordinate units, damping or quality factor, response measure, directions/channels, modal/directional/truncation policy, spectral convention, interpolation, extrapolation, and evidence. A modal response analysis names the exact model declaration and retained modal basis it consumes.

## Reusable Models

Models can take strongly typed region and material parameters:

```fea
model reusable(region body : region<solid>) {
    input body_material : material
    assign input.body_material to input.body
}
```

A study binds every required parameter when selecting the model. `include model.base` composes reusable declarations while preserving defining-module identity. Cycles, missing bindings, wrong region roles, duplicate ids, and conflicting assignments fail; source order never acts as an override rule.

## Validation Boundary

`runmat fea check` validates authored types, units, references, topology roles, composition, procedure compatibility, and required capabilities. Exact execution adds geometry realization, region binding, mesh projection, complete property coverage, solver well-posedness checks, and numerical evidence. A valid source file is necessary but does not by itself qualify a formulation for production use.
