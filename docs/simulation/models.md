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

A model that declares no `load_case` applies all of its loads together in one default case,
`load_case.default`, so a minimal model needs no case line. Once a model declares any
`load_case`, every load must be included in a declared case explicitly; a load left out is
rejected rather than silently dropped or silently applied.

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

### Homogeneous shell sections

```fea
model enclosure {
    section skin_2mm {
        homogeneous_shell {
            thickness = 2 mm
            reference_surface = mid_surface
            shear_correction = 0.8333333333333333
        }
    }

    orientation skin_axes {
        cartesian {
            origin = [0, 0, 0] m
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }

    assign material.aluminum to geometry.part.region.skin
    assign section.skin_2mm to geometry.part.region.skin
    assign orientation.skin_axes to geometry.part.region.skin
}
```

`reference_surface` is `mid_surface`, `negative_normal_surface`, `positive_normal_surface`, or `offset_from_mid_surface`. The offset form also requires `reference_offset = <length>`; a positive offset follows the oriented face normal. Surface choice is physical: it changes membrane–bending coupling and rotary inertia because shell translations and rotations live on that surface. It is not a display offset.

RunMat integrates the linear section law analytically through the thickness. There is no integration-point count to tune. The MITC3 formulation also owns its drilling stabilization; drilling stiffness is not a material or section property.

### Orthotropic lamina materials

Each composite ply references a material with either isotropic elasticity or the complete orthotropic lamina law:

```fea
material carbon_epoxy {
    name = "Unidirectional carbon/epoxy"
    mass_density = 1600 kg/m^3

    orthotropic_lamina {
        longitudinal_modulus = 135 GPa
        transverse_modulus = 10 GPa
        major_poisson_ratio = 0.30
        in_plane_shear_modulus = 5 GPa
        transverse_shear_modulus_13 = 5 GPa
        transverse_shear_modulus_23 = 3.8 GPa
    }

    rayleigh_damping {
        mass_coefficient = 0.2 Hz
        stiffness_coefficient = 20 us
    }
}
```

Axes 1 and 2 are the ply material axes. The `13` and `23` moduli are transverse-shear properties for first-order shear-deformation shells; this block is not a three-dimensional orthotropic-solid law. Positive moduli and positive-definite reciprocal Poisson compliance are checked before property projection. Density is required because every assembled structural operator carries an explicit mass operator, even when a particular static procedure will not use it.

### Composite layups

```fea
model composite_panel {
    orientation laminate_axes {
        cartesian {
            origin = [0, 0, 0] m
            x_axis = [1, 0, 0]
            y_axis = [0, 1, 0]
        }
    }

    layup quasi_isotropic_skin {
        reference_orientation = orientation.laminate_axes
        ply_order = along_surface_normal
        reference_surface = mid_surface

        ply bottom_0 {
            material = material.carbon_epoxy
            thickness = 0.125 mm
            angle = 0 deg
        }
        ply minus_45 {
            material = material.carbon_epoxy
            thickness = 0.125 mm
            angle = -45 deg
        }
        ply plus_45 {
            material = material.carbon_epoxy
            thickness = 0.125 mm
            angle = 45 deg
        }
        ply top_90 {
            material = material.carbon_epoxy
            thickness = 0.125 mm
            angle = 90 deg
        }
    }

    assign layup.quasi_isotropic_skin to geometry.part.region.skin
    assign orientation.laminate_axes to geometry.part.region.skin
}
```

The first declared ply is physically first. With `along_surface_normal`, subsequent plies advance in the oriented face-normal direction; `opposite_surface_normal` reverses that traversal. Ply angle is measured from the assigned orientation's projected x-axis about the shell normal.

`reference_surface` is `mid_surface`, `first_ply_surface`, `last_ply_surface`, or `offset_from_mid_surface`. The offset form requires `reference_offset = <length>`. `shear_correction = <dimensionless>` is optional; when omitted, MITC3 uses its documented formulation value of (5/6).

A laminate is the constitutive owner of its shell region. Do not also assign a homogeneous material to that face. Every ply material and the layup itself are preserved in the compiled property identity and audit record; missing ply materials, missing elasticity or density, a competing material assignment, an unbound orientation, or partial face coverage fails before element assembly.

### Laminate equations used by RunMat

For ply (k), RunMat rotates its plane-stress matrix into shell axes as \(\bar{Q}_k\), locates its lower and upper surfaces at \(z_{k-1}\) and \(z_k\) relative to the selected reference surface, and forms

\[
A = \sum_k \bar{Q}_k (z_k-z_{k-1}), \qquad
B = \frac{1}{2}\sum_k \bar{Q}_k (z_k^2-z_{k-1}^2), \qquad
D = \frac{1}{3}\sum_k \bar{Q}_k (z_k^3-z_{k-1}^3).
\]

The generalized section response is

\[
\begin{bmatrix}N\\M\end{bmatrix}
=
\begin{bmatrix}A&B\\B&D\end{bmatrix}
\begin{bmatrix}\varepsilon_0\\\kappa\end{bmatrix},
\qquad
Q_s = A_s\gamma_s,
\]

where \(A_s\) is the thickness-integrated, angle-transformed `13`/`23` shear law including the selected shear correction. Unsymmetric stacking or a non-mid reference surface therefore retains nonzero \(B\); RunMat does not silently recenter the laminate.

Shell inertia uses all three thickness moments about the same reference surface:

\[
I_0 = \sum_k \rho_k(z_k-z_{k-1}), \qquad
I_1 = \frac{1}{2}\sum_k \rho_k(z_k^2-z_{k-1}^2), \qquad
I_2 = \frac{1}{3}\sum_k \rho_k(z_k^3-z_{k-1}^3).
\]

The sparse mass operator retains the translation–rotation coupling from \(I_1\). It therefore remains physically consistent for density-unsymmetric layups and reference surfaces away from the mass centroid; it is not reduced to diagonal mass plus rotary inertia. The two director rotations use \(I_2\). The formulation gives its stabilized drilling rotation the same positive nodal rotary term so that the numerical drilling degree of freedom is not massless. Rayleigh mass and stiffness terms are integrated per ply with the same moments, so different ply damping coefficients remain distinct rather than being averaged. Recovery evaluates each ply's top and bottom plane-stress state, publishes membrane force, bending moment, and transverse-shear resultants, and reports the maximum recovered ply-surface von Mises value for the existing shell scalar field. The compiled recovery record retains every ply layer so additional ply-specific criteria can be added without reconstructing the layup from source text.

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

`runmat check` validates authored types, units, references, topology roles, composition, procedure compatibility, and required capabilities. Exact execution adds geometry realization, region binding, mesh projection, complete property coverage, solver well-posedness checks, and numerical evidence. A valid source file is necessary but does not by itself validate a formulation for a given use.
