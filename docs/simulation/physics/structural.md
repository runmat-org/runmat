# Structural

Use structural FEA to answer questions about deformation, stress, reactions, natural frequencies,
forced vibration, shock response, buckling, and nonlinear load paths. RunMat keeps the physical
model independent from meshing and execution: geometry defines regions, the mesh discretizes them,
the model assigns materials, sections, supports, and loads, and the study chooses a procedure graph.

The typed `.fea` language covers the complete structural procedure set described on this page, and
the native `.fea` execution path runs it. Each result records the actual formulation, operator
identities, backend, convergence, residual, and recovery evidence.

## What equations does RunMat solve for structural FEA?

For small-strain linear elasticity, RunMat uses displacement as the primary unknown:

$$
\boldsymbol{\varepsilon}(\mathbf{u}) = \tfrac12\left(\nabla\mathbf{u} +
\nabla\mathbf{u}^{\mathsf T}\right), \qquad
\boldsymbol{\sigma}=\mathsf{C}:\boldsymbol{\varepsilon},
$$

$$
\rho\ddot{\mathbf{u}}-\nabla\!\cdot\boldsymbol{\sigma}=\mathbf{b}
\quad\text{in }\Omega,
$$

with essential displacement constraints on $\Gamma_u$ and traction
$\boldsymbol{\sigma}\mathbf{n}=\bar{\mathbf{t}}$ on $\Gamma_t$. Tetrahedral finite elements and
the declared constitutive capabilities produce the canonical semidiscrete system

$$
\mathsf{M}\ddot{\mathbf{u}} + \mathsf{C}\dot{\mathbf{u}} +
\mathsf{K}\mathbf{u}=\mathbf{f}(t).
$$

The selected procedure determines which terms and states are active:

| `.fea` procedure | Discrete problem |
| --- | --- |
| `structural static` | $\mathsf K\mathbf u=\mathbf f$ |
| `structural preload` | Static equilibrium plus retained prestress/geometric state |
| `structural modal` | $\mathsf K\boldsymbol\phi_i=\omega_i^2\mathsf M\boldsymbol\phi_i$ |
| `structural transient` | Direct integration of $\mathsf M\ddot{\mathbf u}+\mathsf C\dot{\mathbf u}+\mathsf K\mathbf u=\mathbf f(t)$ |
| `structural modal_transient` | The same dynamics projected into a retained modal basis |
| `structural harmonic` | $(\mathsf K+i\omega\mathsf C-\omega^2\mathsf M)\hat{\mathbf u}=\hat{\mathbf f}$ |
| `structural modal_harmonic` | Harmonic dynamic stiffness projected into modal coordinates |
| `structural prestressed_modal` | $(\mathsf K+\mathsf K_g)\boldsymbol\phi_i=\omega_i^2\mathsf M\boldsymbol\phi_i$ |
| `structural buckling` | Generalized eigenproblem using elastic and prestress-derived geometric stiffness |
| `structural prestressed_harmonic` | Harmonic response about an explicit prestressed state |
| `structural modal_response_spectrum` | Modal peak response followed by the declared modal combination rule |
| `structural random_response` | Modal transfer functions applied to input CPSD/PSD and integrated statistics |
| `structural nonlinear` | Incremental equilibrium $\mathbf R(\mathbf u,\lambda)=\mathbf f_{int}-\lambda\mathbf f_{ext}=\mathbf0$ |

RunMat does not silently add weak springs, mass, damping, a reference constraint, or a load case to
make these systems solvable.

## How do I run a linear static stress analysis?

This complete model assigns steel to both solids in a STEP assembly, fixes one face on each
disconnected occurrence, applies a resultant force to the left part, and requests displacement and
von Mises stress:

The included STEP file carries the authored body and end-face names the study selects; `select name` resolves only names authored in the CAD file. The study stays in one readable source file; exporting the project preserves its CAD names.

```fea
geometry bracket {
    file = "./geometry/two-box-assembly.step"
    units = m

    region left_body : solid { select name "left_body"; expect exactly_one }
    region right_body : solid { select name "right_body"; expect exactly_one }
    region left_support : face { select name "left_support"; expect exactly_one }
    region left_load_face : face { select name "left_load_face"; expect exactly_one }
    region right_support : face { select name "right_support"; expect exactly_one }
}

mesh bracket on geometry.bracket {
    tetrahedra {
        order = linear
        target_size = 1.5 m
        maximum_deviation = 0.05 m
    }
}

material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

model loaded_bracket {
    assign material.steel to geometry.bracket.region.left_body
    assign material.steel to geometry.bracket.region.right_body
    constraint left_support {
        structural.fixed { target = geometry.bracket.region.left_support }
    }
    constraint right_support {
        structural.fixed { target = geometry.bracket.region.right_support }
    }
    load service_traction {
        traction on geometry.bracket.region.left_load_face = [166.6666666667, 0, 0] Pa
    }
    load_case service { include load.service_traction }
}

study bracket_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
    output stress = analysis.static.field.element.von_mises_stress
}
```

The traction tuple is a global XYZ intensity. In this fixture the selected end face is $6\,\mathrm{m^2}$,
so `[166.6666666667, 0, 0] Pa` integrates to a $1\,\mathrm{kN}$ resultant in +X while boundary
quadrature distributes it consistently over the face mesh. A face target alone does not rotate an
arbitrary traction vector onto that face normal.

```sh
runmat check bracket.fea
runmat plan bracket.fea
runmat run bracket.fea
```

For isotropic linear elasticity, $\mathsf C=\mathsf C(E,\nu)$ is assembled from `youngs_modulus`
and `poisson_ratio`. Dynamics also requires `mass_density`. A face never owns the solid material
behind it: material coverage must be assigned to the participating solid or cell regions.

## Why is my structural stiffness matrix singular?

A singular or non-coercive stiffness operator usually means one of the following:

- rigid-body translation or rotation remains unconstrained;
- a part or included model is disconnected;
- a tie, connector, or contact does not cover the intended regions;
- a beam/shell region lacks its required section or orientation;
- a material/property assignment misses participating elements;
- constraints conflict or collapse the admissible subspace.

RunMat reports the typed target and source span for constraints and preserves constrained-space
operator evidence. Fix the physical model. Do not add arbitrary supports merely to make
factorization succeed; that changes reactions, modes, and load paths.

## How do I run a modal analysis and check natural frequencies?

```fea
study bracket_modes {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket

    analysis modes {
        structural modal { modes = 30 }
        retain basis
    }
}
```

The eigensolve operates on the constrained $\mathsf K$ and $\mathsf M$. Each mode records angular
frequency $\omega_i$, frequency $f_i=\omega_i/(2\pi)$, the mode shape, modal mass/stiffness,
residual, and normalization identity. Check that:

- the first modes are physically plausible and not unconstrained rigid modes;
- cumulative participating mass is sufficient in every excited direction;
- repeated or closely spaced modes are treated as a subspace rather than compared vector-by-vector;
- frequency and mode shapes converge under mesh refinement;
- the basis was produced by the exact operator and mesh used downstream.

## How do I run transient structural dynamics with a pulse or time history?

Direct transient dynamics uses the canonical mass, damping, and stiffness operators and advances the
physical state over the declared time samples. RunMat's direct structural path uses Newmark-family
effective equilibrium; accepted steps retain balance and reaction evidence.

```fea
signal impact_history : force over time {
    samples = [
        { at = 0 ms, value = 0 N },
        { at = 2 ms, value = 1 kN },
        { at = 8 ms, value = 0 N },
    ]
    interpolation = linear
    extrapolation = reject
}

model impacted_bracket {
    assign material.steel to geometry.bracket.region.body
    constraint support {
        structural.fixed { target = geometry.bracket.region.fixed_face }
    }
    excitation impact {
        force {
            target = geometry.bracket.region.loaded_face
            signal = signal.impact_history
            direction = global([0, -1, 0])
            kinematics = fixed_frame
        }
    }
}

study transient_response {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.impacted_bracket
    analysis response {
        structural transient {
            start_time = 0 s
            end_time = 50 ms
            time_step = 0.1 ms
        }
    }
}
```

The time step must resolve load breakpoints and the highest response frequencies of interest.
Initial displacement, velocity, and acceleration are typed initial conditions; they are not inferred
from the first signal sample. Restart state must match the operator and topology identity of the
continuing analysis.

## How do I run harmonic or frequency-response analysis?

Direct harmonic response solves a complex dynamic-stiffness system at each declared frequency.
Modal harmonic response first projects the operators and forcing through a retained basis:

$$
\mathsf K_r=\boldsymbol\Phi^{\mathsf T}\mathsf K\boldsymbol\Phi,\quad
\mathsf M_r=\boldsymbol\Phi^{\mathsf T}\mathsf M\boldsymbol\Phi,\quad
\mathbf f_r=\boldsymbol\Phi^{\mathsf T}\mathbf f.
$$

```fea
analysis modes {
    structural modal { modes = 40 }
    retain basis
}

analysis response after analysis.modes {
    structural modal_harmonic {
        basis = analysis.modes.state.basis
        frequencies = [20, 50, 100, 200] Hz
    }
}
```

Use direct response when modal truncation is unacceptable; use modal response when the retained basis
captures the excited band and participation. Check damping, frequency spacing around resonances,
complex equilibrium residual, and modal truncation. A frequency list is part of study identity.
A modal-reduced sweep is accepted on its reduced-space residual and publishes its full-order
imbalance as the modal truncation measure; `static_correction = applied_loads` and modal
`residual_vectors` reduce it (see [modal truncation and residual vectors](../language/reference/procedures.md#modal-truncation-and-residual-vectors)).

The [solid harmonic qualification case](../qualification/solid-harmonic) checks physical
Tet4/Tet10 direct and modal response against independent element integrals and oscillator
amplitude, phase, and power references. Its one-DOF scope and remaining promotion gates are
documented with the runnable test.

## How do preload, prestressed modes, and buckling connect?

A preload step retains a prestress state. Prestressed modal, buckling, or harmonic steps must name
that exact state and declare their dependency:

```fea
analysis preload {
    structural preload {}
}

analysis buckling after analysis.preload {
    structural buckling {
        prestress = analysis.preload.state.geometric_stiffness
        modes = 10
    }
}
```

A preload publishes several states, so it has no single state to rename with `retain`; the
consumer names the one it needs, here `geometric_stiffness`.

The geometric stiffness $\mathsf K_g$ is derived from the retained equilibrium state. RunMat rejects
topology/operator mismatches rather than applying prestress from a different model. Linear buckling
factors are bifurcation indicators for the linearized problem; they do not include imperfections,
post-buckling response, plasticity, or contact changes unless a separately qualified nonlinear
formulation explicitly models them.

## How do I run response-spectrum or random-vibration analysis?

Response-spectrum analysis evaluates each retained mode against a named spectrum, then combines
modal peaks using the declared combination method. It does not reconstruct a unique time history.
Closely spaced modes require a correlation-aware rule such as CQC rather than unconditional SRSS.

Random response operates on power or cross-spectral density. In modal coordinates,

$$
\mathsf S_{qq}(\omega)=\mathsf H(\omega)\,
\mathsf S_{ff}(\omega)\,\mathsf H(\omega)^{\!*},
$$

and variance follows from integration over angular frequency. Input PSD/CPSD units therefore carry
the inverse-domain unit and cannot be substituted by an amplitude spectrum.

The checked runnable examples
[`modal-response-spectrum.fea`](../examples/modal-response-spectrum.fea)
and [`modal-random-response.fea`](../examples/modal-random-response.fea)
show the complete spectrum declarations, modal state flow, and outputs.

## How do I run nonlinear structural analysis?

`structural nonlinear` declares an explicit load-factor domain. The nonlinear solver advances bounded
increments, forms the current residual and tangent, iterates to the requested equilibrium target,
and records accepted or rejected increment evidence. Contact and follower loading change the
residual/tangent and must be declared as such.

```fea
analysis load_path {
    structural nonlinear {
        start_load_factor = 0
        end_load_factor = 1
    }
}
```

Convergence of Newton iterations is not validation of the constitutive model. Check increment size,
residual and energy norms, contact penetration, load/reaction balance, path dependence, and whether
the implemented material/contact formulation is qualified for the intended use.

## Structural constraints, loads, materials, and outputs

- [Structural constraints](../constraints/structural) documents every support, prescribed-motion,
  coupling, tie, periodic, and equation kind.
- [Structural loads](../loads/structural) documents force, moment, traction, pressure, body force,
  signals, directions, distribution, follower kinematics, and load cases.
- [Models](../models) covers material, beam/shell/solid property ownership and composition.
- [Outputs](../language/outputs-fields-and-observations) covers typed fields, projections, spatial
  reductions, probes, histories, sampling, retention, and partial access.

## How do I inspect Tet10 stress without hiding its within-element variation?

Quadratic tetrahedra recover strain and stress at the same four physical integration points used
by stiffness assembly. Each point is a first-class result entity with a stable identity and physical
position. Element strain, stress, von Mises stress, and strain-energy density remain available as
Jacobian-weighted projections for summaries and element plots; use the integration-point field when
within-element variation or recovery error is the engineering question.

Retain both views in the study when you need to compare them:

```fea
study bracket_check {
    analyze static using model.loaded_bracket

    output stress_at_quadrature = analysis.static.field.integration_point.stress
    output element_stress = analysis.static.field.element.stress
}
```

The corresponding RunMat selectors are typed:

```matlab
pointStress = stepResult.field(fea.StructuralField.integrationPointStress());
elementStress = stepResult.field(fea.StructuralField.stress());
```

Tet4 publishes the same four-entity layout with its constant tensor repeated. That keeps mixed-order
partitioning and paging uniform without implying four independent Tet4 stress states. Recovery
smoothing is region-side aware: integration points extrapolate to element nodes, but nodes shared
by different material regions do not erase a legitimate stress jump.

## How do I know a structural FEA result is credible?

At minimum, review geometry and units, material and section coverage, load resultants, support
reactions, mesh quality, discretization convergence, equation residuals, energy balance, formulation
identity, backend identity, and recovery evidence. Dynamics additionally needs time/frequency
convergence, damping justification, mass participation, and basis truncation evidence. Buckling and
nonlinear work needs imperfection and path-sensitivity studies appropriate to the engineering
decision.

Continue with [meshing](../meshing), [study contracts](../study-contracts),
[verification and validation](../validation), and [result evidence](../evidence).
