---
title: "Models"
category: "FEA"
section: "13.3"
last_updated: "June 22, 2026"
---

# Models

A model attaches physics data to geometry. It defines what material exists where, how the model is constrained, what loads or sources apply, which domains are active, and which analysis steps can be run.

## What A Model Contains

| Part | Purpose |
| --- | --- |
| Geometry identity | The geometry id, revision, units, and reference frame the model belongs to. |
| Materials | Mechanical, thermal, electrical, and plastic material properties. |
| Material assignments | Which regions use which materials, with optional confidence evidence. |
| Boundary conditions | Constraints, prescribed displacements, electromagnetic anchors, and related boundary data. |
| Loads and sources | Forces, moments/torques, pressures, body forces, current density, coil currents, and similar inputs. |
| Domains | Thermo-mechanical, electro-thermal, electromagnetic, or CFD domain data. |
| Interfaces | Contact and coupling interfaces between regions. |
| Steps | The analysis steps the model supports, such as static, modal, thermal, transient, nonlinear, electromagnetic, or CFD. |

## Model Profiles

Profiles create the starting model shape and select the solver used by `run`:

| Profile | Selected solver |
| --- | --- |
| `linear_static_structural` | `linear_static` |
| `modal_structural` | `modal` |
| `acoustic_harmonic` | `acoustic` |
| `thermal_standalone` | `thermal` |
| `transient_structural` | `transient` |
| `nonlinear_structural` | `nonlinear` |
| `thermo_mechanical_coupled` | `transient` |
| `electromagnetic_static` | `electromagnetic` |
| `cfd_steady_state`, `cfd_transient` | `cfd` |
| `cht_coupled` | `cht` |
| `fsi_coupled` | `fsi` |

In a `.fea` file, `model.defaults: profile_scaffold` lets RunMat create the scaffold from the profile. `model.defaults: none` starts from an empty model and expects the file to provide the model data it needs.

## Define Model Data

Model data can be written in a `.fea` file or assembled in RunMat code with typed constructors such as `fea.material(...)`, `fea.boundaryCondition(...)`, `fea.loadCase(...)`, `fea.domain(...)`, and `fea.model(...)`. Use a `.fea` file when the model definition is a portable study artifact. Use `.m` code when the model is generated from parameters, calculations, or loops.

```yaml
model:
  id: bracket_static_model
  profile: linear_static_structural
  defaults: none
  frame: global

materials:
  steel:
    mechanical:
      youngs_modulus_pa: 200000000000.0
      poisson_ratio: 0.30

material_assignments:
  - region: bracket_body
    material: steel

boundary_conditions:
  - id: fixed_base
    region: bracket_base
    kind: fixed

loads:
  - id: tip_force
    region: bracket_tip
    type: force
    vector: [0.0, -1000.0, 0.0]

steps:
  - id: static_step
    kind: static
```

Region references can point directly at geometry region ids or use named aliases from the `regions` block.

`type: moment` uses a global-frame vector in N*m. `type: torque` is accepted as an alias but resolves to the canonical `moment` load kind. Direct moment loads require rotational-DOF structural elements. Beam nodes and shell/plate nodes can own `rx`, `ry`, and `rz`; displacement-only solid regions reject moment loads with an invalid-model error instead of silently converting them to forces.

```yaml
loads:
  - id: tip_moment
    region: node:2
    type: moment
    vector: [0.0, 0.0, 125.0]
```

When rotational DOFs are present, structural results include `structural.rotation` and `structural.reaction_moment`. Beam results can also include `structural.beam_torsion_moment`, `structural.beam_bending_moment`, `structural.beam_bending_stress`, and `structural.beam_torsion_stress`; shell results can include `structural.shell_membrane_force`, `structural.shell_bending_moment`, `structural.shell_transverse_shear`, and `structural.shell_von_mises`.

## Validate Before Solving

Validation checks the model and study shape before a run starts. It catches problems such as:

- empty study or model ids,
- missing geometry meshes,
- unspecified units,
- legacy run-kind/profile mismatches,
- missing required steps,
- invalid family-specific options,
- missing or incompatible domain data.
- direct moment loads on non-structural families or displacement-only structural regions.

`runmat check model.fea` is the usual CLI path. Host integrations can call `fea.validate_study/v1` or lower-level model validation operations.

## Scaffolds And Engineering Data

Profile scaffolds are useful for tests, examples, early integration, and simple exploratory runs. Engineering runs should explicitly review or provide the materials, assignments, loads, constraints, domains, and assumptions that matter for the result.

Validation proves the model is shaped correctly enough to run. Verification and validation evidence is still required before treating a result as production-grade physics. See [Verification & Validation](/docs/fea/validation).
