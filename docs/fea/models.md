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
| Signals | Quantity-typed constants and sampled histories, including interpolation and extrapolation policy. |
| Initial conditions | Family-typed state sampled from a signal before a named procedure step. |
| Response spectra | Direction, damping, response measure, modal combination, truncation, construction, and evidence for modal response-spectrum work. |
| Domains | Thermo-mechanical, electro-thermal, electromagnetic, or CFD domain data. |
| Interfaces | Contact and coupling interfaces between regions. |
| Sequence | Typed procedure steps, activation lifecycles, state transitions, observations, accuracy plans, branches, and restart policy. |

## Model Templates

Templates create a family-appropriate starting model shape. Procedure steps and execution entries,
not the template name, determine what is solved:

| Template | Starting model shape |
| --- | --- |
| `linear_static_structural` | Structural solid model with a static procedure scaffold |
| `modal_structural` | Structural model with a modal-extraction scaffold |
| `acoustic_harmonic` | Acoustic frequency-domain model |
| `thermal_steady` | Steady thermal equilibrium model |
| `thermal_transient` | Transient thermal model |
| `transient_structural` | Transient structural model |
| `nonlinear_structural` | Nonlinear structural model |
| `thermo_mechanical_coupled` | Coupled thermomechanical model |
| `electromagnetic_harmonic` | Harmonic electromagnetic model |
| `cfd_steady_state`, `cfd_transient` | Steady or transient fluid model |
| `cht_coupled` | Conjugate heat-transfer model |
| `fsi_coupled` | Fluid-structure interaction model |

In a `.fea` file, `model.defaults: template_scaffold` lets RunMat create defaults owned by the
selected template. `model.defaults: none` starts from an empty model and expects the file to
provide the physical data it needs.

When a template scaffold has one unambiguous material, the resulting model contains visible,
explicit whole-domain assignments appropriate to the template: `all_solids`, `all_cells`, or both
for a coupled solid/cell model. This is authored model data, not an assembly fallback. If imported
geometry exposes several materials without trustworthy body-to-material ownership, RunMat leaves
the assignments empty so the author must provide the mapping. Volumetric scaffold sources such as
heat generation target the whole active volume rather than borrowing a boundary-face name.

## Define Model Data

Model data can be written in a `.fea` file or assembled in RunMat code with typed constructors such as `fea.material(...)`, the family-specific constraint, boundary, signal, excitation, and initial-condition classes, `fea.ExcitationCase`, `fea.ResponseSpectrum`, concrete domain classes such as `fea.ElectromagneticDomain` and `fea.FluidDomain`, and `fea.model(...)`. Use a `.fea` file when the model definition is a portable study artifact. Use `.m` code when the model is generated from parameters, calculations, or loops.

For multi-procedure models, construct an explicit `fea.AnalysisSequence` and pass it as the
`Sequence` model option. The sequence API and its activation, state, observation, accuracy, branch,
and restart contracts are described in [Procedure Sequences, Observations, and Accuracy](./study-contracts).

```yaml
model:
  id: bracket_static_model
  template: linear_static_structural
  defaults: none
  frame: global

materials:
  steel:
    mechanical:
      youngs_modulus_pa: 200000000000.0
      poisson_ratio: 0.30

material_assignments:
  - target: all_solids
    material: steel

constraints: # typed AnalysisConstraint records
signals: # typed physical signal records
excitation_cases: # typed excitation-case records
excitations: # typed ExcitationAction records
sequence: # typed AnalysisSequence record
execution: # per-step backend, requirement, and solver options
```

Material definitions and material assignments are separate. `all_solids`, `all_shells`,
`all_beams`, and `all_cells` are explicit whole-formulation scopes. A selected scope uses a closed
target variant such as `target: { solid: { region: bracket_body } }`; the region may be a direct
geometry identity or an alias from the `regions` block. Every active material-bearing element must
match exactly one compatible assignment. A CAD face used as a volume boundary is not a shell
material domain. Beam and shell sections, laminate layups, contact/interface laws, and material
orientation are separate model properties rather than material-assignment options.

Direct moment excitations use a typed moment signal and coordinate frame. They require
rotational-DOF structural elements. Beam nodes and shell/plate nodes can own `rx`, `ry`, and `rz`;
displacement-only solid regions reject moments with an invalid-model error instead of silently
converting them to forces.

In RunMat code, construct this with a moment-valued signal and
`fea.StructuralExcitation.moment(...)`; in `.fea`, serialize the corresponding typed signal and
`moment` excitation variants. There is no parallel generic `loads` schema.

When rotational DOFs are present, structural results include `structural.rotation` and `structural.reaction_moment`. Beam results can also include `structural.beam_torsion_moment`, `structural.beam_bending_moment`, `structural.beam_bending_stress`, and `structural.beam_torsion_stress`; shell results can include `structural.shell_membrane_force`, `structural.shell_bending_moment`, `structural.shell_transverse_shear`, and `structural.shell_von_mises`.

## Validate Before Solving

Validation checks the model and study shape before a run starts. It catches problems such as:

- empty study or model ids,
- missing geometry meshes,
- unspecified units,
- sequence-step and execution-plan mismatches,
- missing required steps,
- invalid family-specific options,
- missing or incompatible domain data.
- direct moment loads on non-structural families or displacement-only structural regions.

`runmat check model.fea` is the usual CLI path. Host integrations can call `fea.validate_study/v1` or lower-level model validation operations.

## Scaffolds And Engineering Data

Template scaffolds are useful for tests, examples, early integration, and simple exploratory runs.
Engineering runs should explicitly review or provide the materials, assignments, excitations,
constraints, domains, and assumptions that matter for the result.

Validation proves the model is shaped correctly enough to run. Verification and validation evidence is still required before treating a result as production-grade physics. See [Verification & Validation](/docs/fea/validation).
