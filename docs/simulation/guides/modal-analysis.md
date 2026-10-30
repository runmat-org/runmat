---
description: Worked modal analysis in RunMat. Extract a steel tuning fork's natural frequencies and mode shapes, check the lowest frequency against an excitation band, and run the study from the CLI.
keywords: [modal analysis, natural frequency, mode shape, eigenfrequency, vibration, resonance, tuning fork, Tet10, frequency margin, excitation band]
doc_type: guide
family: [structural]
examples: [tuning-fork-modal]
---
A modal analysis finds the natural frequencies of a supported structure and the shape it moves in at each one. Use it to keep resonances away from the frequencies a machine, a road or a motor will drive. Use it also to see which parts of a design are flexible, and to choose the modes a forced-response study needs. A mode shape is a pattern of motion, not an amplitude: how far the structure actually moves depends on the forcing and the damping, which a [harmonic or transient response](../language/reference/procedures.md#structural-procedures) adds.

## The question

A steel tuning fork is clamped at the bottom of its stem and mounted on equipment that vibrates anywhere from 0 to 250 Hz.

- Is the fork's lowest natural frequency at least **20 % above the top of the excitation band**, that is, above **300 Hz**?
- Which motion does each of the first two modes involve?

<!-- fea:example:tuning-fork-modal:setup -->

| Input | Value | Where |
| --- | --- | --- |
| Geometry | 100 mm tall; tines 8 mm wide, 6 mm thick, 65 mm above the bridge, 20 mm apart | `tuning-fork.step`, millimeters |
| Support | all displacement fixed | the stem's bottom face, "Stem end" |
| Steel | E = 200 GPa, ν = 0.3, ρ = 7,850 kg/m³ | the body, "Tuning fork" |
| Mesh | quadratic tetrahedra (Tet10), 30 mm target size, 0.1 mm surface deviation; 11,144 nodes | |
| Analysis | `structural modal`, 2 modes | |

No load is applied: a modal analysis asks how the supported structure can vibrate freely. Density matters as much as stiffness, because the frequencies come from their ratio.

## Inspect the model

<!-- fea:example:tuning-fork-modal:geometry -->

Only the stem's bottom face is fixed; the rest of the stem, the bridge and both tines are free to move. The 30 mm target size is larger than the part's features, so the 6 mm tines and the 0.1 mm surface deviation set the actual element size.

## Results

### Mode 1

<!-- fea:example:tuning-fork-modal:mode-1 -->

The first natural frequency is **[364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency)**. Color shows the relative amplitude of the mode shape, scaled so that its largest displacement is 1; the deformed shape is exaggerated for viewing. Rotate the view to see whether the tines move in the plane of the fork or across it, and whether they move together or against each other.

### Mode 2

<!-- fea:example:tuning-fork-modal:mode-2 -->

The second natural frequency is **[475.17 Hz](receipt:tuning-fork-modal#mode-2-frequency)**. Compare the tine tips and the bridge with mode 1. The lowest modes of a clamped fork are not necessarily the familiar audible mode, in which the tines move against each other in the plane of the fork. The stem's stiffness, the tine section and the bridge decide the order.

A mode shape's sign is arbitrary: the same mode can be stored with every displacement reversed.

## Check against the criterion

| Criterion | Limit | Result | Margin |
| --- | --- | --- | --- |
| Lowest natural frequency | above 300 Hz (250 Hz + 20 %) | [364.85 Hz](receipt:tuning-fork-modal#mode-1-frequency) | 46 % above the band's 250 Hz top |

The first mode is well clear of the excitation band. If the band reached 330 Hz, the margin would fall to 11 % and a stiffer stem or a shorter tine would be the design response. A frequency is mesh-dependent like any result: before relying on a margin this size, rerun with a finer mesh and confirm the first frequency changes by much less than the margin. The [mesh refinement guide](./mesh-refinement.md) shows the procedure.

## Build and modify the study

The complete project opens in RunMat Desktop with the STEP geometry, the study and the saved modes.

<!-- fea:example:tuning-fork-modal:edit -->

The analysis step requests the number of modes and retains the modal basis for later steps:

```fea source=tuning-fork-modal.fea
    analysis modes { structural modal { modes = 2 } retain basis }
```

A modal basis feeds the modal-superposition procedures: harmonic response, transient response, response spectrum and random vibration (see [studies, analyses and state](../language/studies-analyses-and-state.md#order-steps-and-pass-state)). To state the criterion in the study, publish the frequencies as an output of the study ([not yet in the current release](pending:output.mode-location)):

```fea fragment
output frequencies = analysis.modes.field.mode.modal_frequency
```

## Try a change

- **Use aluminum** (E = 68.9 GPa, ρ = 2,700 kg/m³). The ratio E/ρ is almost the same as steel's, so the frequencies barely change. Frequency scales with $\sqrt{E/\rho}$ when only the material changes.
- **Request more modes** (`modes = 10`) to see the next bending and twisting patterns and how far above the band they lie.
- **Shorten the tines in CAD** and re-export with the same face names. Frequency rises roughly with the inverse square of the tine length for the tine bending modes.
- **Fix more of the stem.** A support's stiffness is part of the vibration problem: fixing extra faces raises the frequencies, and so does an idealized rigid clamp compared with a real fixture.

## Run from the command line

```sh
runmat check tuning-fork-modal.fea
runmat plan tuning-fork-modal.fea
runmat run tuning-fork-modal.fea
```

`run` meshes the fork, extracts the requested modes and publishes the frequencies and the mode shapes, as nodal displacement fields with the modal basis. Read them in `.m` code with `fea.results(runId, "ModeIndices", [1 2])`; see [get started](../get-started.md#read-results-in-m-code).

## How RunMat computes this

RunMat assembles the stiffness matrix $\mathbf K$ and the mass matrix $\mathbf M$ from the Tet10 elements, removes the fixed degrees of freedom, and solves the generalized eigenproblem

$$
\mathbf K\boldsymbol\phi_i=\omega_i^2\mathbf M\boldsymbol\phi_i,
\qquad f_i=\frac{\omega_i}{2\pi},
$$

for the lowest `modes` eigenpairs with a subspace iteration on the shared sparse solver. Each mode vector is normalized to unit modal mass, $\boldsymbol\phi_i^{\mathsf T}\mathbf M\boldsymbol\phi_i = 1$, and the retained basis is stored with the result for later modal-superposition steps. The figures rescale each mode for display; the stored vectors keep their mass normalization. The element matrices, the eigen solver options and the treatment of rigid-body modes are in the [structural physics reference](../physics/structural.md).

## Validation

[Modal validation](../validation/modal.md) states how frequencies are checked against independent references and mesh refinement, and how the basis is checked for residual, mass normalization and orthogonality.
