## Modal studies in RunMat

Use a modal study to answer **which natural frequencies a structure has** and **how its parts move in each mode**. These patterns help you identify flexible regions, distinguish bending from twisting, and decide which modes a vibration-response model needs.

A RunMat `.fea` study connects exact geometry, elastic stiffness, density and supports. It extracts frequencies and recovers the original mode vectors onto the solver mesh, so you can inspect the motion and reopen the complete result.

| You define | RunMat computes |
| --- | --- |
| Geometry and elastic material properties | Structural stiffness |
| Density and the supported mass model | Structural mass |
| Physical restraints | The admissible free-vibration motions |
| Number of requested modes | Natural frequencies and recovered mode shapes |

A mode shape describes a **pattern of motion**, rather than the vibration amplitude under a particular force. Predicting an amplitude requires a response analysis with forcing and damping.

## Explore a tuning fork’s natural modes

**Which bending patterns appear when the stem is fixed and the tines are free?** This example uses an exact STEP tuning fork with two free tines, a connecting bridge and a stem fixed at its bottom face. There is no applied end load: the extraction asks how the supported structure can vibrate freely.

<!-- fea:example:tuning-fork-modal:setup -->

The fork is **100 mm tall**. Its tines are **8 mm wide and 6 mm thick**, separated by a **20 mm gap**, and extend **65 mm above the bridge**. Steel has Young’s modulus **200 GPa**, Poisson’s ratio **0.3** and density **7,850 kg/m³**. This is an instructional fork with rectangular sections, rather than a model of a commercial instrument.

### Inspect the structure and support

<!-- fea:example:tuning-fork-modal:geometry -->

Follow the connection from the clamped stem through the bridge to each tine. Only the stem’s bottom face is fixed. The tines and bridge remain free to participate in the extracted motion. The study requests quadratic tetrahedra with a **30 mm** target size and **0.1 mm** maximum deviation; local feature constraints can require smaller cells. Compare the resulting modes across mesh refinements when assessing accuracy.

### Inspect the first mode

<!-- fea:example:tuning-fork-modal:mode-1 -->

The frequency shown above the figure comes from the saved modal basis, in **hertz**. The colored shape is the original recovered eigenvector, rescaled for inspection relative to the fork’s dimensions. The gray outline shows the undeformed fork. Color measures amplitude relative to that mode’s largest stored vector magnitude: **1** is its maximum. Rotate the view to see whether the tines move within the fork’s plane or perpendicular to it.

Follow the motion down through the bridge and stem: fixing the bottom face still allows the rest of the stem to bend. Compare the two tine tips to distinguish motion together from motion in opposite directions.

### Compare the second mode

<!-- fea:example:tuning-fork-modal:mode-2 -->

Compare the two tine tips, their relative directions, and the bridge motion. A fork has several families of natural motion; its lowest pair need not be the familiar audible in-plane tuning mode. The model’s stiffness, cross-section, mass and stem restraint determine their ordering.

These are static views of modal patterns. Reversing an entire eigenvector gives the same mode. Display scale and color do not imply a measured vibration amplitude, a forcing level or an animation time.

## Build and modify this study in RunMat

The complete project opens in RunMat Desktop with exact STEP geometry, its reproducible geometry generator, the editable study, original modal basis and saved figures. The excerpts below come directly from the study: import and mesh the fork, assign stiffness and density, fix the stem end, then extract two modes and retain their basis. The support alias identifies this STEP model; select the corresponding stem face when using an updated shape.

<!-- fea:example:tuning-fork-modal:edit -->

### Try a change

Try changes that distinguish the physical inputs:

- **Change stiffness or density.** With geometry and supports fixed, higher stiffness generally raises frequencies and higher density generally lowers them.
- **Change the stem restraint.** A fixture’s stiffness is part of the vibration problem. Fixing extra faces changes the physical model.
- **Change tine dimensions in CAD.** Import the updated STEP model and check the stem support selection. The complete project also includes the generator used to create this fork.
- **Request more modes and refine.** Examine the frequency range and shapes needed for your application, and compare them across meshes.

Each change produces a new extraction. The embedded figures continue to show the original modes.

### Run from the command line

From the downloaded project directory:

```sh
runmat check tuning-fork-modal.fea
runmat plan tuning-fork-modal.fea
runmat run tuning-fork-modal.fea
```

**Check** validates the source. **Plan** describes geometry, meshing, extraction and output work. **Run** executes the study and publishes its basis and recovered fields. Geometry import, meshing and extraction run in a native Desktop or CLI installation.

## How RunMat extracts natural modes

For linear free vibration, RunMat solves

$$
\mathbf K\boldsymbol\phi_i=\omega_i^2\mathbf M\boldsymbol\phi_i,
\qquad f_i=\frac{\omega_i}{2\pi}.
$$

Stiffness $\mathbf K$ and mass $\mathbf M$ determine each mode vector $\boldsymbol\phi_i$ and frequency $f_i$ in hertz. Supports constrain the admissible motion. RunMat normalizes and retains the modal basis, then reconstructs its original vectors on the mesh for durable field inspection. The relative display is a view transformation; opening the figure does not repeat the eigensolve.

## Supported scope and next steps

This extraction uses RunMat's modal formulation. Check material, mass and restraint assignments, then compare frequencies and mode shapes across meshes. For close or repeated modes, compare their physical motion or subspace rather than requiring identical signed vectors.

[Modal validation](/docs/simulation/validation/modal) describes the spectral, basis and workflow criteria. To predict driven motion, continue with a response study and its forcing, damping, frequency-range and modal-coverage requirements.
