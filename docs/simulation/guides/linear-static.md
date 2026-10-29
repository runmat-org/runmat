## Structural studies in RunMat

Use linear static analysis to answer questions such as **how far a part deflects**, **where stress concentrates**, and **what forces its supports carry** under a steady load.

A RunMat study connects geometry, material, mesh, supports, and loads in one readable `.fea` file. The native solver publishes results that you can inspect on the mesh, save, and reopen. Start from imported exact geometry or construct the geometry in the study itself.

| You define | RunMat computes |
| --- | --- |
| Elastic material properties and solid geometry | Displacement and strain |
| Prescribed displacements and fixed supports | Stress and support reactions |
| Surface traction or volume loads | The structure's equilibrium response |
| Linear or quadratic tetrahedral elements | Fields on the resulting solver mesh |

This procedure models small-strain elastic behavior. Inertia, plasticity, changing contact, and large displacement belong to their corresponding dynamic or nonlinear procedures.

## Explore a steel cantilever

**How much does a steel beam bend under a downward end load, and where is stress highest?**

The beam is **1 m long**, with a **100 mm × 100 mm** square cross-section. Its end at **x = 0** is fixed. A downward traction of **100 kPa** acts over the free end at **x = 1 m**—a total load of **1,000 N** in the negative z direction. Steel has Young's modulus **200 GPa** and Poisson's ratio **0.3**.

<!-- fea:example:cantilever-static:setup -->

### Inspect the model

<!-- fea:example:cantilever-static:geometry -->

Rotate the solver mesh to inspect the beam. The fixed end is the x = 0 face; the opposite face carries the load shown in the schematic. This example uses **quadratic tetrahedra** with a requested target size of **150 mm**.

### See the deformation

<!-- fea:example:cantilever-static:displacement -->

Color shows **displacement magnitude in meters**. The fixed end stays in place and the free end moves downward. The displayed shape exaggerates the deformation to make it visible; the scale beneath the figure applies to the shape, while the legend reports physical displacement.

### Find the stressed region

<!-- fea:example:cantilever-static:stress -->

Color shows **element von Mises stress in pascals**, projected from the quadratic elements’ integration points. Each surface triangle displays its owning element’s value; this is an element field, without visual smoothing. Look near the fixed end, where the support resists bending. Von Mises stress is a scalar summary of the stress components; the study also retains the underlying structural result fields.

All three views come from the same saved solve. You can rotate and zoom immediately after the viewer loads; exploring these results does not run an analysis on your machine.

## Build and modify this study in RunMat

The complete example opens in RunMat Desktop with its source, exact STEP geometry, and saved results. These excerpts are taken directly from that source. The end-face aliases identify the included STEP model; select the corresponding support and load faces when using your own part.

<!-- fea:example:cantilever-static:edit -->

### Try a change

- **Double the traction** from 100 kPa to 200 kPa. For the same mesh and linear model, displacement and stress should double.
- **Change Young's modulus** to see how stiffness changes deformation.
- **Refine the mesh** and compare free-end displacement and stress away from the support corners.

Each change produces a new solve. The embedded figures continue to show the original example.

### Run from the command line

From the downloaded project directory:

```sh
runmat check cantilever-static.fea
runmat plan cantilever-static.fea
runmat run cantilever-static.fea
```

**Check** validates the source. **Plan** describes the geometry, meshing, analysis, and output work. **Run** executes the study and publishes its results. Geometry construction, meshing, and solving run in a native Desktop or CLI installation.

## How RunMat computes the response

The solver finds displacements that balance the applied forces with the material's elastic response and the prescribed supports. Small strain comes from the displacement gradient; elastic stiffness relates strain to stress:

$$
\boldsymbol\varepsilon = \tfrac12(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T}),\qquad
\boldsymbol\sigma = \mathsf C : \boldsymbol\varepsilon,\qquad
\mathsf K\mathbf u = \mathbf f.
$$

**Displacement** answers how much the part moves. **Stress** describes its internal loading. **Support reactions** describe the forces needed to hold the constrained regions in place. Check that the reactions balance the applied load when interpreting a study.

For this beam, compare the displacement near the free end as you refine the mesh. Stress at an idealized clamp can be sensitive to the boundary condition and local mesh size; inspect the distribution and compare locations away from sharp support corners as well.

<details>
<summary>Material, mesh, and display details</summary>

The source declares density **7,850 kg/m³**. This traction-only static case uses no mass-dependent load. The requested mesh deviation is **1 mm**. Geometry and displacement use meters; stress uses pascals. The undeformed outline in the displacement view helps distinguish the original shape from the exaggerated deformation.

</details>

## Supported scope and next steps

This example uses RunMat's **linear-static formulation** for small-strain elastic solids. Its saved result demonstrates the complete workflow. For your own model, check material and boundary assignments, load balance, and convergence of the quantities that matter.

Read [Structural validation](/docs/simulation/validation/structural) for the formulation checks, physical references, acceptance criteria, and available assessment records behind this capability.
