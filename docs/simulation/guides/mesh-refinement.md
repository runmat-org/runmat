## Refine the quantities that matter

Mesh refinement helps you decide whether a computed quantity is stable enough for its intended use. In RunMat, repeat the same physical study with a changed mesh, then compare displacement, stress, reactions or another quantity at the same physical location. Keep the geometry, material, loads, supports and solver policy fixed while isolating the effect of discretization.

A small solver residual means the discrete equations were solved closely. Refinement asks a different question: how much does the answer change when those equations are built from a different mesh? Choose the quantity first, then design the comparison.

## Compare three meshes of the same cantilever

These projects use the cantilever from the linear-static guide: a 1 m steel beam with a 100 mm square cross-section, a fixed x = 0 face, and a 100 kPa downward traction on the free end. All three use quadratic tetrahedra and the same material and loading. Only the requested target size changes: **300 mm**, **175 mm**, and **150 mm**.

A requested size is not an element count. Surface constraints, feature dimensions and meshing quality can make actual element sizes smaller, so inspect the resulting mesh rather than comparing the request alone.

### Coarse request: 300 mm

<!-- fea:example:cantilever-coarse:geometry -->

### Medium request: 175 mm

<!-- fea:example:cantilever-medium:geometry -->

### Fine request: 150 mm

<!-- fea:example:cantilever-static:geometry -->

Rotate the meshes to see where resolution has actually changed. The saved projects contain their own original solver meshes and results; the browser does not regenerate them.

## Compare displacement and stress separately

For bending, a useful global quantity is free-end displacement. A useful local quantity is stress near the fixed end. They can stabilize at different rates: displacement reflects the overall deformation, while stress depends on spatial derivatives and local support behavior.

The saved runs shown here give the following values:

| Requested size | Nodes | Elements | Maximum displacement | Maximum element von Mises stress |
| --- | ---: | ---: | ---: | ---: |
| 300 mm | 748 | 273 | 0.199405 mm | 4.8639 MPa |
| 175 mm | 774 | 295 | 0.199356 mm | 5.1441 MPa |
| 150 mm | 786 | 305 | 0.199370 mm | 5.1440 MPa |

Using the finer result as the denominator, displacement changes by **0.025%** and then **0.007%**; the stress peak changes by **5.446%** and then **0.003%**. These small displacement changes support checking that global quantity further. The stress peak's larger first change and very small second change still need additional refinement or a local reference before treating the peak as stable. This sequence supplies observed changes, not an error bound.


<!-- fea:example:cantilever-static:displacement -->

The color legend reports physical displacement in meters. The deformation is enlarged for visibility, with the scale stated under the figure. Read the numerical values when comparing meshes; visually similar exaggerated shapes can hide a meaningful difference.

<!-- fea:example:cantilever-static:stress -->

Stress color is an original element field. Adjacent triangles can belong to elements with different projected stresses; no visual smoothing has been applied. The largest element value is not automatically a mesh-independent design stress. Sharp corners and idealized supports can make a local peak especially sensitive to refinement.

## Repeat the comparison in RunMat

Open the projects below to inspect all three studies and their saved results. The source excerpts show exactly which target size each project requests.

### Coarse project

<!-- fea:example:cantilever-coarse:edit -->

### Medium project

<!-- fea:example:cantilever-medium:edit -->

### Fine project

<!-- fea:example:cantilever-static:edit -->

Choose a result quantity and location, run a sequence of meshes, and record the actual element/node inventory and the value from each solve. Use at least enough levels to see a trend. Compare relative changes against a tolerance chosen for that quantity and its engineering purpose; do not select a tolerance merely because the last two answers satisfy it.

## Plan refinement around the model

For a new part, concentrate refinement where the physics changes quickly: a bending support, a small loaded face, a material interface, or a thermal gradient. Element order matters as well as size. Compare size refinement within one order first; use an order comparison as a separate experiment. Quadratic elements can represent bending more effectively than a coarse linear-tetrahedral mesh, but choosing a higher order does not establish convergence by itself.

Each new mesh is a new solve. Displaying a denser surface or smoothing a contour does not refine the equilibrium problem. Changing a support, a traction area or a material assignment at the same time also changes the physics, so keep those comparisons separate from a mesh study.

Use [structural validation](/docs/simulation/validation/structural) to inspect the reference and convergence requirements behind the formulation. This guide demonstrates a controlled comparison across meshes; it is not an automatic adaptation loop or a declaration that every plotted quantity has converged.
