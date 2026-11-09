## Heat-transfer studies in RunMat

Use a thermal study to answer **how hot a component gets**, **where heat flows**, and **whether a cooling path can carry the applied power**. RunMat connects exact geometry, thermal materials, heat sources and boundary conditions in one editable `.fea` study, then publishes temperature and heat-flux fields on the solver mesh.

| You define | RunMat computes |
| --- | --- |
| Solid geometry and thermal conductivity | Temperature throughout the solid |
| Heater flux or volumetric heat generation | Heat-flux direction and magnitude |
| Prescribed temperatures or convection to an ambient | Heat exchange through the selected boundaries |
| Mesh and analysis procedure | Saved fields you can inspect and reopen |

A steady study asks for the temperature after the system reaches equilibrium. A transient study adds heat capacity and time-dependent inputs when warm-up and cool-down matter.

## Explore an electronics heat sink

**How does heat entering a mounting face reach the cooling fins?** This example imports a real finned heat-sink STEP model from the KiCad library. A distributed heater supplies approximately **10 W** to its flat mounting face. Aluminum conducts that heat into the fins, while a prescribed convection boundary exchanges heat with **25 °C air**.

<!-- fea:example:heatsink-steady:setup -->

The model uses conductivity **160 W/(m·K)** and a uniform film coefficient **35 W/(m²·K)**. It is an instructional solid-conduction model: air is represented by its ambient temperature and film coefficient, rather than an airflow mesh. Convection applies to the entire outer surface, including the heated mounting face. The heater is a distributed surface input; this example does not resolve a chip, board or thermal contact layer.

### Inspect the heat path

<!-- fea:example:heatsink-steady:geometry -->

Rotate the model to see the flat base, the thin fins and the three mounting holes. The imported part measures approximately **125 × 35 × 35 mm**. The requested mesh uses linear tetrahedra, a **10 mm** target size and **0.1 mm** maximum deviation. The STEP geometry and its original license travel with the complete example project. [KiCad’s source model](https://github.com/KiCad/kicad-packages3D/blob/b8b3cfdfad88ba66f21002b3de51dc6f7d55ba5a/Heatsink.3dshapes/Heatsink_125x35x50mm_3xFixationM3.step) supplies the shape; the material and thermal conditions here are RunMat study inputs.

### Follow the temperature into the fins

<!-- fea:example:heatsink-steady:temperature -->

Color shows solved temperature in **kelvin**. The ambient is 298.15 K; subtract 273.15 to read Celsius. Look for the temperature difference between the heated base and the fins. The displayed range comes from the saved field, and rotation or zoom does not run another thermal solve.

### See where the heat travels

<!-- fea:example:heatsink-steady:heat-flux -->

Color shows **element heat-flux magnitude in W/m²**, recovered from the solved temperature gradient. It measures heat flow per unit area, rather than total heater power. The complete project retains all three vector components, including direction; this contour displays their magnitude. Compare the base and fin roots with the fin tips to see how the conducting path distributes the heater input.

Temperature answers how hot the solid is; heat flux describes energy transport. To check the whole model’s energy balance, compare integrated heater input with integrated convection loss.

## Build and modify this study in RunMat

The complete project opens in RunMat Desktop with the original STEP geometry, editable study and saved results. The excerpts below come directly from that study: import and mesh the solid, assign aluminum, apply the heater and convection boundaries, then request temperature and heat-flux fields. The face aliases identify this STEP model; select the corresponding heating and cooling faces when using another part.

<!-- fea:example:heatsink-steady:edit -->

### Try a change

Try a change with a clear physical meaning:

- **Increase heater power.** The source is heat flux in W/m². Multiply it by the resolved mounting-face area to obtain watts; the example’s approximately 4,366 mm² face converts 2,290 W/m² into approximately 10 W.
- **Improve air cooling.** Increase the film coefficient while retaining the same ambient. This changes the heat-transfer boundary; it does not simulate a fan or infer an airflow speed.
- **Change the conducting material.** Compare temperature spread through the base and fins when conductivity changes.
- **Refine near thin fins and mounting holes.** Compare peak temperature and integrated cooling power, not just the appearance of the contour.

Each change produces a new solve. The embedded figures continue to show the original example.

### Run from the command line

From the downloaded project directory:

```sh
runmat check heatsink-steady.fea
runmat plan heatsink-steady.fea
runmat run heatsink-steady.fea
```

**Check** validates the source. **Plan** describes the geometry, mesh, analysis and output work. **Run** executes the study and publishes its results. Geometry import, meshing and solving run in a native Desktop or CLI installation.

## How RunMat computes heat transfer

Inside the solid, steady conduction satisfies

$$-\nabla\cdot(k\nabla T)=Q,\qquad \mathbf q=-k\nabla T.$$

Conductivity $k$ relates the temperature gradient to heat flux $\mathbf q$. Volumetric heat generation $Q$ supplies energy inside a solid; this example instead supplies it through a surface flux. At a convection boundary, outward heat loss is

$$q_{out}=h(T-T_\infty).$$

RunMat integrates these solid and boundary terms, solves for nodal temperature, and recovers heat flux from the gradient. The mounting-face heat input is positive inward. Convection removes energy where the surface is hotter than the ambient. The surface temperature is solved; it is not fixed to the air temperature.

## Supported scope and next steps

This example uses RunMat's steady thermal formulation. Its uniform film coefficient and distributed heater deliberately keep the cooling model simple. For a specific assembly, use the actual source footprint, contact path and cooling conditions; choose a transient study when thermal timing matters.

[The thermal validation section](/docs/simulation/validation/thermal) describes the formulation, energy, field-recovery and workflow checks. Continue with [thermal loads](/docs/simulation/loads/thermal) to choose between power, temperature and convection boundaries.
