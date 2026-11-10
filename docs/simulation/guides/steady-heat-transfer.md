## Explore steady heat conduction

Hold one end of a steel bar at **300 K** and the other at **350 K**. RunMat solves for temperature inside the solid and recovers heat flux from its temperature gradient. The remaining surfaces have no applied heat flux.

<!-- fea:example:steady-conduction:setup -->

### Inspect the bar

<!-- fea:example:steady-conduction:geometry -->

The study imports the cantilever STEP geometry and uses linear tetrahedra. Steel has a thermal conductivity of **45 W/(m·K)**. This model contains no volumetric heat source or convection boundary.

### Inspect temperature

<!-- fea:example:steady-conduction:temperature -->

Color shows temperature in kelvin. Compare the prescribed ends with the interior. Rotating or zooming the saved view does not repeat the solve.

### Inspect heat flux

<!-- fea:example:steady-conduction:heat-flux -->

Color shows element heat-flux magnitude in **W/m²**. The saved output retains the vector components. Heat flows from the hotter end toward the colder end.

## Edit and run the study

Open the project in RunMat Desktop, or download it for the CLI. These excerpts come from the study source.

<!-- fea:example:steady-conduction:edit -->

From the project directory:

```sh
runmat check steady-conduction.fea
runmat plan steady-conduction.fea
runmat run steady-conduction.fea
```

Change either prescribed temperature or the conductivity, then solve again. The embedded figures retain the original example.

## Model heater power and air cooling

The downloadable heat-sink example imports a finned KiCad STEP model. A distributed heater supplies approximately **10 W** to the mounting face. Aluminum with conductivity **160 W/(m·K)** conducts heat into the fins, and a uniform film coefficient of **35 W/(m²·K)** represents cooling to **25 °C air**. Convection applies to the entire outer surface, including the heated mounting face. The example does not resolve airflow, a chip, a board or a thermal contact layer.

<!-- fea:example:heatsink-steady:setup -->

<!-- fea:example:heatsink-steady:edit -->

This project includes the geometry and its original license. Run it in a native Desktop or CLI installation to compute its fields. No saved heat-sink result is included in this guide.

## Thermal formulation

Steady conduction satisfies

$$-\nabla\cdot(k\nabla T)=Q,\qquad \mathbf q=-k\nabla T.$$

Conductivity $k$ relates the temperature gradient to heat flux $\mathbf q$. Volumetric generation $Q$ supplies heat inside a solid. The bar sets $Q=0$ and prescribes end temperatures. For convection, outward heat loss is $q_{out}=h(T-T_\infty)$.

RunMat integrates the solid and boundary terms, solves for nodal temperature, and recovers element heat flux. Choose a transient study when heat capacity and warm-up or cool-down matter.

See [thermal validation](/docs/simulation/validation/thermal) for formulation and recovery checks, and [thermal loads](/docs/simulation/loads/thermal) for temperature, power and convection boundaries.
