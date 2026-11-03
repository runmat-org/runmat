# Joule heating

Electrothermal analysis couples steady current flow to heat transfer. The resistive loss
$\sigma(T)|\nabla V|^2$ is the volumetric heat source, and the temperature changes the
conductivity, which changes where the current flows. RunMat solves the two together, steady or over
time, from a `.fea` study:

| Procedure | What runs |
| --- | --- |
| `electrothermal steady {}` | Current flow and steady heat transfer iterated to the two-way fixed point |
| `electrothermal transient { start_time; end_time; time_step }` | Current flow at every time step and backward-Euler heat transfer, iterated over the whole history |

## What equations are solved?

$$
\nabla\cdot(\sigma(T)\nabla V)=0,\qquad
\sigma(T)=\frac{\sigma_{ref}}{1+\alpha\,(T-T_{ref})},
$$

$$
\rho c_p\frac{\partial T}{\partial t}-\nabla\cdot(k\nabla T)=\sigma(T)|\nabla V|^2+Q_{other},
$$

with $\alpha$ the material's `resistive_heating_coefficient` (resistivity linear in temperature,
$\rho(T)=\rho_{ref}(1+\alpha(T-T_{ref}))$) and $T_{ref}$ its electrical reference temperature.

Both fields live on the same canonical mesh, so the loss reaches the thermal solve element by element
with no interpolation, and the integrated heat source equals the electrical power.

**Coupling.** Each element's conductivity is evaluated at its mean temperature. A sweep solves current
flow with the current temperatures, then the heat transfer with the resulting Joule density. Sweeps
repeat until the largest change of any element temperature is below $10^{-6}$ of the temperature
rise. For a transient study each sweep solves current flow at every time point (sources such as a
fault-current signal are sampled at that time) and then the whole transient with the tabulated Joule
source. With $\alpha=0$ the first sweep is already exact (one-way coupling) and the second confirms it.

## How do I author an electrothermal study?

```fea
model heated_conductor {
    thermal { reference_temperature = 293.15 K }
    assign material.copper to geometry.busbar.region.body
    constraint ground {
        electrical.potential { target = geometry.busbar.region.ground_terminal; value = 0 V }
    }
    load current {
        electric_current on geometry.busbar.region.input_terminal = 400 A
    }
    load cooling {
        thermal.convection {
            target = geometry.busbar.region.exterior
            ambient_temperature = 298.15 K
            coefficient = 10 W/m^2/K
        }
    }
}

study heated_busbar {
    geometry = geometry.busbar
    mesh = mesh.busbar
    model = model.heated_conductor
    analysis operating_point { electrothermal steady {} }
    output temperature = analysis.operating_point.field.node.temperature
    output current_density = analysis.operating_point.field.element.electric_current_density
}
```

## What does a run publish?

Two cases:

- `<analysis>`: the heat transfer at the converged operating point (temperature, temperature
  gradient, heat flux; the full history for a transient), with the converged Joule source bound.
- `<analysis>/conduction`: current flow at the converged temperatures (potential, electric field,
  current density, Joule power density), at the last time point of a transient.

Each authored output belongs to the case that produces its quantity. The run log records every
sweep's temperature change and the Joule power and terminal power ($\sum V I$), which agree.

## What material and boundary data are required?

Every conductor needs `conductivity`, `resistive_heating_coefficient` and `reference_temperature`
(electrical), and `conductivity`, `specific_heat` and `mass_density` (thermal). Terminals need at
least one prescribed potential per conductor; thermal boundaries need a sink (fixed temperature,
convection or radiation). Heat crossing a joint can use a thermal contact interface; an electrical
contact resistance is modeled as a thin body with its own conductivity.

## How accurate is it?

A bar with fixed end temperatures carrying a fixed current has the closed-form solution
$\theta(x)=\left(\cos(m(x-L/2))/\cos(mL/2)-1\right)/\alpha$ with $m^2=\alpha J^2\rho_{ref}/k$. RunMat
reproduces it to the tolerances in [electromagnetic validation](../validation/electromagnetic), and
its Joule power matches $I^2R(T)$ integrated along the bar.

See [electromagnetics](./electromagnetic), [heat transfer](./thermal),
[electromagnetic loads](../loads/electromagnetic), [thermal loads](../loads/thermal), and
[current status](../status).
