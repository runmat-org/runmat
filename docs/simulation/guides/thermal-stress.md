---
description: A worked example of thermal stress. A steel bar clamped at both ends and heated by 80 K is solved by the restrained-bar formula and by a thermomechanical finite-element analysis in RunMat, from thermal strain and the thermal load vector to stress, support force and a design check.
keywords: [thermal stress, thermal expansion, thermal strain, thermomechanical analysis, restrained bar, coefficient of thermal expansion, thermal load vector, clamp reaction, Tet10, expansion joint, thermal buckling]
doc_type: guide
family: [thermomechanical]
examples: [heated-bar-thermal-stress]
---
A part that cannot expand freely when its temperature changes develops thermal stress. Pipes between anchors, rails, bolted flanges and electronics on a stiff board all carry it. This page works one example in full: the thermal strain and restrained-bar theory of a strength-of-materials course, the finite-element formulation that carries a temperature field into a structural solve, and the solution in RunMat.

## Example 1: A clamped steel bar heated by 80 K

A steel bar is built in between two rigid walls at room temperature, 20 °C. In service it heats uniformly to 100 °C. The walls do not move. The bar must stay elastic, below its 250 MPa yield strength, and the walls must be designed for the force the bar exerts on them.

**Given**

| Quantity | Value |
| --- | --- |
| Length | L = 1 m, along x |
| Cross-section | square, 100 mm × 100 mm (A = 0.01 m²) |
| Material | steel, E = 200 GPa, ν = 0.3, α = 12 × 10⁻⁶ K⁻¹ |
| Reference temperature | T_ref = 293.15 K (20 °C), stress-free |
| Service temperature | T = 373.15 K (100 °C), uniform: ΔT = 80 K |
| Supports | both end faces fixed in all directions |
| Yield strength | 250 MPa |

**Find**

1. The axial stress in the bar.
2. The force each wall carries.
3. Whether the bar stays below yield, and what the result means for the design.

## Thermal expansion of a restrained bar

### Thermal strain

A material heated by ΔT from a stress-free state expands equally in every direction. The thermal strain is

$$
\varepsilon_{th} = \alpha\,\Delta T ,
$$

and stress arises only from the part of the strain that differs from it. In one dimension, Hooke's law becomes

$$
\sigma = E\,(\varepsilon - \alpha\,\Delta T) .
$$

### Free and restrained bars

A bar free to expand has no stress: $\varepsilon = \alpha\Delta T$, and its length grows by $\delta = \alpha\,\Delta T\,L$. Between rigid walls the total strain is zero, so the stress and the axial force are

$$
\sigma = -E\,\alpha\,\Delta T, \qquad N = \sigma A = -E A\,\alpha\,\Delta T .
$$

The stress is compressive and depends on neither the length nor the section: a long rail and a short pin heated by the same amount carry the same stress. For this bar, a free bar would grow by 0.96 mm, and the restrained bar carries σ = −192 MPa and N = −1,920 kN.

### Three dimensions

In a solid, the thermal strain is a uniform expansion, $\boldsymbol\varepsilon_{th} = \alpha\,\Delta T\,\mathbf I$, and the stress follows from the elastic part of the strain:

$$
\boldsymbol\sigma = \mathsf C : (\boldsymbol\varepsilon - \boldsymbol\varepsilon_{th}) .
$$

Away from the walls the restrained bar is in uniaxial compression, and its free sides grow sideways by the thermal strain plus the Poisson expansion under the compressive stress, $\varepsilon_y = \alpha\Delta T + \nu\,|\sigma|/E = \alpha\,\Delta T\,(1 + \nu)$. At the walls, the clamped faces also hold the section's width and depth fixed. The material there is restrained in three directions, which the one-dimensional formula does not see.

### Finite-element formulation

**Weak form.** Substituting the thermoelastic stress into the principle of virtual work moves the thermal strain to the right-hand side:

$$
\int_\Omega \boldsymbol\varepsilon(\mathbf w) : \mathsf C : \boldsymbol\varepsilon(\mathbf u)\, d\Omega
= \int_\Omega \boldsymbol\varepsilon(\mathbf w) : \mathsf C : \boldsymbol\varepsilon_{th}\, d\Omega
\quad \text{for every admissible } \mathbf w .
$$

The temperature acts as a load. No force is applied anywhere, yet the bar is loaded by its own restrained expansion.

**Discretization.** With ten-node tetrahedra and quadratic shape functions, $\mathbf u^h = \sum_a N_a \mathbf u_a$ and $\boldsymbol\varepsilon = \mathbf B_e \mathbf u_e$, as in the [linear static example](./linear-static.md#finite-element-formulation). The left side gives the usual element stiffness, and the right side gives a thermal load vector:

$$
\mathbf K_e = \int_{\Omega_e} \mathbf B_e^{\mathsf T}\mathbf D\,\mathbf B_e\, d\Omega, \qquad
\mathbf f_{th,e} = \int_{\Omega_e} \mathbf B_e^{\mathsf T}\mathbf D\,\boldsymbol\varepsilon_{th}\, d\Omega .
$$

**Assembly and supports.** Assembling gives $\mathbf K\,\mathbf u = \mathbf f_{th}$. The clamped faces prescribe zero displacement, and the reactions there are the wall forces, as in a static analysis. Stress is recovered from the elastic strain only, $\boldsymbol\sigma = \mathbf D\,(\mathbf B_e\mathbf u_e - \boldsymbol\varepsilon_{th})$.

Here the temperature is prescribed as a uniform field. When it comes from a heat-conduction solution instead, the same thermal load vector carries the computed temperature into the structural solve; the [thermomechanical physics reference](../physics/thermomechanical.md) covers that coupling.

## Solution

Each block below is an excerpt of `thermal-stress-static.fea`.

### Geometry

The bar is imported from a STEP file. Its two end faces carry CAD names, so the study can clamp them by name.

```fea source=thermal-stress-static.fea
geometry bar {
    file = "./geometry/bar.step"
    units = mm
    region body : solid { select name "Bar"; expect exactly_one }
    region end_a : face { select name "End A"; expect exactly_one }
    region end_b : face { select name "End B"; expect exactly_one }
}
```

### Mesh

Quadratic tetrahedra at a 50 mm target size give 2,692 elements and 4,679 nodes. The stress in the middle of the bar is uniform, but near the clamps it varies across the section, and quadratic elements follow that variation.

```fea source=thermal-stress-static.fea
mesh bar on geometry.bar {
    tetrahedra {
        order = quadratic
        target_size = 0.05 m
        maximum_deviation = 0.001 m
    }
}
```

### Material

A thermal stress analysis needs the elastic constants and the expansion coefficient. The `reference_temperature` is the stress-free temperature, and `modulus_temperature_coefficient = 0 K^-1` keeps E constant over the 80 K rise. The conductivity and specific heat serve heat-conduction steps and are unused when the temperature is prescribed.

```fea source=thermal-stress-static.fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
    thermal {
        reference_temperature = 293.15 K
        modulus_temperature_coefficient = 0 K^-1
        conductivity = 45 W/m/K
        specific_heat = 500 J/kg/K
        expansion_coefficient = 0.000012 K^-1
    }
}
```

### Supports and temperature

Both end faces are fixed: the rigid walls of the problem statement. The temperature is applied as a `temperature_field` load on the whole body, 373.15 K against the 293.15 K reference, which is the $\Delta T$ in $\boldsymbol\varepsilon_{th}$.

```fea source=thermal-stress-static.fea
model clamped_bar {
    thermal { reference_temperature = 293.15 K }
    thermomechanical { reference_temperature = 293.15 K }
    assign material.steel to geometry.bar.region.body
    constraint clamp_a {
        structural.fixed { target = geometry.bar.region.end_a }
    }
    constraint clamp_b {
        structural.fixed { target = geometry.bar.region.end_b }
    }
    load operating_temperature {
        temperature_field on geometry.bar.region.body = 373.15 K
    }
    load_case operating { include load.operating_temperature }
}
```

### Study and outputs

The `thermomechanical steady` procedure assembles $\mathbf K\mathbf u = \mathbf f_{th}$ and solves it. The outputs publish the displacement and stress fields, the von Mises stress at the center of the bar, the reaction on one wall, and the largest displacement.

```fea source=thermal-stress-static.fea
study heated_static {
    geometry = geometry.bar
    mesh = mesh.bar
    model = model.clamped_bar
    analysis coupled { thermomechanical steady {} }
    output displacement = analysis.coupled.field.node.displacement
    output stress = analysis.coupled.field.element.stress
    output von_mises = analysis.coupled.field.element.von_mises_stress
    output mid_von_mises = interpolate(analysis.coupled.field.node.von_mises_stress, at = [500 mm, 50 mm, 50 mm])
    output clamp_reaction = resultant(analysis.coupled.field.node.reaction_force, over = geometry.bar.region.end_a)
    output largest_displacement = max(magnitude(analysis.coupled.field.node.displacement), over = geometry.bar.region.body)
}
```

### Run

From the command line, in the example folder:

```sh
runmat check thermal-stress-static.fea
runmat run thermal-stress-static.fea
```

### Wall force

The reaction on the wall at x = 0 is **[1,975.84 kN](receipt:heated-bar-thermal-stress#clamp-reaction)**, pointing in +x: the bar pushes on the wall and the wall pushes back. The other wall carries the same force in the opposite direction, so the bar as a whole is in equilibrium with no applied load. The force is 2.9 % above the 1,920 kN of the one-dimensional formula. The clamped faces also stop the section from growing sideways near the walls, which makes the bar stiffer there; holding the stiffer bar to zero total length takes more force.

### Stress

At the center of the bar the von Mises stress is **[197.584 MPa](receipt:heated-bar-thermal-stress#mid-von-mises)**. In uniaxial compression the von Mises stress equals the magnitude of the axial stress, and dividing the wall force by the area gives the same 197.6 MPa: the axial force is the same at every section, and at mid-length it is carried uniformly.

### Deformation

The largest displacement is **[0.0905 mm](receipt:heated-bar-thermal-stress#largest-displacement)**, at the corners of the sections away from the walls. The bar cannot lengthen, so this is the sideways growth of the section. The lateral strain $\alpha\,\Delta T\,(1 + \nu) = 1.25 \times 10^{-3}$ moves a corner 50 mm from the axis by about 0.062 mm in each of two directions, or 0.088 mm in all.

### Answers

1. The axial stress is about [197.584 MPa](receipt:heated-bar-thermal-stress#mid-von-mises) in compression in the body of the bar, against 192 MPa from $E\alpha\Delta T$.
2. Each wall carries [1,975.84 kN](receipt:heated-bar-thermal-stress#clamp-reaction).
3. The stress is 79 % of the 250 MPa yield strength. The bar stays elastic, but with little margin for a modest temperature rise.

## Discussion

**Design meaning.** An 80 K rise uses almost four-fifths of the yield strength of a fully restrained steel bar, whatever its size, and a wall must resist about 2 MN. Restraint, not load, is the problem, so the usual design response removes restraint: a sliding support, an expansion joint or a flexible loop lets the part grow, and the thermal stress falls toward zero. Where restraint cannot be avoided, a material with lower $E\alpha$ reduces the stress in proportion.

**Model limits.** The walls are perfectly rigid. Real supports deflect and slip, so the computed force is an upper bound. The temperature is uniform and α and E are constant; a temperature gradient through the section adds bending, which a conduction solve feeding this study would capture. The analysis is linear and assumes the bar stays straight. Compression in a long, slender member can buckle it before it yields. For this stocky bar the fixed–fixed Euler load, $4\pi^2 EI/L^2 \approx 66$ MN, needs a temperature rise of over 2,700 K, so buckling is not a concern; for a rail or a thin pipe it often governs.

**Refinement.** The stress in the body of the bar is uniform and settles on a coarse mesh. The stress near the clamped edges does not: where the clamped face meets the free sides, the restraint produces a singular point, like the clamped edge of the [linear static example](./linear-static.md#discussion). Judge the design on the stress in the body, or model the real support.

## Try a change

**Problem 1.1.** Remove `clamp_b` so the bar can grow from the wall at x = 0. Predict the stress and the movement of the free end before running.

*Answer.* The bar expands freely: away from the wall the stress is essentially zero, the wall force vanishes, and the free end moves by $\alpha\,\Delta T\,L = 0.96$ mm.

**Problem 1.2.** Change the material to aluminum (E = 68.9 GPa, α = 23 × 10⁻⁶ K⁻¹). Predict the restrained stress.

*Answer.* $E\alpha\Delta T = 68.9\ \text{GPa} \times 23 \times 10^{-6} \times 80 \approx 127$ MPa: lower than steel, because aluminum's lower modulus outweighs its larger expansion.

**Problem 1.3.** Heat the bar to 150 °C instead (`temperature_field ... = 423.15 K`). Is it still elastic? At what temperature rise does the body of the bar reach yield?

## Next

- [Linear static analysis](./linear-static.md): the structural analysis this example builds on.
- [Thermomechanical physics reference](../physics/thermomechanical.md): coupling a heat-conduction solve to a structural one.
- [Thermomechanical validation](../validation/thermomechanical.md): this bar against the restrained-bar formula.
