# Thermomechanical FEA in RunMat: thermal expansion, stress, and coupled state

Thermomechanical analysis computes structural response caused by a temperature field, optionally in
combination with mechanical loading. RunMat treats temperature as typed state and thermal strain as
a constitutive contribution; it does not convert temperature into an arbitrary equivalent force.

The `.fea` language checks steady, transient, and nonlinear thermomechanical procedure contracts.
The numerical library currently admits experimental structural realizations driven by a
**one-pass prescribed temperature-delta profile**. It is not yet a bidirectionally coupled thermal
and structural solve, and the public typed `.fea` executor does not connect it end to end.

## What thermomechanical equation is implemented?

For isotropic thermal expansion,

$$
\boldsymbol\varepsilon_{th}=\alpha(T-T_{ref})\mathsf I,
\qquad
\boldsymbol\sigma=\mathsf C:
(\boldsymbol\varepsilon(\mathbf u)-\boldsymbol\varepsilon_{th}).
$$

The structural equilibrium problem is

$$
\mathsf K\mathbf u=\mathbf f_{mech}+\mathbf f_{th},
\qquad
\mathbf f_{th}=\int_\Omega\mathsf B^{\mathsf T}\mathsf C
\boldsymbol\varepsilon_{th}\,d\Omega.
$$

The current experimental realization evaluates a prescribed temperature-difference profile, combines
it with each admitted structural spatial operator (Tet4, Tet10, Euler–Bernoulli beam, MITC3 shell,
or the explicitly synthetic test branch), and records
`OnePassPrescribedThermomechanicalProfile`. It does not solve the heat equation in the same step or
feed deformation back into temperature.

## Which thermomechanical procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `thermomechanical steady` | Coupled static thermal/structural state | One-pass prescribed-temperature structural response |
| `thermomechanical transient` | Time samples and coupled evolving state | Same profile family with experimental transient structural policy |
| `thermomechanical nonlinear` | Load-factor domain and coupled nonlinear state | Same profile family with experimental nonlinear structural policy |

These are not bidirectional thermal solves. The transient/nonlinear coupling helpers adjust bounded
convergence policies from a thermal-strain severity measure; they do not create a thermal operator.

## How do I describe a thermally loaded structural study?

```fea
model heated_bracket_model {
    assign material.aluminum to geometry.bracket.region.body
    constraint support {
        structural.fixed { target = geometry.bracket.region.mounting_face }
    }
    load operating_temperature {
        temperature_field on geometry.bracket.region.body = 373.15 K
    }
}

study heated_bracket {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.heated_bracket_model
    formulation = experimental
    analysis coupled { thermomechanical steady {} }
    output displacement = analysis.coupled.field.node.displacement
    output stress = analysis.coupled.field.element.von_mises_stress
}
```

The material needs elastic and thermal-expansion capabilities and a declared reference temperature.
Constraints control whether expansion is free, partially restrained, or fully restrained. A uniform
temperature rise in a free homogeneous body should produce expansion but no stress; that is a useful
model check.

## How do I transfer a solved temperature field into structural analysis?

The intended study graph names the exact retained thermal state and makes dependency explicit:

```fea
analysis temperature { thermal steady {} retain temperature }

analysis deformation after analysis.temperature {
    thermomechanical steady {
        temperature = analysis.temperature.state.temperature
    }
}
```

The typed state must retain mesh/topology, units, reference temperature, and producer identity.
Today this full source-to-source state transfer is a contract target, not an admitted public
execution path; use current status/evidence rather than assuming the two blocks will run.

## Why is my thermal stress wrong?

Check coefficient-of-expansion units, reference temperature, material coverage, restraint realism,
temperature projection between meshes/regions, through-thickness gradients, contact/tie behavior,
and temperature dependence of elastic properties. Validate the free-expansion limit
$\Delta L/L=\alpha\Delta T$ and the fully restrained one-dimensional limit
$\sigma\approx-E\alpha\Delta T$ where those assumptions apply.

See [heat transfer](./thermal), [structural analysis](./structural),
[structural constraints](../constraints/structural), and [current status](../status).
