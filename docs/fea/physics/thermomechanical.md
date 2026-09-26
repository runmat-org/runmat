# Thermomechanical FEA in RunMat: thermal expansion, stress, and coupled state

Thermomechanical analysis computes structural response caused by a temperature field, optionally in
combination with mechanical loading. RunMat treats temperature as typed state and thermal strain as
a constitutive contribution; it does not convert temperature into an arbitrary equivalent force.

The `.fea` language checks steady, transient, and nonlinear thermomechanical procedure contracts.
The numerical library admits experimental prescribed-temperature structural realizations and an
experimental **retained thermal state to static solid solve** path. Neither is bidirectionally
coupled, and the public typed `.fea` executor does not yet connect the retained-state path end to end.

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

The physical static Tet4/Tet10 branch integrates the thermal load and subtracts material-owned
thermal eigenstrain during stress/energy recovery. A prescribed field records
`OnePassPrescribedThermomechanicalProfile`; a retained solved nodal field records
`OneWaySolvedThermoelasticState`. The latter interpolates the retained temperatures at integration
points and bypasses reduced-profile stiffness/load adjustments. Other experimental prescribed
branches include reduced beam, shell, and explicitly synthetic profiles; those do not establish
the physical solid formulation's equations or qualification. Neither path solves heat in the
structural step or feeds deformation back into temperature.

## Which thermomechanical procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `thermomechanical steady` | Coupled static thermal/structural state | Prescribed profile, or retained solved temperature on Tet4/Tet10 through the Rust library |
| `thermomechanical transient` | Time samples and coupled evolving state | Not currently admitted as a coupled transient solve |
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
The `.fea` source-to-source workflow above remains a contract target; do not assume the two blocks
will execute as a coupled sequence yet.

The Rust numerical library provides the narrower executable path:

1. Publish a completed physical thermal result with `CanonicalSemanticState::from_thermal_result`.
2. Author a static thermomechanical consumer with a thermal `StateInput`, `Prestress` use, and a
   `Reuse` transition requiring identical topology and preserved reference temperature.
3. Call `run_linear_static_from_thermal_state` with the retained state and canonical solver mesh.

The result includes the thermal transition and a source-linked `SolverResultIdentity`. Missing
declared state, incompatible topology/reference, and an active prescribed temperature field that
conflicts with the retained source reject. Temperature/eigenstrain/eigenstress snapshots are
integration-point fields, not fabricated nodal fields. Recovery means deriving strain, stress,
and energy from retained solutions—not reconstructing lost data.

Thermal binding owns node alignment, material-capability validation, and source lineage. The
receiving structural assembly alone constructs equivalent loads and element recovery data.
The aligned global temperature copy is released after assembly; the immutable source and the
element-local temperatures needed for recovery are preserved. This avoids a second unused load
and recovery allocation. Static preflight additionally reserves source/alignment/load numerical
storage, local element temperatures and thermal recovery coefficients, and the 52 thermal output
values per solid element plus one coupling residual, and owned displacement/von-Mises snapshot
copies (including the temporary construction copy). These topology-derived charges participate
in memory-ceiling admission and resource-plan identity (estimation revision 25). They are not a
portable source artifact encoding/metadata bound or measured peak-memory qualification; transport
decoding and result serialization lifetimes still require separate accounting.

This path remains experimental. Evidence includes a solved-temperature-to-restrained Tet4
expansion reference with displacement, stress, reaction balance, and energy checks, plus separate
Tet4/Tet10 load and linear-temperature recovery references. Affine Tet10 quadratic-temperature
energy and an analytic curved-Tet10 map with thermal and mechanical strain are checked against
exact simplex moments. Volume averages use retained nodal temperatures, independently of the four
reported recovery samples. Curved thermal recovery compares 8-, 12-, and, when needed, 16-point
Gauss rules per Duffy coordinate and rejects unresolved integrals; this comparison is not a
certified error bound or general curved-element qualification. Spatial convergence, temperature-dependent constitutive iteration,
cross-mesh projection, portable execution/persistence, and downstream thermal-prestress procedures
are not qualified by these tests. Production admission remains closed.

## Why is my thermal stress wrong?

Check coefficient-of-expansion units, reference temperature, material coverage, restraint realism,
temperature projection between meshes/regions, through-thickness gradients, contact/tie behavior,
and temperature dependence of elastic properties. Validate the free-expansion limit
$\Delta L/L=\alpha\Delta T$ and the fully restrained one-dimensional limit
$\sigma\approx-E\alpha\Delta T$ where those assumptions apply.

See [heat transfer](./thermal), [structural analysis](./structural),
[structural constraints](../constraints/structural), and [current status](../status).
