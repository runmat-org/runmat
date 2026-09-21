# Physics Families And Procedures

The procedure declaration states the physical question. Its typed contract determines required material capabilities, conditions, excitation quantities, independent domain, state inputs/outputs, admissible fields, numerical controls, and formulation evidence. Execution policy chooses eligible placement and resources; it does not change the physics.

## Structural Procedures

The typed language and native structural whole-study path cover these procedure shapes:

| Procedure | Primary intent | Important explicit inputs/state |
| --- | --- | --- |
| `structural static` | linear static response | load case, constraints, accuracy |
| `structural preload` | retained prestress state | load case and retained state |
| `structural modal` | eigenmodes | requested mode count; may retain a modal basis |
| `structural transient` | direct time response | start, end, time step, signals/initial state |
| `structural modal_transient` | reduced time response | retained modal basis and time domain |
| `structural harmonic` | direct frequency response | frequency samples and damping |
| `structural modal_harmonic` | reduced frequency response | retained modal basis and frequencies |
| `structural prestressed_modal` | modes about a prestressed state | retained prestress and mode count |
| `structural buckling` | linearized buckling modes | retained prestress and mode count |
| `structural prestressed_harmonic` | frequency response about prestress | prestress plus frequencies |
| `structural modal_response_spectrum` | combined modal response spectrum | modal basis plus named model spectra |
| `structural random_response` | statistical response from CPSD input | modal basis, named random excitation, damping/mass truncation |
| `structural nonlinear` | load-factor nonlinear response | explicit load-factor domain and nonlinear formulation |

Not every formulation/backend/scale combination in this table is production-qualified. A study must declare `require formulation = production` or `experimental`; admission and the result evidence are authoritative.

## Other Typed Family Contracts

The language also checks thermal, acoustic, electromagnetic, fluid, thermo-mechanical, electro-thermal, conjugate heat-transfer, and fluid-structure procedure contracts. These declarations provide strong authoring types and allow the remaining solvers to converge on one model, constraint, excitation, state, output, and execution vocabulary.

The current native typed `.fea` executor rejects a graph containing a non-structural procedure. Some lower-level runtime object operations and numerical components for those families exist, but they are not a complete typed-source geometry-to-publication path and must not be treated as one. See [Current status](./status).

## State Flow Is Explicit

```fea
analysis preload {
    structural preload {}
    retain prestress
}
analysis modes after analysis.preload {
    structural prestressed_modal {
        prestress = analysis.preload.state.prestress
        modes = 20
    }
    retain basis
}
```

Procedure order alone does not invent a data dependency. A consumer names the exact retained state. The compiler checks family, state type, producer, ordering, and compatibility before execution.

## Model Requirements

The selected procedures drive complete model admission:

- static structures need compatible elasticity and constraint/load coverage;
- dynamics additionally need density, damping policy where applicable, and a physical independent domain;
- modal reductions require a validated operator-bound basis;
- spectrum/random response require named typed environmental inputs and truncation/combination policy;
- thermal procedures require thermal capabilities and temperature/flux/source conditions;
- electromagnetic, acoustic, and fluid procedures require their family-specific material, gauge/reference, source, boundary, and interface data.

Missing data is an error. The compiler or solver cannot choose a first material, average several candidates, fix an arbitrary degree of freedom, invent damping, or infer a reference boundary.

## Results

Each completed step publishes the exact resolved workload, numerical receipt, semantic state, and only the fields admitted for that procedure. Authored output expressions select and transform those typed fields. Multi-step results retain step identity; queries must select a step when an aggregate is ambiguous.

Use [Study contracts](./study-contracts) for observations, accuracy, restart, and state semantics, and [Verification and validation](./validation) for qualification requirements.
