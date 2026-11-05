---
description: Every .fea analysis procedure by physics family (static, modal, transient, harmonic, buckling, response spectrum, random response, nonlinear) with fields, domains and state flow.
keywords: [analysis procedures, structural modal, transient analysis, harmonic response, buckling, response spectrum, random vibration, nonlinear analysis, state transfer]
doc_type: language-reference
examples: [cantilever-static, tuning-fork-modal]
---
# Procedures

An analysis step is an `analysis <name> { … }` block in a study. It holds exactly one procedure block, `<family> <procedure> { <fields> }`, plus optional `retain`, `accuracy`, `load_cases` and `initial_conditions` members. Steps run in dependency order; `after analysis.<step>` orders a step after another, and a state binding (`basis = analysis.modes.state.basis`) makes one step consume another's result.

```fea fragment
analysis modes {
    structural modal { modes = 12 }
    retain basis
}

analysis response after analysis.modes {
    structural modal_harmonic {
        basis = analysis.modes.state.basis
        frequencies = [50 Hz, 100 Hz, 150 Hz]
    }
}
```

Which procedures each family runs is generated from the solver's procedure registry in the [family and procedure matrix](../../validation/status.md#families-and-procedures). `runmat check` applies the same registry, so a combination the family does not run is reported at its line before anything is meshed.

## What each procedure solves

With $\mathbf K$, $\mathbf M$ and $\mathbf C_d$ the assembled stiffness, mass and damping matrices and $\mathbf f$ the load vector:

| Procedure | Equation | Independent domain |
| --- | --- | --- |
| `static` (`steady`) | $\mathbf K \mathbf u = \mathbf f$ | none |
| `modal` | $(\mathbf K - \omega_i^2 \mathbf M)\,\boldsymbol\phi_i = \mathbf 0$, for the lowest `modes` pairs | mode count |
| `transient` | $\mathbf M \ddot{\mathbf u} + \mathbf C_d \dot{\mathbf u} + \mathbf K \mathbf u = \mathbf f(t)$ (first order, $\mathbf C \dot{\mathbf T} + \mathbf K \mathbf T = \mathbf q(t)$, for heat transfer) | time |
| `harmonic` | $(\mathbf K + i\omega \mathbf C_d - \omega^2 \mathbf M)\,\hat{\mathbf u} = \hat{\mathbf f}(\omega)$ | frequency list |
| `buckling` | $(\mathbf K + \lambda_i \mathbf K_\sigma)\,\boldsymbol\phi_i = \mathbf 0$, with $\mathbf K_\sigma$ the geometric stiffness of a preload | mode count |
| `nonlinear` | $\mathbf r(\mathbf u, \lambda) = \mathbf f_\text{int}(\mathbf u) - \lambda \mathbf f = \mathbf 0$, solved by Newton iteration over the load factor $\lambda$ | load factor |
| modal-reduced forms | the same equations projected onto a retained modal basis, $\mathbf u \approx \boldsymbol\Phi \mathbf q$ | as above |

Each family's [physics reference](../../physics/index.md) states its weak form, element matrices, solver and recovery.

## Structural procedures

| Procedure block | Fields | Publishes |
| --- | --- | --- |
| `structural static` | none | `primary` (displacement state) |
| `structural preload` | none; the step's loads are the preload | `geometric_stiffness` |
| `structural modal` | `modes` (positive integer); optional `residual_vectors` (one to three global directions, `[[0, 0, 1]]`) | `modal_basis` |
| `structural transient` | `start_time`, `end_time`, `time_step` | `primary` |
| `structural modal_transient` | `basis`, `start_time`, `end_time`, `time_step` | `primary` |
| `structural harmonic` | `frequencies` | `primary` |
| `structural modal_harmonic` | `basis`, `frequencies`; optional `static_correction = applied_loads` or `none` (default) | `primary` |
| `structural prestressed_modal` | `prestress`, `modes`; optional `residual_vectors` | `modal_basis` |
| `structural buckling` | `prestress`, `modes` | buckling modes |
| `structural prestressed_harmonic` | `prestress`, `frequencies`; optional `basis` (modal reduction instead of a direct solve) and, with `basis`, `static_correction` | `primary` |
| `structural modal_response_spectrum` | `basis`, `spectra = [model.response_spectrum.<name>, …]` | `primary` |
| `structural random_response` | `basis`, `excitation = model.random_excitation.<name>`, `modal_damping_ratio` (number), `minimum_cumulative_mass_ratio` (number) | `primary` |
| `structural nonlinear` | `start_load_factor`, `end_load_factor`; optional `kinematics = small_strain` (default) or `large_deformation` | `primary` |

### Modal truncation and residual vectors

A modal-reduced step solves in the space its basis spans: the retained modes, then any residual vectors, then any static correction. RunMat accepts the step when the reduced equations meet the step's relative accuracy tolerance. It then evaluates the full-order equilibrium of the expanded response and publishes the largest relative imbalance as the modal truncation measure (`modal_truncation_balance_error` in `fea.results`, with the frequency where it occurs). That measure reports how much response the reduced space leaves out. It is published for every modal-reduced sweep and is not compared with the algebraic tolerance. A direct solve is accepted on its full-order balance and reports it as `max_frequency_balance_error`.

Two enrichments shrink the truncation measure:

- `residual_vectors = [[0, 0, 1], …]` on `structural modal` or `structural prestressed_modal` adds, for each global direction, the static response to a unit acceleration in that direction with the retained modes removed. Use it for base excitation and inertia loading.
- `static_correction = applied_loads` on a modal-reduced frequency step adds the static response to the step's applied load (real and imaginary parts at the sweep's reference frequency), with the retained shapes removed. The reduced space then contains the exact quasi-static response, so the truncation measure falls sharply below the first retained mode. Use it for concentrated or surface loads.

```fea fragment
analysis modes {
    structural modal { modes = 8 }
    retain basis
}

analysis response after analysis.modes {
    structural modal_harmonic {
        basis = analysis.modes.state.basis
        frequencies = [20, 40, 60] Hz
        static_correction = applied_loads
    }
}
```

`structural nonlinear` runs as a study's single step, and inside a procedure graph (after a preload, before a modal step) [not yet in the current release](pending:structural.nonlinear.sequence).

## Thermal procedures

| Procedure block | Fields | Publishes |
| --- | --- | --- |
| `thermal steady` | optional `heat_source = analysis.<em-step>.state.<s>` | `primary` (temperature state) |
| `thermal transient` | `start_time`, `end_time`, `time_step`; optional `heat_source` | `primary` |

`heat_source` heats the thermal step with the Joule loss density of an upstream `electromagnetic harmonic` step (induction heating).

## Electromagnetic procedures

| Procedure block | Fields |
| --- | --- |
| `electromagnetic static` | none; the model's `electromagnetic { equation = … }` selects electrostatics, DC conduction or magnetostatics |
| `electromagnetic harmonic` | `frequencies` |
| `electromagnetic transient` [not yet in the current release](pending:electromagnetic.transient) | `start_time`, `end_time`, `time_step` |
| `electromagnetic nonlinear` [not yet in the current release](pending:electromagnetic.nonlinear) | `start_load_factor`, `end_load_factor` |

## Acoustic procedures

| Procedure block | Fields |
| --- | --- |
| `acoustic harmonic` | `frequencies` |
| `acoustic modal` | `modes` |
| `acoustic transient` | `start_time`, `end_time`, `time_step` |
| `acoustic static` [not yet in the current release](pending:acoustic.static) | none |
| `acoustic nonlinear` [not yet in the current release](pending:acoustic.nonlinear) | `start_load_factor`, `end_load_factor` |

## Fluid procedures

| Procedure block | Fields |
| --- | --- |
| `fluid steady` | none |
| `fluid transient` | `start_time`, `end_time`, `time_step` |
| `fluid harmonic` [not yet in the current release](pending:fluid.harmonic) | `frequencies` |
| `fluid nonlinear` [not yet in the current release](pending:fluid.nonlinear) | `start_load_factor`, `end_load_factor` |

## Coupled procedures

| Procedure block | Fields |
| --- | --- |
| `thermomechanical preload` | `temperature = analysis.<thermal-step>.state.<s>`; publishes `geometric_stiffness` for prestressed modal, buckling and harmonic steps |
| `thermomechanical steady` | none |
| `thermomechanical nonlinear` | `start_load_factor`, `end_load_factor` |
| `thermomechanical transient` [not yet in the current release](pending:thermomechanical.transient) | `start_time`, `end_time`, `time_step` |
| `thermomechanical harmonic` [not yet in the current release](pending:thermomechanical.harmonic) | `frequencies` |
| `electrothermal steady`, `electrothermal transient` | none; time fields for `transient` |
| `electrothermal harmonic` [not yet in the current release](pending:electrothermal.harmonic) | `frequencies` |
| `electrothermal nonlinear` [not yet in the current release](pending:electrothermal.nonlinear) | `start_load_factor`, `end_load_factor` |
| `conjugate_heat_transfer steady`, `conjugate_heat_transfer transient` | none; time fields for `transient` |
| `conjugate_heat_transfer harmonic` [not yet in the current release](pending:conjugate-heat-transfer.harmonic) | `frequencies` |
| `conjugate_heat_transfer nonlinear` [not yet in the current release](pending:conjugate-heat-transfer.nonlinear) | `start_load_factor`, `end_load_factor` |
| `fluid_structure_interaction steady`, `fluid_structure_interaction transient` | none; time fields for `transient` |
| `fluid_structure_interaction harmonic` [not yet in the current release](pending:fluid-structure-interaction.harmonic) | `frequencies` |
| `fluid_structure_interaction nonlinear` [not yet in the current release](pending:fluid-structure-interaction.nonlinear) | `start_load_factor`, `end_load_factor` |

`static` and `steady` are the same procedure; use whichever reads naturally for the family.

## Domains

No procedure infers its domain from a signal, an output or another step; each states it:

| Field | Type | Rule |
| --- | --- | --- |
| `start_time`, `end_time` | time | $0 \le t_\text{start} < t_\text{end}$. |
| `time_step` | time | Positive, and divides $t_\text{end} - t_\text{start}$ exactly. |
| `frequencies` | list of frequencies | Finite, nonnegative, nonempty and strictly increasing, such as `[10 Hz, 20 Hz, 50 Hz]`. |
| `modes` | positive integer | Number of eigenpairs, from the lowest. |
| `start_load_factor`, `end_load_factor` | numbers | Finite and increasing; the load is scaled by the factor. |

## State flow

Each step publishes typed states: a displacement or temperature state (`primary`), a modal basis (`modal_basis`), a preload's geometric stiffness (`geometric_stiffness`). `retain <name>` publishes a step's state under a chosen name, so `retain basis` after `structural modal` publishes `analysis.<step>.state.basis`.

A consuming step names the state it needs:

```fea fragment
analysis preload { structural preload {} }

analysis modes after analysis.preload {
    structural prestressed_modal {
        prestress = analysis.preload.state.geometric_stiffness
        modes = 6
    }
    retain basis
}

analysis response after analysis.modes {
    structural prestressed_harmonic {
        prestress = analysis.preload.state.geometric_stiffness
        basis = analysis.modes.state.basis
        frequencies = [60 Hz]
    }
}
```

The checker verifies each binding: the producer exists, the state kind matches, the producer runs on the same mesh and solution space, and the state is complete and persisted. `after` orders steps; it does not replace a state binding. Every state binding adds its own dependency, so a step that consumes two producers waits for both.

## Members of every analysis step

| Member | Form | Meaning |
| --- | --- | --- |
| `retain <name>` | statement | Publishes the step's state under `<name>`. |
| `load_cases = [load_case.<a>, …]` | list | The load cases this step applies. `[]` applies none; omitting it applies the model's load cases. Supports and prescribed values always apply. |
| `initial_conditions = [model.initial.<a>, …]` | list | Initial state for a step that starts from one, such as a transient step. See [initial conditions](../../constraints/initial-conditions.md). |
| `accuracy { algebraic_relative_tolerance = <number> }` | block | The relative algebraic tolerance of the step's linear and nonlinear solves, strictly between 0 and 1. It tightens or loosens solver stopping only; it is not a bound on discretization error. |

```fea fragment
analysis heat_up {
    thermal transient {
        start_time = 0 s
        end_time = 60 s
        time_step = 0.5 s
    }
    load_cases = [load_case.heater_on]
    initial_conditions = [model.initial.ambient]
    accuracy { algebraic_relative_tolerance = 1e-8 }
}
```

## Response spectrum and random response

```fea source=modal-response-spectrum.fea
    analysis modes {
        structural modal { modes = 30 }
        retain basis
    }
    analysis response_spectrum after analysis.modes {
        structural modal_response_spectrum {
            basis = analysis.modes.state.basis
            spectra = [model.response_spectrum.qualification_x]
        }
    }
```

The spectrum is a model declaration (`response_spectrum <name> { … }`), and a random excitation a `random_excitation <name> { … }` power spectral density; both are referenced from the step. The full studies are [`modal-response-spectrum.fea`](../../examples/modal-response-spectrum.fea) and [`modal-random-response.fea`](../../examples/modal-random-response.fea).
