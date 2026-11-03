# Procedures

An `analysis <name>` block owns exactly one procedure block. A dependent analysis uses
`analysis <name> after analysis.<producer>`. Required fields vary by procedure; the checker does
not invent domains or state inputs.

## Algebraic accuracy

An optional analysis-level block authors the existing typed relative algebraic
criterion, independently of physics family:

```fea
analysis potential {
    electromagnetic static {}
    accuracy { algebraic_relative_tolerance = 1e-11 }
}
```

The tolerance must be a finite dimensionless constant strictly between zero and
one. The block accepts exactly this one assignment; duplicate blocks, duplicate
or unknown settings, nested settings and dimensional values are rejected.
Omission preserves the procedure's baseline accuracy plan. Explicit authoring
changes only its relative algebraic criteria, preserving evidence requirements,
mandatory status, other criteria and resource budgets. It does not certify
field error or spatial convergence. Execution must honor
the resulting typed accuracy contract; a small residual alone is not a continuum
error bound. Authoring tests verify preservation across structural, thermal,
acoustic and electrical procedures, and reject malformed or ambiguous settings.

## Structural procedures

| Procedure block | Required authored data |
| --- | --- |
| `structural static` | compatible model, constraints, and applicable static loads |
| `structural preload` | static preload definition; publishes prestress state |
| `structural modal` | positive `modes`; publishes modal basis state |
| `structural transient` | `start_time`, `end_time`, `time_step` |
| `structural modal_transient` | retained `basis` plus time domain |
| `structural harmonic` | strictly increasing `frequencies` |
| `structural modal_harmonic` | retained `basis` plus frequencies |
| `structural prestressed_modal` | retained `prestress` plus mode count |
| `structural buckling` | retained `prestress` plus mode count |
| `structural prestressed_harmonic` | retained `prestress` plus frequencies; optional retained `basis` selects modal reduction |
| `structural modal_response_spectrum` | retained `basis` and one or more model spectra |
| `structural random_response` | retained `basis`, random excitation, frequency domain, and accuracy controls |
| `structural nonlinear` | load-factor domain and nonlinear controls |

## Thermal procedures

`thermal steady` uses a static independent domain. `thermal transient` requires the same explicit
time range and exact step partition as structural transient analysis.

## Other typed families

The semantic model recognizes `electromagnetic`, `acoustic`, `fluid`, `thermomechanical`,
`electrothermal`, `conjugate_heat_transfer`, and `fluid_structure_interaction`. Their generic
procedure forms are `static` or family-appropriate `steady`, `transient`, `harmonic`, acoustic
`modal`, and `nonlinear` where the family admits it. Each form lowers to a typed physics family,
procedure, independent domain, solution space, and state kind.

Language recognition is not an execution or validation claim. The typed structural-study runner
also supports an explicit thermal-state prefix feeding thermomechanical preload and supported
structural consumers. This does not promote every recognized coupled family: promotion requires
complete formulation, runner, publication, local/cluster, validation, and release evidence.

## State flow

Procedures publish typed states. `retain <name>` names an unambiguous state output. Consumers bind
states explicitly, for example `basis = analysis.modes.state.basis`. The checker verifies state
kind, producer, topology, solution space, completeness, persistence, and transition method. `after`
alone supplies ordering and does not substitute for a required state binding.

A prestressed harmonic response can consume both states explicitly:

```fea
analysis response after analysis.modes {
    structural prestressed_harmonic {
        prestress = analysis.preload.state.geometric_stiffness
        basis = analysis.modes.state.modal_basis
        frequencies = [60 Hz]
    }
}
```

Without `basis`, this procedure uses a direct solve. With `basis`, it uses modal reduction.
For multiple state producers, `after` names one consumed producer; every explicit state
binding independently creates a required dependency and transition. It does not discard
the other producer or permit its state to be unavailable.

## Load cases

Inside a procedure, `load_cases = [load_case.service]` selects exactly the named load cases
for that step. `load_cases = []` selects no load cases; omitting the property leaves the
model's cases eligible under their existing activation rules. Selection does not disable
supports, prescribed temperatures, or other boundary conditions. Unknown and duplicate
case names are rejected. Use explicit empty selection on a heat/preload step when a
mechanical excitation is intended only for a later harmonic response.

## Domains

- Time domains require `0 <= start_time < end_time`; positive `time_step` must partition the range
  exactly.
- Frequency lists must be finite, nonnegative, nonempty, and strictly increasing.
- Modal counts are positive integers.
- Nonlinear load-factor ranges require finite increasing endpoints.

No procedure infers one of these domains from a signal, output request, or another analysis.

## Examples

A single-step static procedure has no independent-domain fields:

```fea
analysis solve {
    structural static {}
}
```

Transient procedures state their complete time domain:

```fea
analysis heat_up {
    thermal transient {
        start_time = 0 s
        end_time = 60 s
        time_step = 0.5 s
    }
}
```

Dependent procedures bind the state they consume instead of relying on block order:

```fea
analysis modes {
    structural modal { modes = 24 }
    retain basis
}

analysis spectrum after analysis.modes {
    structural modal_response_spectrum {
        basis = analysis.modes.state.basis
        spectra = [model.spectrum.design]
    }
}
```
