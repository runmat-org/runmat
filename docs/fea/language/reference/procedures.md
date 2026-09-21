# Procedure reference

An `analysis <name>` block owns exactly one procedure block. A dependent analysis uses
`analysis <name> after analysis.<producer>`. Required fields vary by procedure; the checker does
not invent domains or state inputs.

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
| `structural prestressed_harmonic` | retained `prestress` plus frequencies |
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

Language recognition is not an execution or maturity claim. The public typed `.fea` runner is
currently structural-only; family promotion requires its complete formulation, runner,
publication, local/cluster, validation, and release evidence.

## State flow

Procedures publish typed states. `retain <name>` names an unambiguous state output. Consumers bind
states explicitly, for example `basis = analysis.modes.state.basis`. The checker verifies state
kind, producer, topology, solution space, completeness, persistence, and transition method. `after`
alone supplies ordering and does not substitute for a required state binding.

## Domains

- Time domains require `0 <= start_time < end_time`; positive `time_step` must partition the range
  exactly.
- Frequency lists must be finite, nonnegative, nonempty, and strictly increasing.
- Modal counts are positive integers.
- Nonlinear load-factor ranges require finite increasing endpoints.

No procedure infers one of these domains from a signal, output request, or another analysis.
