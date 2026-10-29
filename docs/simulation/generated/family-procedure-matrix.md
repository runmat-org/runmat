---
description: Every physics family and analysis procedure RunMat FEA knows, with its solution space and whether the current release runs it or lists it as planned.
keywords: [family, procedure, analysis matrix, release status, supported analyses, FEA]
doc_type: release-status
generated: RUNMAT_BLESS_DOCS=1 cargo test -p runmat-analysis-core --lib committed_docs_matrix_matches_the_registry
---
# Families and procedures

<!-- Generated from runmat-analysis-core ANALYSIS_IDENTITIES; do not edit by hand. -->

Each row is one family and procedure pair from the closed registry that `runmat check`, `runmat run` and the qualification gate share. A planned pair links its pending id; `runmat check` reports the same id when a study uses it.

| Family | Procedure | Solution space | Status |
|---|---|---|---|
| structural | static | direct | runs |
| structural | modal | direct | runs |
| structural | prestressed_modal | direct | runs |
| structural | buckling | direct | runs |
| structural | transient | direct | runs |
| structural | harmonic | direct | runs |
| structural | prestressed_harmonic | direct | runs |
| structural | nonlinear | direct | runs |
| structural | transient | modal reduced | runs |
| structural | harmonic | modal reduced | runs |
| structural | prestressed_harmonic | modal reduced | runs |
| structural | response_spectrum | modal reduced | runs |
| structural | random_response | modal reduced | runs |
| thermal | static | direct | runs |
| thermal | transient | direct | runs |
| electromagnetic | static | direct | runs |
| electromagnetic | harmonic | direct | runs |
| electromagnetic | transient | direct | [not yet in the current release](pending:electromagnetic.transient) |
| electromagnetic | nonlinear | direct | [not yet in the current release](pending:electromagnetic.nonlinear) |
| acoustic | harmonic | direct | runs |
| acoustic | modal | direct | runs |
| acoustic | transient | direct | runs |
| acoustic | static | direct | [not yet in the current release](pending:acoustic.static) |
| acoustic | nonlinear | direct | [not yet in the current release](pending:acoustic.nonlinear) |
| fluid | static | direct | runs |
| fluid | transient | direct | runs |
| fluid | harmonic | direct | [not yet in the current release](pending:fluid.harmonic) |
| fluid | nonlinear | direct | [not yet in the current release](pending:fluid.nonlinear) |
| thermomechanical | static | direct | runs |
| thermomechanical | nonlinear | direct | runs |
| thermomechanical | transient | direct | [not yet in the current release](pending:thermomechanical.transient) |
| thermomechanical | harmonic | direct | [not yet in the current release](pending:thermomechanical.harmonic) |
| electrothermal | static | direct | runs |
| electrothermal | transient | direct | runs |
| electrothermal | harmonic | direct | [not yet in the current release](pending:electrothermal.harmonic) |
| electrothermal | nonlinear | direct | [not yet in the current release](pending:electrothermal.nonlinear) |
| conjugate_heat_transfer | static | direct | runs |
| conjugate_heat_transfer | transient | direct | runs |
| conjugate_heat_transfer | harmonic | direct | [not yet in the current release](pending:conjugate-heat-transfer.harmonic) |
| conjugate_heat_transfer | nonlinear | direct | [not yet in the current release](pending:conjugate-heat-transfer.nonlinear) |
| fluid_structure_interaction | static | direct | runs |
| fluid_structure_interaction | transient | direct | runs |
| fluid_structure_interaction | harmonic | direct | [not yet in the current release](pending:fluid-structure-interaction.harmonic) |
| fluid_structure_interaction | nonlinear | direct | [not yet in the current release](pending:fluid-structure-interaction.nonlinear) |
