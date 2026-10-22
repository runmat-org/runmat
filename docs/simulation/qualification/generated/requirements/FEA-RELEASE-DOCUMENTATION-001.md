# FEA-RELEASE-DOCUMENTATION-001: Runnable exact public documentation and limitations

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## exact_physical_and_typed_contract_docs

Evidence kind: PublicationIntegrity. Criterion: Each formulation must document: supported equations and assumptions; supported element/cell types and order; material models and units; boundary conditions, loads, and sources; available typed constraints, relations, connectors, frames, schedules, enforcement behavior, reactions, and conflict diagnostics; typed signals, excitations, initial conditions, standard waveforms, generated-input callables...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## numerical_execution_and_limitations_docs

Evidence kind: PublicationIntegrity. Criterion: partition-local observation evaluation, global reduction attribution, bounded history streaming, and observation cost admission; adaptive behavior where supported; limitations and explicit rejection cases; runnable examples; and benchmark provenance. The central FEA documentation should connect geometry loading, declarative construction, immutable derivation, `.m` generation, AP242 parameter/prope...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## checked_language_guides_and_future_envelopes

Evidence kind: PublicationIntegrity. Criterion: Optimization, sensitivity analysis, uncertainty quantification, topology optimization, nonlinear post-buckling, arbitrary user elements, and executable constitutive laws must be documented as future envelopes rather than implied capabilities. Documentation may describe the typed dependency and extension seams, but it must not present capability metadata as an implemented solver feature.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

