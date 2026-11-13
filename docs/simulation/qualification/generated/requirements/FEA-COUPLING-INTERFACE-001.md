# FEA-COUPLING-INTERFACE-001: Typed physical interfaces and conservative transfers

0 PASS, 0 PARTIAL, 0 FAIL, 12 OPEN, 0 TAGGED. Generated; do not edit.

## identity

Evidence kind: SemanticRealization. Criterion: Resolve exact interface sides with original geometry/revision and deterministic mapping; CHT declares exactly one perfect-transfer, conductance or resistance law; FSI declares physical kinematics. Do not infer terminals or interfaces from coordinate extrema.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | canonical_mixed_topology | public_study | OPEN |  |
| electrothermal | canonical_mixed_topology | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |

## transfer

Evidence kind: PhysicalBalance. Criterion: Validate matching or nonmatching transfer maps, conservation, error, frame/orientation, units, source/target lineage and state compatibility before consumption. Preserve each side conservation law and original interface evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | canonical_mixed_topology | public_study | OPEN |  |
| electrothermal | canonical_mixed_topology | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |

## seams

Evidence kind: RejectionInvariant. Criterion: Two-way policies, executable constitutive laws and future derivative/optimization capability remain explicitly bounded; capability metadata cannot imply implemented coupling, gradients or user-law callbacks inside assembly, recovery, iterative solves or distributed ranks.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | canonical_mixed_topology | public_study | OPEN |  |
| electrothermal | canonical_mixed_topology | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_mixed_topology | public_study | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |

