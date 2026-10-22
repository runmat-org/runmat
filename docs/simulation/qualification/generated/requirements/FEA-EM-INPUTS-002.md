# FEA-EM-INPUTS-002: Every declared electromagnetic input runs

0 PASS, 0 FAIL, 12 OPEN, 0 TAGGED. Generated; do not edit.

## scalar_potential_inputs

Evidence kind: PhysicalBalance. Criterion: terminal_voltage, electric current terminals, electric_charge, equipotential, floating and integral-current constraints run for electrostatics and steady conduction, with total current and charge balances closed.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | numerical_kernel | OPEN |  |
| electrostatic | tet4 | public_study | OPEN |  |
| electrostatic | tet10 | numerical_kernel | OPEN |  |
| electrostatic | tet10 | public_study | OPEN |  |
| electrical_conduction | tet4 | numerical_kernel | OPEN |  |
| electrical_conduction | tet4 | public_study | OPEN |  |
| electrical_conduction | tet10 | numerical_kernel | OPEN |  |
| electrical_conduction | tet10 | public_study | OPEN |  |

## magnetic_inputs

Evidence kind: SemanticRealization. Criterion: current_density, magnetic_vector_potential, magnetic.insulation and periodic constraints run for magnetostatics and harmonic Maxwell with typed rank-correct constraints.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| magnetostatic | edge_elements | numerical_kernel | OPEN |  |
| magnetostatic | edge_elements | public_study | OPEN |  |
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

