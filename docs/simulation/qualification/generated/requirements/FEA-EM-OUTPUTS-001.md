# FEA-EM-OUTPUTS-001: Electromagnetic outputs and fields

0 PASS, 0 PARTIAL, 0 FAIL, 24 OPEN, 0 TAGGED. Generated; do not edit.

## terminal_quantities

Evidence kind: PhysicalBalance. Criterion: Voltage drop between named faces, terminal current, resistance and loss power publish and agree with V = I R for the busbar and with energy balances; a high-voltage electrode reports peak field and an assessment against its limit, sharp versus rounded.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | public_study | OPEN |  |
| electrostatic | tet10 | public_study | OPEN |  |
| electrostatic | edge_elements | public_study | OPEN |  |
| electrical_conduction | tet4 | public_study | OPEN |  |
| electrical_conduction | tet10 | public_study | OPEN |  |
| electrical_conduction | edge_elements | public_study | OPEN |  |
| magnetostatic | tet4 | public_study | OPEN |  |
| magnetostatic | tet10 | public_study | OPEN |  |
| magnetostatic | edge_elements | public_study | OPEN |  |
| harmonic_maxwell | tet4 | public_study | OPEN |  |
| harmonic_maxwell | tet10 | public_study | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

## vector_field_publication

Evidence kind: PublicationIntegrity. Criterion: Current density, magnetic flux density and electric field publish as element or node vector fields with magnitudes (and phasors for AC) in the form the viewer reads for cut planes, field lines, arrows and isosurfaces.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | public_study | OPEN |  |
| electrostatic | tet10 | public_study | OPEN |  |
| electrostatic | edge_elements | public_study | OPEN |  |
| electrical_conduction | tet4 | public_study | OPEN |  |
| electrical_conduction | tet10 | public_study | OPEN |  |
| electrical_conduction | edge_elements | public_study | OPEN |  |
| magnetostatic | tet4 | public_study | OPEN |  |
| magnetostatic | tet10 | public_study | OPEN |  |
| magnetostatic | edge_elements | public_study | OPEN |  |
| harmonic_maxwell | tet4 | public_study | OPEN |  |
| harmonic_maxwell | tet10 | public_study | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

