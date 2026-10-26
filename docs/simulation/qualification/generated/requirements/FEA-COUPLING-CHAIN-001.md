# FEA-COUPLING-CHAIN-001: Chained thermal-structural studies

0 PASS, 0 PARTIAL, 0 FAIL, 24 OPEN, 0 TAGGED. Generated; do not edit.

## thermal_to_prestress_to_modal

Evidence kind: PublicRun. Criterion: A study runs thermal, then preload, then buckling or prestressed modal in sequence on one mesh without the mixed-sequence rejection, with the retained states typed and checked.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | numerical_kernel | OPEN |  |
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | numerical_kernel | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |

## thermomechanical_references

Evidence kind: IndependentReference. Criterion: A uniformly heated constrained bar (80 K rise) gives axial stress -E alpha deltaT (-192 MPa for the page's bar); the first bending frequency of a bar heated 200 K over unheated is 0.963 within the declared tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | numerical_kernel | OPEN |  |
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | numerical_kernel | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |

