# FEA-EM-NONLINEAR-001: Saturating iron

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## bh_curve_inputs

Evidence kind: RejectionInvariant. Criterion: Tabulated B-H data is validated monotone and dimensioned, evaluated with a monotone spline, and extrapolated to the vacuum slope above the last point; non-monotone or non-physical data is rejected.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| saturating_magnetics | edge_elements | numerical_kernel | OPEN |  |
| saturating_magnetics | edge_elements | public_study | OPEN |  |

## newton_with_line_search

Evidence kind: IndependentReference. Criterion: The nonlinear magnetic solve converges with Newton and line search, publishes iteration history, and fails typed when it stagnates.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| saturating_magnetics | edge_elements | numerical_kernel | OPEN |  |
| saturating_magnetics | edge_elements | public_study | OPEN |  |

## saturating_toroid_reference

Evidence kind: IndependentReference. Criterion: A saturating toroid reproduces the analytic B(H) relation (H = N I / l) within the declared tolerance in magnetostatic, harmonic effective-permeability and transient forms, and a current-sensor core near saturation reports flux through iron and gap.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| saturating_magnetics | edge_elements | numerical_kernel | OPEN |  |
| saturating_magnetics | edge_elements | public_study | OPEN |  |

