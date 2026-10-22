# FEA-CONTACT-OPERATOR-001: Authored surface contact, search, friction and coupled inequality equilibrium

1 PASS, 0 FAIL, 79 OPEN, 0 TAGGED. Generated; do not edit.

## surface_search_gap

Evidence kind: IndependentReference. Criterion: Resolve authored primary and secondary surface identities through the shared topology projection. Build deterministic broad-phase and narrow-phase search. Compute oriented normals, closest projection, gap, quadrature, and active-set state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## normal_friction_equilibrium

Evidence kind: IndependentReference. Criterion: Implement a documented normal enforcement method and friction law. Assemble contact residual and tangent into nonlinear equilibrium. Update contact state as geometry changes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## history_search_diagnostics

Evidence kind: IndependentReference. Criterion: Preserve contact history across restart and supported adaptation. Diagnose initial penetration, unmatched surfaces, normal ambiguity, search exhaustion, and parameter sensitivity. The contact system owns deterministic surface pairing and search updates.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## gap_active_enforcement

Evidence kind: IndependentReference. Criterion: The contact system owns gap and normal definitions, active-set or complementarity state, and normal enforcement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## friction_tangent_history

Evidence kind: IndependentReference. Criterion: The contact system owns tangential/friction state, residual and tangent contributions, and history/restart data.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## complementarity_penetration

Evidence kind: IndependentReference. Criterion: The contact system owns complementarity and penetration evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## references

Evidence kind: IndependentReference. Criterion: Evidence includes gap closure, patch contact, Hertz-type normal contact, sliding/friction, objectivity, action-reaction balance, and mesh/contact-parameter convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::objectivity::material_attached_frame_and_point_traction_are_objective_under_proper_rotation` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests/objectivity.rs:8) - Tri3 contact frame and penalty friction point-law proper-rotation objectivity; no global Hertz, sliding body, action-reaction or mesh/penalty convergence credit... |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## inequality

Evidence kind: RejectionInvariant. Criterion: Authored contact interfaces compile to nonlinear inequalities, never the equality matrix. Diagnose tied/contacting surface overlap without erasing their mathematical distinction. Stable authored identities survive changing geometry while normals, pairings and active state update in the owning nonlinear formulation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | native_process | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | durable | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_mixed_topology | claimed_accelerator | OPEN |  |

