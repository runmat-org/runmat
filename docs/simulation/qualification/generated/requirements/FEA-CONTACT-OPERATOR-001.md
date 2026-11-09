# FEA-CONTACT-OPERATOR-001: Authored surface contact, search, friction and coupled inequality equilibrium

0 PASS, 11 PARTIAL, 0 FAIL, 69 OPEN, 0 TAGGED. Generated; do not edit.

## surface_search_gap

Evidence kind: IndependentReference. Criterion: Resolve authored primary and secondary surface identities through the shared topology projection. Build deterministic broad-phase and narrow-phase search. Compute oriented normals, closest projection, gap, quadrature, and active-set state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::surfaces::tests::canonical_contact_surfaces_bind_original_face_occurrences_and_current_geometry` (crates/runmat-analysis/fea/src/physics/structural/contact/surfaces/tests.rs:27) - Canonical contact surfaces bind original face occurrences and current geometry for both element orders. |
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::operator::tests::resources::global_duplicate_secondary_laws_and_conflicting_surface_roles_reject` (crates/runmat-analysis/fea/src/physics/structural/contact/operator/tests/resources.rs:106) - Duplicate secondary laws and conflicting surface roles are rejected on a Tet10 fixture; gap evaluation itself is covered in the penalty tests.<br>`runmat-analysis-fea[lib]::physics::structural::contact::surfaces::tests::canonical_contact_surfaces_bind_original_face_occurrences_and_current_geometry` (crates/runmat-analysis/fea/src/physics/structural/contact/surfaces/tests.rs:27) - Canonical contact surfaces bind original face occurrences and current geometry for both element orders. |
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
| penalty_coulomb_contact | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::pressure_unloading_changes_the_friction_radius_and_retains_energy_accounting` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:149) - At the point law, pressure unloading changes the friction radius and retains energy accounting; no assembled body.<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::frictionless_and_zero_coefficient_have_no_tangential_resistance` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:195) - Frictionless and zero-coefficient pairs have no tangential resistance at the point law.<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::contact::energy::opening_releases_stored_tangential_energy_without_inventing_friction_heat` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/energy.rs:65) - Opening releases stored tangential energy without inventing friction heat in a coupled solid solve for both orders. |
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::pressure_unloading_changes_the_friction_radius_and_retains_energy_accounting` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:149) - At the point law, pressure unloading changes the friction radius and retains energy accounting; no assembled body.<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::frictionless_and_zero_coefficient_have_no_tangential_resistance` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:195) - Frictionless and zero-coefficient pairs have no tangential resistance at the point law.<br>`runmat-analysis-fea[lib]::solve::nonlinear::solid::tests::contact::energy::opening_releases_stored_tangential_energy_without_inventing_friction_heat` (crates/runmat-analysis/fea/src/solve/nonlinear/solid/tests/contact/energy.rs:65) - Opening releases stored tangential energy without inventing friction heat in a coupled solid solve for both orders. |
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
| penalty_coulomb_contact | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::history::tests::original_point_frame_and_ordinal_cold_revalidation_reject_tampering` (crates/runmat-analysis/fea/src/physics/structural/contact/history/tests.rs:159) - Original point, frame and ordinal cold revalidation reject tampering for both orders; search diagnostics for initial penetration are in the operator resource te... |
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::history::tests::original_point_frame_and_ordinal_cold_revalidation_reject_tampering` (crates/runmat-analysis/fea/src/physics/structural/contact/history/tests.rs:159) - Original point, frame and ordinal cold revalidation reject tampering for both orders; search diagnostics for initial penetration are in the operator resource te... |
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
| penalty_coulomb_contact | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::history::tests::tangent::complete_stick_and_slip_tangents_include_frame_terms_and_match_residual_changes` (crates/runmat-analysis/fea/src/physics/structural/contact/history/tests/tangent.rs:7) - Complete stick and slip tangents include frame terms and match residual changes for both element orders.<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::tangent::local_consistent_tangent_matches_two_step_sizes_in_all_smooth_regimes` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests/tangent.rs:7) - The local consistent tangent matches two finite-difference step sizes in all smooth regimes of the point law. |
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::history::tests::tangent::complete_stick_and_slip_tangents_include_frame_terms_and_match_residual_changes` (crates/runmat-analysis/fea/src/physics/structural/contact/history/tests/tangent.rs:7) - Complete stick and slip tangents include frame terms and match residual changes for both element orders.<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::tangent::local_consistent_tangent_matches_two_step_sizes_in_all_smooth_regimes` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests/tangent.rs:7) - The local consistent tangent matches two finite-difference step sizes in all smooth regimes of the point law. |
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
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::operator::tests::resources::explicit_operator_controls_and_initial_penetration_limits_reject_instead_of_defaulting` (crates/runmat-analysis/fea/src/physics/structural/contact/operator/tests/resources.rs:38) - Explicit operator controls and initial-penetration limits are rejected instead of defaulted on a Tet10 fixture; complementarity residuals are not asserted. |
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
| penalty_coulomb_contact | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::operator::tests::reference::disjoint_canonical_bodies_under_prescribed_compression_match_closed_penalty_force_energy_and_tangent` (crates/runmat-analysis/fea/src/physics/structural/contact/operator/tests/reference.rs:7) - Two disjoint canonical bodies under prescribed compression match the closed penalty force, energy and tangent for both orders; Hertz-type curved contact is not ...<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::objectivity::material_attached_frame_and_point_traction_are_objective_under_proper_rotation` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests/objectivity.rs:8) - Tri3 contact frame and penalty friction point-law proper-rotation objectivity; no global Hertz, sliding body, action-reaction or mesh/penalty convergence credit...<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::analytic_sliding_unloading_reversal_and_separation_keep_original_history` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:32) - Analytic sliding, unloading, reversal and separation keep the original history at the point law. |
| penalty_coulomb_contact | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::contact::operator::tests::reference::disjoint_canonical_bodies_under_prescribed_compression_match_closed_penalty_force_energy_and_tangent` (crates/runmat-analysis/fea/src/physics/structural/contact/operator/tests/reference.rs:7) - Two disjoint canonical bodies under prescribed compression match the closed penalty force, energy and tangent for both orders; Hertz-type curved contact is not ...<br>`runmat-analysis-fea[lib]::physics::structural::contact::penalty::tests::analytic_sliding_unloading_reversal_and_separation_keep_original_history` (crates/runmat-analysis/fea/src/physics/structural/contact/penalty/tests.rs:32) - Analytic sliding, unloading, reversal and separation keep the original history at the point law. |
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

