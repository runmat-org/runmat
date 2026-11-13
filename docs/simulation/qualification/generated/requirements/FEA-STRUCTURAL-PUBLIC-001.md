# FEA-STRUCTURAL-PUBLIC-001: Structural capabilities the page claims

0 PASS, 3 PARTIAL, 0 FAIL, 97 OPEN, 0 TAGGED. Generated; do not edit.

## shells_and_beams_run

Evidence kind: IndependentReference. Criterion: A shell panel (MITC3, homogeneous and laminate layups with orthotropic plies) and a beam frame (general, rectangular, circular and tube sections) run from a `.fea` study on imported geometry and publish results; a plate matches the plate-bending formula, a frame matches beam theory, and a frame with code limits written as assessments returns verdicts.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |

## load_inventory_runs

Evidence kind: PhysicalBalance. Criterion: Every structural load the docs list runs and conserves its resultant: total, distributed and remote-point forces, moments and torque, traction, pressure (wind, mapped from a fluid result, hydrostatic), gravity and body force, inertial and centrifugal acceleration, base acceleration and earthquake motion, bolt preload, bearing loads on holes, thermal expansion, line loads, harmonic amplitudes, load...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |

## connection_inventory_runs

Evidence kind: PhysicalBalance. Criterion: Fixed supports, prescribed motion, couplings, ties, periodic constraints, equations, axial springs and dashpots, point springs to ground, elastic foundations, six-DOF bushings, rotational springs, nonlinear springs and gaps, frequency-dependent mounts and bonded interfaces run with reactions that balance the loads, and no weak spring, mass, damping, reference constraint or load case is added silen...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |

## large_deformation

Evidence kind: IndependentReference. Criterion: A `structural nonlinear` cantilever under a tip load to large rotation matches the elastica solution with Newton and load stepping.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| small_strain_j2 | tet4 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |

## plasticity_example

Evidence kind: IndependentReference. Criterion: A published J2 plasticity example (linear isotropic hardening, follower loads) runs and matches the manufactured or analytic reference including unloading.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| nonlinear_structural | tet4 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet4 | public_study | OPEN |  |
| nonlinear_structural | tet10 | numerical_kernel | OPEN |  |
| nonlinear_structural | tet10 | public_study | OPEN |  |
| small_strain_j2 | tet4 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet4 | public_study | OPEN |  |
| small_strain_j2 | tet10 | numerical_kernel | OPEN |  |
| small_strain_j2 | tet10 | public_study | OPEN |  |

## contact_example

Evidence kind: IndependentReference. Criterion: A published frictionless and Coulomb contact example (bolted bracket) runs from the public command and matches a Hertz or sliding reference.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| penalty_coulomb_contact | tet10 | public_study | OPEN |  |
| small_strain_j2_penalty_coulomb_contact | tet4 | numerical_kernel | OPEN |  |
| small_strain_j2_penalty_coulomb_contact | tet4 | public_study | OPEN |  |
| small_strain_j2_penalty_coulomb_contact | tet10 | numerical_kernel | OPEN |  |
| small_strain_j2_penalty_coulomb_contact | tet10 | public_study | OPEN |  |

## hand_calculation_references

Evidence kind: IndependentReference. Criterion: Cantilever tip deflection matches PL^3/(3EI) (0.199 against 0.200 mm), the first bending frequency matches Euler-Bernoulli (81.3 against 81.5 Hz), and a random-vibration study matches Miles' equation, with the coarse-mesh failure (0.1197 mm Tet4) flagged as cannot-tell rather than accepted.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |

## result_publication_defects

Evidence kind: PublicationIntegrity. Criterion: Explicit spatial observations without a canonical solver mesh, duplicate recovered semantic samples and missing buckling mode-shape fields are fixed: observations resolve, samples are unique, and a buckling run publishes its mode shape field.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-runtime[lib]::analysis::render_topology::tests::boundary_triangles_face_out_of_their_elements_whatever_the_stored_winding` (crates/runmat-runtime/src/analysis/render_topology/tests.rs:152) - every rendered boundary triangle faces out of its adjacent element whatever the stored face winding; shading and culling are not covered.<br>`runmat-runtime[lib]::analysis::storage::tests::persisted_run_keeps_typed_metadata_as_its_single_content_copy` (crates/runmat-runtime/src/analysis/storage/tests.rs:75) - a persisted run keeps its content once, as typed metadata (no untyped copy), and materializes back to the original fields; observation chunking is covered by pu... |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | numerical_kernel | PARTIAL | `runmat-runtime[lib]::analysis::field_scene::tests::dense_poster_scene_draws_boundary_triangles_and_feature_edges_only` (crates/runmat-runtime/src/analysis/field_scene/tests.rs:304) - a dense poster scene draws only its boundary triangles and, by default, feature edges (the cube's 12 creases) instead of the full mesh wireframe; rendering itse... |
| linear_structural | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | PARTIAL | `runmat-runtime[lib]::analysis::storage::field_reader::transient_sample_tests::time_and_index_samples_select_the_retained_transient_state_and_report_its_time` (crates/runmat-runtime/src/analysis/storage/field_reader.rs:401) - a time or one-based state sample resolves to the persisted transient snapshot the integrator retained and reports its exact time; the solve itself is not covere... |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |

## independent_structural_references

Evidence kind: IndependentReference. Criterion: Independent references exist and pass for a thick-walled cylinder under pressure (Lame), torsion of a prismatic bar, multi-point constraint and rigid-body coupling cases, and Euler column and plate buckling load factors, each with a mesh-convergence record.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |

## vibration_qualification

Evidence kind: IndependentReference. Criterion: A random-response study driven by a PSD (GEVS-type) from modal data reports RMS and 3-sigma stress, matching an independent modal combination.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| modal_structural | tet4 | numerical_kernel | OPEN |  |
| modal_structural | tet4 | public_study | OPEN |  |
| modal_structural | tet10 | numerical_kernel | OPEN |  |
| modal_structural | tet10 | public_study | OPEN |  |

