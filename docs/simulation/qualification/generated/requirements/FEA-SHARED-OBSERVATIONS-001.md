# FEA-SHARED-OBSERVATIONS-001: Typed field recovery, bounded observations and monitors

0 PASS, 3 PARTIAL, 0 FAIL, 45 OPEN, 0 TAGGED. Generated; do not edit.

## typed_output_request_kinds

Evidence kind: SemanticRealization. Criterion: Results are requested through a typed observation system. It prevents accidental production of enormous result sets, supports useful answers without full-field transfer, and keeps result selection separate from visualization. Calling `plot` or `mesh` never decides what physical quantity to recover. The authored output contract distinguishes: `FieldOutput`: a spatial field snapshot or governed seri...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## quantity_location_and_cost_contract

Evidence kind: SemanticRealization. Criterion: precision, compression, retention, and partial-access policy; whether it is required for solver control, publication, visualization, or downstream sequence state; and an estimate of computation, resident memory, stored bytes, transfer, and collective cost. Quantity and reduction compatibility is validated before solve. A tensor invariant is not a string expression, a complex field cannot silently ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-core[lib]::study::authored_observation::tests::point_interpolation_of_a_cell_field_samples_the_containing_cell` (crates/runmat-analysis/core/src/study/authored_observation.rs:532) - interpolate(...) of a finite-volume cell field at one point lowers to the containing cell's native value and passes authored validation; path sampling of cell f...<br>`runmat-analysis-fea[lib]::results::observation::topology::tests::sampled_observation_topology_retains_each_original_rule_order_and_identity` (crates/runmat-analysis/fea/src/results/observation/topology/tests.rs:12) - Tet4/Tet10 sampled topology retains original four/fifteen-point rule identities, positions, edge IDs and integration indices and rejects mismatched topology/cou... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## typed_derived_operation_catalog

Evidence kind: SemanticRealization. Criterion: Field recovery belongs to the producing physics family. Observation evaluation consumes those governed fields or explicitly documented solver-available quantities. It may compose typed operations but cannot recreate family equations or reach into private solver storage. Derived quantities use a typed expression or operation graph over admitted physical quantities. It checks dimensions, shapes, top...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## attributed_extrema_and_small_defaults

Evidence kind: SemanticRealization. Criterion: Each overload declares typed inputs/named arguments/output, quantity and unit transformation, shape, scalar/complex/statistical domain, field location/frame/coordinate rules, purity and effects, distributability, reduction ordering and tie policy, equivalence, cost shape, frontend availability, lowering owner, and documentation. The catalog drives parser/LSP completion, hover, signature help, name...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## bounded_partitioned_evaluation

Evidence kind: SemanticRealization. Criterion: Large field outputs are partitioned and chunked according to canonical ownership. Point probes route to the owning partition. Spatial reductions compute partition-local contributions and use deterministic or tolerance-governed collectives. Global extrema carry location and attribution through the reduction. Surface and volume integrals use the same quadrature, measures, orientation, and topology i...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::results::observation::partition_tests::partitioned_boundary_reduction_matches_single_partition_evaluation` (crates/runmat-analysis/fea/src/results/observation/partition_tests.rs:491) - Partitioned boundary reduction matches single-partition evaluation.<br>`runmat-analysis-fea[lib]::results::observation::tests::invalid_and_unbounded_requests_fail_before_field_evaluation` (crates/runmat-analysis/fea/src/results/observation/tests.rs:2403) - Invalid and unbounded requests fail before field evaluation. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## governed_observation_corpus

Evidence kind: SemanticRealization. Criterion: Required governed cases include: nodal, cell, element, integration-point, face, edge, modal, and global outputs where supported; frame transformations, tensor invariants, complex magnitude/phase, and sign conventions; point, path, boundary, volume, interface, connector, and reference-point observations; final, increment, time-point, frequency, mode, event, and adaptive sampling; integral, resultan...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::results::observation::acceleration_tests::governed_extrema_reject_unsupported_work_before_provider_allocation` (crates/runmat-analysis/fea/src/results/observation/acceleration_tests.rs:218) - Governed extrema reject unsupported work before provider allocation.<br>`runmat-analysis-fea[lib]::results::observation::tests::element_field_point_sample_reads_the_containing_element` (crates/runmat-analysis/fea/src/results/observation/tests.rs:2944) - A point sample of an element field returns the native value of the element that contains the point (not a centroid match), with that element as its entity, on a...<br>`runmat-analysis-fea[lib]::results::observation::tests::element_field_over_a_boundary_region_reads_the_face_owners` (crates/runmat-analysis/fea/src/results/observation/tests.rs:3012) - an element field over a boundary-face region reads each face's owning element; its area-weighted average and its maximum match the owners' values on a three-ele...<br>`runmat-analysis-fea[lib]::results::observation::tests::node_located_von_mises_binds_its_declared_transfer_and_interpolates` (crates/runmat-analysis/fea/src/results/observation/tests.rs:3134) - a nodal von Mises field, which the solver provides only as the volume-weighted transfer of element values, binds by its node location and is sampled at a point ...<br>`runmat-analysis-fea[lib]::results::observation::tests::reaction_resultant_over_a_face_region_sums_its_nodes_once` (crates/runmat-analysis/fea/src/results/observation/tests.rs:3197) - a reaction-force resultant over a boundary face region is the discrete sum of the nodal reactions at the nodes of that region's faces, each node counted once, o...<br>`runmat-runtime[lib]::analysis::field_scene::deformation::scene_tests::original_modal_vectors_keep_their_values_while_display_shape_is_normalization_invariant` (crates/runmat-runtime/src/analysis/field_scene/deformation.rs:64) - A modal scene's displayed positions are invariant under eigenvector rescaling while its original field values and scalar range remain intact. Synthetic scene on... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

