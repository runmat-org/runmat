# FEA-EM-QUALIFICATION-001: Independent per-formulation evidence and governed electromagnetic execution

10 PASS, 0 FAIL, 65 OPEN, 0 TAGGED. Generated; do not edit.

## references

Evidence kind: SpatialConvergence. Criterion: Pass independent capacitors, conductive bars, magnetostatic references, cavity and waveguide modes, skin effect, material interfaces and manufactured fields applicable to each formulation, with mesh and frequency convergence and gauge/divergence, field, energy and power evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| electrostatic | tet4 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::electrical::tests::charge_density::tet4_charged_slab_has_independent_spatial_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs:104) - Uniform-charge Tet4 slab on three meshes: independent quadratic potential, linear field, charge conservation and energy convergence; no Tet10 or nonlinear diele...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:295) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrostatic | tet10 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:295) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrical_conduction | tet4 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:295) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrical_conduction | tet10 | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:295) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| magnetostatic | tet4 | numerical_kernel | OPEN |  |
| magnetostatic | tet10 | numerical_kernel | OPEN |  |

## execution

Evidence kind: ExecutionEquivalence. Criterion: Parallel independent frequencies and sources and justified mature operator partitions preserve equivalent original fields and balances. Whole-task resource admission, cancellation, retries, durable/remote lineage and publication remain governed across every claimed placement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | public_study | OPEN |  |
| electrostatic | tet10 | public_study | OPEN |  |
| electrical_conduction | tet4 | public_study | OPEN |  |
| electrical_conduction | tet10 | public_study | OPEN |  |
| magnetostatic | tet4 | public_study | OPEN |  |
| magnetostatic | tet10 | public_study | OPEN |  |
| electrostatic | canonical_mixed_topology | native_process | OPEN |  |
| electrostatic | canonical_mixed_topology | durable | OPEN |  |
| electrostatic | canonical_mixed_topology | remote_execution | OPEN |  |
| electrostatic | canonical_mixed_topology | parallel_cpu | OPEN |  |
| electrostatic | canonical_mixed_topology | domain_decomposed | OPEN |  |
| electrostatic | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| electrical_conduction | canonical_mixed_topology | native_process | OPEN |  |
| electrical_conduction | canonical_mixed_topology | durable | OPEN |  |
| electrical_conduction | canonical_mixed_topology | remote_execution | OPEN |  |
| electrical_conduction | canonical_mixed_topology | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_mixed_topology | domain_decomposed | OPEN |  |
| electrical_conduction | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| magnetostatic | canonical_mixed_topology | native_process | OPEN |  |
| magnetostatic | canonical_mixed_topology | durable | OPEN |  |
| magnetostatic | canonical_mixed_topology | remote_execution | OPEN |  |
| magnetostatic | canonical_mixed_topology | parallel_cpu | OPEN |  |
| magnetostatic | canonical_mixed_topology | domain_decomposed | OPEN |  |
| magnetostatic | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | native_process | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | durable | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | remote_execution | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | domain_decomposed | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| electrostatic | canonical_mixed_topology | public_study | OPEN |  |
| electrical_conduction | canonical_mixed_topology | public_study | OPEN |  |
| magnetostatic | canonical_mixed_topology | public_study | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | public_study | OPEN |  |

## accuracy

Evidence kind: PublicationIntegrity. Criterion: Publish correct formulation and release documentation with original controls and evidence. Missing, failed or indeterminate mandatory accuracy prevents publication; cold claims require authenticated original source lineage and never reconstruct lost physics by decoding summaries.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:36) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrostatic | tet10 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:36) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrical_conduction | tet4 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:36) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrical_conduction | tet10 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:36) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| magnetostatic | tet4 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::magnetic_tests::magnetic_receipts_supply_only_original_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/magnetic_tests.rs:55) - Both affine geometry orders/signs receipts preserve original algebraic residual/gauge/energy claims and block production when independent convergence evidence i...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_portable_source_bound_cold_result_preserves_both_orders_and_signs` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:22) - In-process affine both-order magnetic source-bound cold bytes and sign/zero-current inventory/work; no native process, restart, partition or measured RSS eviden... |
| magnetostatic | tet10 | public_study | PASS | `runmat-analysis-fea[lib]::execution::accuracy_source::magnetic_tests::magnetic_receipts_supply_only_original_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/magnetic_tests.rs:55) - Both affine geometry orders/signs receipts preserve original algebraic residual/gauge/energy claims and block production when independent convergence evidence i...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_portable_source_bound_cold_result_preserves_both_orders_and_signs` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:22) - In-process affine both-order magnetic source-bound cold bytes and sign/zero-current inventory/work; no native process, restart, partition or measured RSS eviden... |
| electrostatic | canonical_mixed_topology | native_process | OPEN |  |
| electrostatic | canonical_mixed_topology | durable | OPEN |  |
| electrostatic | canonical_mixed_topology | remote_execution | OPEN |  |
| electrostatic | canonical_mixed_topology | parallel_cpu | OPEN |  |
| electrostatic | canonical_mixed_topology | domain_decomposed | OPEN |  |
| electrostatic | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| electrical_conduction | canonical_mixed_topology | native_process | OPEN |  |
| electrical_conduction | canonical_mixed_topology | durable | OPEN |  |
| electrical_conduction | canonical_mixed_topology | remote_execution | OPEN |  |
| electrical_conduction | canonical_mixed_topology | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_mixed_topology | domain_decomposed | OPEN |  |
| electrical_conduction | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| magnetostatic | canonical_mixed_topology | native_process | OPEN |  |
| magnetostatic | canonical_mixed_topology | durable | OPEN |  |
| magnetostatic | canonical_mixed_topology | remote_execution | OPEN |  |
| magnetostatic | canonical_mixed_topology | parallel_cpu | OPEN |  |
| magnetostatic | canonical_mixed_topology | domain_decomposed | OPEN |  |
| magnetostatic | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | native_process | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | durable | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | remote_execution | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | domain_decomposed | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| electrostatic | canonical_mixed_topology | public_study | OPEN |  |
| electrical_conduction | canonical_mixed_topology | public_study | OPEN |  |
| magnetostatic | canonical_mixed_topology | public_study | OPEN |  |
| harmonic_maxwell | canonical_mixed_topology | public_study | OPEN |  |

