# FEA-EM-QUALIFICATION-001: Independent per-formulation evidence and governed electromagnetic execution

0 PASS, 13 PARTIAL, 0 FAIL, 62 OPEN, 0 TAGGED. Generated; do not edit.

## references

Evidence kind: SpatialConvergence. Criterion: Pass independent capacitors, conductive bars, magnetostatic references, cavity and waveguide modes, skin effect, material interfaces and manufactured fields applicable to each formulation, with mesh and frequency convergence and gauge/divergence, field, energy and power evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::maxwell::tests::terminal::current_driven_strip_impedance_converges_to_the_skin_effect_solution` (crates/runmat-analysis/fea/src/physics/maxwell/tests/terminal.rs:253) - Independent skin-effect closed form for strip impedance with convergence in mesh size at three frequencies.<br>`runmat-analysis-fea[lib]::physics::maxwell::tests::terminal::voltage_drive_and_low_frequency_limit_match_dc_resistance_and_inductance` (crates/runmat-analysis/fea/src/physics/maxwell/tests/terminal.rs:302) - A voltage-driven strip at 1 Hz matches the DC resistance and the low-frequency inductance. |
| electrostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::charge_density::tet4_charged_slab_has_independent_spatial_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests/charge_density.rs:116) - Uniform-charge Tet4 slab on three meshes: independent quadratic potential, linear field, charge conservation and energy convergence; no Tet10 or nonlinear diele...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:338) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:338) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrical_conduction | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:338) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n... |
| electrical_conduction | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:338) - Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact field/energy and Tet4 refinement across three meshes; harmonic polynomial is n...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::copper_bar_carries_sigma_v_over_l_and_dissipates_v_squared_over_r` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:877) - The validation table's steady-current check: copper (59.6 MS/m) with 1 mV across a 1 m bar gives the uniform current density sigma V / L, the Joule power V^2 / ... |
| magnetostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::square_uniform_current_has_independent_potential_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:84) - Magnetostatic reference: uniform-current square conductor against the Fourier series with mesh convergence of potential, flux density and energy on Tet4 and Tet... |
| magnetostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::square_uniform_current_has_independent_potential_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:84) - Magnetostatic reference: uniform-current square conductor against the Fourier series with mesh convergence of potential, flux density and energy on Tet4 and Tet... |

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
| electrostatic | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:38) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrostatic | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:38) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrical_conduction | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:38) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| electrical_conduction | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::electrical_tests::electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/electrical_tests.rs:19) - Both equations/orders retained receipts provide only algebraic residual/closure evidence; independent spatial/convergence obligations remain indeterminate. Not ...<br>`runmat-analysis-fea[lib]::execution::electrical_static::tests::equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity` (crates/runmat-analysis/fea/src/execution/electrical_static/tests.rs:38) - In-process both scalar equations and orders retain original slab fields and canonical source-bound cold identity; changed controls/equation/mesh reject. No prod... |
| magnetostatic | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::magnetic_tests::magnetic_receipts_supply_only_original_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/magnetic_tests.rs:55) - Both affine geometry orders/signs receipts preserve original algebraic residual/gauge/energy claims and block production when independent convergence evidence i...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_portable_source_bound_cold_result_preserves_both_orders_and_signs` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:22) - In-process affine both-order magnetic source-bound cold bytes and sign/zero-current inventory/work; no native process, restart, partition or measured RSS eviden...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_compact_retained_result_round_trips_within_its_planned_wire` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:174) - The compact retained result (values per (element, rule point), layout identities once, geometry from the mesh) round-trips byte-exactly, stays within its planne...<br>`runmat-analysis-fea[lib]::results::observation::catalog::sampled::tests::large_catalog_metadata_validation_fits_the_typed_decode_reservation` (crates/runmat-analysis/fea/src/results/observation/catalog/sampled/tests.rs:494) - Sampled publication catalogs of 0.3 and 2.3 MB (2- and 4-cell slabs, beyond the fixed per-contract allowance) are admitted by the accuracy reservation basis (fo... |
| magnetostatic | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::execution::accuracy_source::magnetic_tests::magnetic_receipts_supply_only_original_algebraic_equilibrium_evidence` (crates/runmat-analysis/fea/src/execution/accuracy_source/magnetic_tests.rs:55) - Both affine geometry orders/signs receipts preserve original algebraic residual/gauge/energy claims and block production when independent convergence evidence i...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_portable_source_bound_cold_result_preserves_both_orders_and_signs` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:22) - In-process affine both-order magnetic source-bound cold bytes and sign/zero-current inventory/work; no native process, restart, partition or measured RSS eviden...<br>`runmat-analysis-fea[lib]::execution::magnetic_static::tests::magnetic_compact_retained_result_round_trips_within_its_planned_wire` (crates/runmat-analysis/fea/src/execution/magnetic_static/tests.rs:174) - The compact retained result (values per (element, rule point), layout identities once, geometry from the mesh) round-trips byte-exactly, stays within its planne... |
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

