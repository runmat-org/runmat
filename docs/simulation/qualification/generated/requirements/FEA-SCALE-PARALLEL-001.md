# FEA-SCALE-PARALLEL-001: Every family saturates the machine through one thread plan

0 PASS, 2 PARTIAL, 0 FAIL, 114 OPEN, 0 TAGGED. Generated; do not edit.

## one_thread_plan_authority

Evidence kind: ResourceAdmission. Criterion: Exactly one thread-plan authority binds the admitted cores for assembly, solve, recovery and publication of every family; no family reserves cores and runs serially, and a study that requests N threads is observed to use N.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| modal_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| structural_stability | canonical_tetrahedra | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| magnetostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermoelastic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrothermal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_modal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |

## parallel_assembly_and_recovery

Evidence kind: ExecutionEquivalence. Criterion: Operator assembly, boundary and excitation realization, field recovery and publication run in parallel and agree with the serial reference within the declared tolerance, bit-for-bit where the policy declares it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| modal_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| structural_stability | canonical_tetrahedra | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| magnetostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermoelastic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrothermal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_modal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |

## parallel_solve

Evidence kind: ExecutionEquivalence. Criterion: Direct (Cholesky, LDLT, LU), Krylov, AMG and block-eigen solves of the family run in parallel on the shared sparse layer and reproduce the serial solution and residual history within tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| modal_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| structural_stability | canonical_tetrahedra | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| magnetostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermoelastic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrothermal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_modal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |

## shared_parallel_kernels

Evidence kind: ExecutionEquivalence. Criterion: The shared parallel kernels under every family agree with their serial references: element assembly into the sparse pattern is bit-identical to the serial loop at 1, 2, 3, 7 and 16 threads; scoped Krylov solves equal the serial solution and reject nested execution; pooled CSR products preserve serial bits and output shape.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | PARTIAL | `runmat-analysis-fea[lib]::algebra::element_assembly::tests::assembly_is_bitwise_identical_to_the_serial_element_loop_for_every_thread_count` (crates/runmat-analysis/fea/src/algebra/element_assembly/tests.rs:331) - Element assembly is bit-identical to the serial loop at 1, 2, 3, 7 and 16 threads on a synthetic 3,000-element map; no family operator is assembled here.<br>`runmat-analysis-fea[lib]::algebra::reference::persistent_pool_tests::installed_pool_csr_rows_preserve_serial_bits_and_output_shape` (crates/runmat-analysis/fea/src/algebra/reference.rs:534) - Pooled CSR matrix-vector rows preserve serial bits at 2 and 4 threads on a 2,048-row matrix.<br>`runmat-analysis-fea[lib]::solve::linear::tests::scoped_pcg_matches_serial_and_rejects_nested_execution` (crates/runmat-analysis/fea/src/solve/linear/tests.rs:284) - A scoped PCG on a two-unknown SPD system equals the serial solution and repeats exactly; nested scoped execution is rejected. Tiny system, one preconditioner.<br>`runmat-sparse-solvers[lib]::ordering::tests::nested_dissection_is_a_valid_deterministic_permutation` (crates/runmat-sparse-solvers/src/ordering/tests.rs:13) - Nested-dissection ordering of a 20x20x20 Poisson problem is a valid permutation, identical across runs and at 4 threads. |

## thread_count_determinism

Evidence kind: ExecutionEquivalence. Criterion: The same study yields identical result digests, or differences within the stated tolerance, across 1, 2, 4 and all admitted threads.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | parallel_cpu | PARTIAL | `runmat[test:fea_determinism]::a_study_gives_identical_results_across_runs_workers_and_threads` (crates/runmat-cli/tests/fea_determinism.rs:138) - The loaded static box study gives bit-identical state digests and assessed values at serial and host-admissible parallel layouts (up to 8 total threads) through... |
| modal_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| structural_stability | canonical_tetrahedra | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| magnetostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermoelastic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrothermal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_modal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |

## strong_scaling_table

Evidence kind: PerformanceMeasurement. Criterion: A strong-scaling table from 1 to all cores of the stated machine, produced with the exclusive benchmark gate, is generated and published for the family's published example, with parallel efficiency and the hardware recorded.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| modal_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| structural_stability | canonical_tetrahedra | parallel_cpu | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | parallel_cpu | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermal_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrical_conduction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| magnetostatic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | parallel_cpu | OPEN |  |
| thermoelastic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| electrothermal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_modal | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |

