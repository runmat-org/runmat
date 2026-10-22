# FEA-SCALE-PARALLEL-001: Every family saturates the machine through one thread plan

0 PASS, 0 FAIL, 115 OPEN, 0 TAGGED. Generated; do not edit.

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

## thread_count_determinism

Evidence kind: ExecutionEquivalence. Criterion: The same study yields identical result digests, or differences within the stated tolerance, across 1, 2, 4 and all admitted threads.

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

