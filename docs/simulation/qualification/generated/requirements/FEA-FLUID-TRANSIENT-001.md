# FEA-FLUID-TRANSIENT-001: Physical transient integration and original cold chronology

0 PASS, 4 PARTIAL, 0 FAIL, 9 OPEN, 0 TAGGED. Generated; do not edit.

## inertia

Evidence kind: IndependentReference. Criterion: Integrate actual momentum time derivatives rather than return a terminal steady profile; initial state and step interval bind the original inertial operator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::transient::original_uniform_acceleration_has_exact_inertia_pressure_and_cold_binding` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/transient.rs:64) - A uniformly accelerating field has the exact inertia-driven pressure and binds cold; a nontrivial unsteady reference is not covered here.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::transient::original_mass_term_matches_independent_cell_integral_on_every_axis` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/transient.rs:248) - Independent rho*volume*delta-U/dt cell action on every axis; not temporal field convergence. |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## convergence

Evidence kind: TemporalConvergence. Criterion: Show temporal convergence independently of spatial error on a nontrivial transient reference, including a transient vortex; report field error and conservative balance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::transient::exponential_translation_pressure_shows_first_order_time_convergence` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/transient.rs:133) - Uniform exponential translation pressure on three time steps; no vortex or independent spatial sweep. |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## history

Evidence kind: ColdReplay. Criterion: Preserve complete original interval state, scheduled observations, one-sided discontinuities, predecessor order and cumulative admitted work. Reject reordered/substituted state, invalid intervals, budgets and cancellation without accepting a suffix.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::fluid::tests::transient_workload::authored_history_retains_and_publishes_fields_at_sampled_times` (crates/runmat-analysis/fea/src/execution/fluid/tests/transient_workload.rs:394) - an authored time history retains the sampled interval's fields (and drops the others), and native field recovery publishes them at their physical times beside t...<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::time_block::authored_time_block_cold_recovery_and_split_match_original_acceleration` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_block.rs:34) - Original acceleration history and split replay; chronology/tamper/budget/cancel checks, no native process credit. |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## producer_exit

Evidence kind: ProcessLifecycle. Criterion: After actual native producer termination a fresh consumer resumes original transient fields and outputs without producer memory; split and uninterrupted histories agree within the governed accuracy envelope.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | canonical_tetrahedra | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::fluid_transient::tests::terminated_native_producers_preserve_original_history_and_admission_rejects_substitution` (crates/runmat-execution-runner-native/src/analysis_kernel/fluid_transient/tests.rs:167) - Child actually exits before new consumers resume; uniform translation and substitution/budget rejection only. |

