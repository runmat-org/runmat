# FEA-THERMAL-EXECUTION-001: Governed thermal studies, persistence and execution evidence

0 PASS, 4 PARTIAL, 0 FAIL, 84 OPEN, 0 TAGGED. Generated; do not edit.

## admission

Evidence kind: ResourceAdmission. Criterion: Admit source/initial-field realization, physical assembly, solve, retained fields, receipts and publication before the respective work; reject shortages and substituted budgets and record requested resources separately from measured usage.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | PARTIAL | `runmat-analysis-fea[lib]::planning::resource::thermal_preassembly::tests::post_solve_scratch_is_charged_inside_the_numerical_maximum` (crates/runmat-analysis/fea/src/planning/resource/thermal_preassembly/tests.rs:59) - Both-order field construction and manifest scratch are charged inside the numerical maximum of the preassembly ceiling; no measured RSS or whole native pipeline... |
| thermal_transient | tet10 | public_study | PARTIAL | `runmat-analysis-fea[lib]::planning::resource::thermal_preassembly::tests::post_solve_scratch_is_charged_inside_the_numerical_maximum` (crates/runmat-analysis/fea/src/planning/resource/thermal_preassembly/tests.rs:59) - Both-order field construction and manifest scratch are charged inside the numerical maximum of the preassembly ceiling; no measured RSS or whole native pipeline... |
| thermal_steady | tet4 | native_process | OPEN |  |
| thermal_steady | tet10 | native_process | OPEN |  |
| thermal_transient | tet4 | native_process | OPEN |  |
| thermal_transient | tet10 | native_process | OPEN |  |
| thermal_steady | tet4 | durable | OPEN |  |
| thermal_steady | tet10 | durable | OPEN |  |
| thermal_transient | tet4 | durable | OPEN |  |
| thermal_transient | tet10 | durable | OPEN |  |
| thermal_steady | tet4 | remote_execution | OPEN |  |
| thermal_steady | tet10 | remote_execution | OPEN |  |
| thermal_transient | tet4 | remote_execution | OPEN |  |
| thermal_transient | tet10 | remote_execution | OPEN |  |
| thermal_steady | tet4 | parallel_cpu | OPEN |  |
| thermal_steady | tet10 | parallel_cpu | OPEN |  |
| thermal_transient | tet4 | parallel_cpu | OPEN |  |
| thermal_transient | tet10 | parallel_cpu | OPEN |  |
| thermal_steady | tet4 | domain_decomposed | OPEN |  |
| thermal_steady | tet10 | domain_decomposed | OPEN |  |
| thermal_transient | tet4 | domain_decomposed | OPEN |  |
| thermal_transient | tet10 | domain_decomposed | OPEN |  |
| thermal_steady | tet4 | claimed_accelerator | OPEN |  |
| thermal_steady | tet10 | claimed_accelerator | OPEN |  |
| thermal_transient | tet4 | claimed_accelerator | OPEN |  |
| thermal_transient | tet10 | claimed_accelerator | OPEN |  |

## execution

Evidence kind: ExecutionEquivalence. Criterion: Demonstrate parallel assembly and parameter-study execution equivalence; qualify actual native, durable, encrypted remote, partitioned and claimed accelerator behavior. Host fallback cannot satisfy device execution and fanout cannot satisfy a distributed solve.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |
| thermal_steady | tet4 | native_process | OPEN |  |
| thermal_steady | tet10 | native_process | OPEN |  |
| thermal_transient | tet4 | native_process | OPEN |  |
| thermal_transient | tet10 | native_process | OPEN |  |
| thermal_steady | tet4 | durable | OPEN |  |
| thermal_steady | tet10 | durable | OPEN |  |
| thermal_transient | tet4 | durable | OPEN |  |
| thermal_transient | tet10 | durable | OPEN |  |
| thermal_steady | tet4 | remote_execution | OPEN |  |
| thermal_steady | tet10 | remote_execution | OPEN |  |
| thermal_transient | tet4 | remote_execution | OPEN |  |
| thermal_transient | tet10 | remote_execution | OPEN |  |
| thermal_steady | tet4 | parallel_cpu | OPEN |  |
| thermal_steady | tet10 | parallel_cpu | OPEN |  |
| thermal_transient | tet4 | parallel_cpu | OPEN |  |
| thermal_transient | tet10 | parallel_cpu | OPEN |  |
| thermal_steady | tet4 | domain_decomposed | OPEN |  |
| thermal_steady | tet10 | domain_decomposed | OPEN |  |
| thermal_transient | tet4 | domain_decomposed | OPEN |  |
| thermal_transient | tet10 | domain_decomposed | OPEN |  |
| thermal_steady | tet4 | claimed_accelerator | OPEN |  |
| thermal_steady | tet10 | claimed_accelerator | OPEN |  |
| thermal_transient | tet4 | claimed_accelerator | OPEN |  |
| thermal_transient | tet10 | claimed_accelerator | OPEN |  |

## publication

Evidence kind: PublicationIntegrity. Criterion: Classify standalone thermal operation versions correctly and preserve original thermal result, reference, topology, source and accuracy lineage through cold receipt/state publication; reject corruption, incompatible restart and partial publication.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |
| thermal_steady | tet4 | native_process | OPEN |  |
| thermal_steady | tet10 | native_process | OPEN |  |
| thermal_transient | tet4 | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::thermal::tests::native_thermal_serial_transport_import_and_terminal_state_round_trip` (crates/runmat-execution-runner-native/src/analysis_kernel/thermal/tests.rs:76) - Both-order serial native artifact import and semantic terminal state roundtrip; no producer-death, remote, partitioned or GPU credit. |
| thermal_transient | tet10 | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::thermal::tests::native_thermal_serial_transport_import_and_terminal_state_round_trip` (crates/runmat-execution-runner-native/src/analysis_kernel/thermal/tests.rs:76) - Both-order serial native artifact import and semantic terminal state roundtrip; no producer-death, remote, partitioned or GPU credit. |
| thermal_steady | tet4 | durable | OPEN |  |
| thermal_steady | tet10 | durable | OPEN |  |
| thermal_transient | tet4 | durable | OPEN |  |
| thermal_transient | tet10 | durable | OPEN |  |
| thermal_steady | tet4 | remote_execution | OPEN |  |
| thermal_steady | tet10 | remote_execution | OPEN |  |
| thermal_transient | tet4 | remote_execution | OPEN |  |
| thermal_transient | tet10 | remote_execution | OPEN |  |
| thermal_steady | tet4 | parallel_cpu | OPEN |  |
| thermal_steady | tet10 | parallel_cpu | OPEN |  |
| thermal_transient | tet4 | parallel_cpu | OPEN |  |
| thermal_transient | tet10 | parallel_cpu | OPEN |  |
| thermal_steady | tet4 | domain_decomposed | OPEN |  |
| thermal_steady | tet10 | domain_decomposed | OPEN |  |
| thermal_transient | tet4 | domain_decomposed | OPEN |  |
| thermal_transient | tet10 | domain_decomposed | OPEN |  |
| thermal_steady | tet4 | claimed_accelerator | OPEN |  |
| thermal_steady | tet10 | claimed_accelerator | OPEN |  |
| thermal_transient | tet4 | claimed_accelerator | OPEN |  |
| thermal_transient | tet10 | claimed_accelerator | OPEN |  |

## delivery

Evidence kind: AccuracyAcceptance. Criterion: Provide release evidence and reproducible examples documenting formulations, boundaries, integration method, nonlinear closure, units, accuracy limits and supported execution envelopes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

