# FEA-SHARED-EXECUTION-001: Placement-independent logical studies and governed scheduling

0 PASS, 7 PARTIAL, 0 FAIL, 137 OPEN, 0 TAGGED. Generated; do not edit.

## logical_study_identity

Evidence kind: ExecutionEquivalence. Criterion: A canonical logical study identity includes: geometry source identity and admitted geometry revision; canonical mesh request or solver-mesh digest; authored problem and material assignments; authored signal, excitation, initial-condition, response-spectrum, and combination contracts; generator provenance for generated authored inputs; materialized signal/field digests enter the downstream realized...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | PARTIAL | `runmat[test:check_cli]::a_script_running_the_same_study_twice_publishes_two_attempts` (crates/runmat-cli/tests/check_cli.rs:1069) - Two executions of one random-response study from a script keep one definition run id, publish distinct attempts, and the definition id resolves to the latest at... |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## optional_typed_placement

Evidence kind: ExecutionEquivalence. Criterion: The shared CLI placement surface should preserve local defaults and reuse current cluster vocabulary: The reusable placement options are: project selection; cluster selection; queue, defaulting consistently with `job submit`; worker count, with zero retaining its existing driver-only/automatic meaning; trust identity, preferably supplied by project or named target configuration; idempotency key; a...

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

## whole_study_placement

Evidence kind: ExecutionEquivalence. Criterion: Execute one complete study on one allocated worker. Materialize and validate callable-backed signals before admitting dependent assembly or solve work. Preserve the existing local inline behavior as the default. Support local process isolation, durable local execution, and remote execution. Publish the same logical result and evidence shape across placements.

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

## deterministic_independent_fanout

Evidence kind: ExecutionEquivalence. Criterion: Execute independent sweep entries concurrently. Deduplicate shared geometry, mesh, signal, response-spectrum, and realized-excitation artifacts by logical digest. Permit independent frequency points, load cases, parameter samples, sequence branches, observation reductions, and study variants to become tasks when their result composition is deterministic. Honor fail-fast and collect-all policies wi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::frequency_tests::native_frequency_fanout_and_fanin_are_portable_and_order_independent` (crates/runmat-execution-runner-native/src/analysis_kernel/frequency_tests.rs:472) - A two-frequency native fan-out and fan-in publishes samples in authored order with recovered fields per frequency; order independence under reversed completion ... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## verified_solver_partitioning

Evidence kind: ExecutionEquivalence. Criterion: Partition element or cell assembly. Use deterministic sparse contribution composition. Add distributed matrix-vector and preconditioner strategies only after the serial operator is verified. Define capability requirements by formulation, order, scalar type, backend, and required native kernel. Establish strong and appropriately tolerance-bounded result equivalence against the serial reference.

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

## checkpointed_iterative_graphs

Evidence kind: ExecutionEquivalence. Criterion: Represent sequence progression, coupled iterations, and adaptive iterations as checkpointed graph progression. Fence iteration publication so stale attempts cannot commit state. Transfer only canonical field/history artifacts between stages. Make convergence decisions deterministic from persisted evidence.

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

## one_driver_and_semantic_scheduler

Evidence kind: ExecutionEquivalence. Criterion: Use driver-owned scheduling already supported by worker allocations. Do not create nested independent schedulers inside FEA solvers. An analysis root job owns a task graph containing only the granularity justified for its level. Initially this is frozen-input admission, any required signal materialization, a mesh prerequisite, excitation realization, validated procedure-step execution, observation...

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

## cooperative_group_requirements

Evidence kind: ExecutionEquivalence. Criterion: Independent distribution uses the existing durable task model for whole studies, mesh stages, sweep entries, load cases, frequency points, recovery partitions, and other replay-safe tasks. These tasks communicate through immutable content-addressed artifacts and may be retried independently under the existing effect policy. Cooperative distributed algebra is a different execution class. It require...

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

## collective_failure_and_epoch_fencing

Evidence kind: ExecutionEquivalence. Criterion: A failed independent task may retry alone. A lost participant in a tightly coupled Krylov, nonlinear, or coupled iteration cannot publish or retry independently. The group either recovers from a collectively valid checkpoint or restarts under a new epoch. Stale epochs are fenced from all state and result publication.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | PARTIAL | `runmat-execution-runner-native[lib]::remote::pool_tests::remote_rank_failure_releases_a_peer_blocked_in_a_collective` (crates/runmat-execution-runner-native/src/remote/pool_tests.rs:2186) - A remote rank failure releases a peer blocked in a collective within five seconds and surfaces the primary structured error with its source ordinal; SPMD runtim... |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | PARTIAL | `runmat-execution-runner-native[lib]::driver::collective::tests::failure_before_submission_fences_a_late_collective_request` (crates/runmat-execution-runner-native/src/driver/collective.rs:314) - A collective request submitted after a rank failure is fenced and fails with the original reason; the gang then completes a later barrier. In-process broker onl... |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## exact_owned_and_ghost_partitions

Evidence kind: ExecutionEquivalence. Criterion: The partition plan defines owned and ghost DOFs, elements/cells, constraint rows, interfaces, and result ranges. It has its own execution identity and evidence but does not alter authored-problem or discretized-problem identity. No worker independently guesses ownership. Distributed assembly emits partition-local canonical contributions. A global matrix is not gathered unless a selected small-prob...

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

## deterministic_parallel_boundaries

Evidence kind: ExecutionEquivalence. Criterion: Parallelism is introduced at natural deterministic boundaries.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | PARTIAL | `runmat-sparse-solvers[lib]::ordering::tests::nested_dissection_is_a_valid_deterministic_permutation` (crates/runmat-sparse-solvers/src/ordering/tests.rs:13) - Fill-reducing ordering is deterministic across thread counts; assembly and solve boundaries are covered by the sibling tests. |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## bounded_shared_memory_assembly

Evidence kind: ExecutionEquivalence. Criterion: Parallel element/cell assembly into partition-local contribution buffers. Deterministic merge ordering for reproducible modes. Parallel recovery and balance evaluation. Bounded memory and cancellation checkpoints.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | PARTIAL | `runmat-analysis-fea[lib]::algebra::element_assembly::tests::assembly_is_bitwise_identical_to_the_serial_element_loop_for_every_thread_count` (crates/runmat-analysis/fea/src/algebra/element_assembly/tests.rs:331) - Parallel element assembly into partition-local buffers merges to a bit-identical result for every thread count; memory ceilings are not asserted. |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## canonical_distributed_contributions

Evidence kind: ExecutionEquivalence. Criterion: Partition by canonical element/cell identities. Externalize immutable partition inputs. Produce canonical contribution artifacts with exact ownership. Join contributions deterministically and reject duplicates or omissions.

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

## serial_reference_and_equivalence_policy

Evidence kind: ExecutionEquivalence. Criterion: Begin with embarrassingly parallel studies, load cases, and frequencies. Add distributed sparse matrix-vector work and preconditioning family by family. Define halo/ghost ownership, communication budgets, convergence aggregation, and failure recovery explicitly. Never claim bitwise equivalence when floating-point reduction order makes only tolerance-bounded equivalence meaningful; encode the appli...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | PARTIAL | `runmat-analysis-fea[lib]::algebra::element_assembly::tests::assembly_is_bitwise_identical_to_the_serial_element_loop_for_every_thread_count` (crates/runmat-analysis/fea/src/algebra/element_assembly/tests.rs:331) - Parallel assembly equals the serial reference bit for bit; the policy for remote and distributed equivalence is not exercised.<br>`runmat-analysis-fea[lib]::algebra::reference::persistent_pool_tests::installed_pool_csr_rows_preserve_serial_bits_and_output_shape` (crates/runmat-analysis/fea/src/algebra/reference.rs:534) - Pooled CSR products equal the serial reference bitwise. |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## actual_worker_capabilities

Evidence kind: ExecutionEquivalence. Criterion: Capabilities should describe actual worker requirements, including: analysis host ABI; physics formulation and implementation revision; element order and supported topology; scalar/complex precision; CPU instruction or accelerator class where required; deterministic platform cohort where strict replay requires it; native exact-CAD kernel only for tasks that perform geometry admission or meshing; a...

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

## public_entrypoint_and_task_contracts

Evidence kind: ExecutionEquivalence. Criterion: Re-audit current `origin/dev`; consume landed typed value, gang, collective, distributed, accelerator, and provider contracts before adding any platform-facing type. Reuse the already-public `runmat-types` numeric and storage facts immediately where their semantics are exact; do not wait for the broader runtime migration to duplicate them locally. Define canonical study, sequence, step, state-tran...

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

## immutable_meshing_and_materialization

Evidence kind: ExecutionEquivalence. Criterion: Make the existing meshing-owned exact DAG orchestration awaitable and implement a remote pool executor over the generic runner. Keep the driver as the root-workload orchestrator and workers as stage kernels; do not duplicate the DAG, wrap CAD in MATLAB, or create nested worker-side scheduling. Add a portable exact-meshing root workload with chunked source-CAD admission and a canonical terminal mes...

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

## bounded_lifecycle_and_partial_access

Evidence kind: ExecutionEquivalence. Criterion: Add Desktop placement and lifecycle integration using generic execution services and partial result access, including sequence/step state, generated physical inputs, observations, accuracy status, and provenance. Add conformance tests for borrowed, moved, materialized, device-resident, and distributed value admission, including copy-accounting and forbidden implicit-conversion cases. A simple veri...

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

