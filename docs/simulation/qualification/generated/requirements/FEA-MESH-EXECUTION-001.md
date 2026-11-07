# FEA-MESH-EXECUTION-001: Deterministic, distributed and bounded meshing

0 PASS, 4 PARTIAL, 2 FAIL, 17 OPEN, 0 TAGGED. Generated; do not edit.

## worker_batches

Evidence kind: ExecutionEquivalence. Criterion: Deterministic edge and face batches execute on worker processes without manual partitioning, and the same artifact results from one or many workers.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## backend_equivalence

Evidence kind: ExecutionEquivalence. Criterion: Serial, local-process, durable-local and remote pool runs return the same canonical topology and digest.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## partitioned_volume_cdt

Evidence kind: ExecutionEquivalence. Criterion: A single connected body's volume mesh partitions across workers with owned and ghost regions, and the assembled mesh has the same validity and quality guarantees as the serial mesh and a stated speedup on at least 4 workers.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |

## parallel_meshing_on_one_machine

Evidence kind: ExecutionEquivalence. Criterion: Meshing saturates the admitted CPU threads of one machine under the single thread plan, with identical output at 1, 2 and 4 threads.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |

## worker_loss_and_transfer

Evidence kind: ProcessLifecycle. Criterion: Worker loss retries once with fencing, transfers are resumable and encrypted end to end, stale results never evict inputs, and durable supervisor restart and durable cancellation recover.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | native_process | PARTIAL | `runmat-execution-runner-native[test:meshing_process_conformance]::main` (crates/runmat-execution-runner-native/tests/meshing_process_conformance.rs:269) - The whole harness-less meshing process-conformance target (worker child processes, progress, cancellation, remote conformance and remote recovery) exits success... |
| shared_infrastructure | canonical_mixed_topology | durable | PARTIAL | `runmat-execution-runner-native[lib]::supervisor::tests::detached_job_survives_supervisor_reopen_and_submission_is_idempotent` (crates/runmat-execution-runner-native/src/supervisor/tests.rs:81) - A detached job survives supervisor reopen and idempotent resubmission; meshing stages are not the workload here.<br>`runmat-execution-runner-native[lib]::supervisor::tests::cancellation_is_durable_and_corrupt_metadata_recovers_indeterminate` (crates/runmat-execution-runner-native/src/supervisor/tests.rs:143) - Cancellation is durable and corrupt metadata recovers as indeterminate. |
| shared_infrastructure | canonical_mixed_topology | remote_execution | FAIL | `runmat-execution-runner-native[lib]::remote::driver_replacement_tests::exact_meshing_admission_retries_and_cancels_after_encrypted_driver_replacement` (crates/runmat-execution-runner-native/src/remote/driver_replacement_tests.rs:823) - Exact meshing admission retries and cancels after an actual encrypted driver replacement.<br>`runmat-execution-runner-native[lib]::remote::driver_replacement_tests::exact_meshing_admission_exhausts_retry_budget_after_two_driver_losses` (crates/runmat-execution-runner-native/src/remote/driver_replacement_tests.rs:851) - The retry budget is exhausted after two driver losses; worker-process loss and transfer resumption are covered by the process-conformance target.<br>`runmat-execution-runner-native[lib]::remote::pool_objects::tests::remote_meshing_catalog_bounds_actual_files_and_preserves_content_identity` (crates/runmat-execution-runner-native/src/remote/pool_objects/tests.rs:135) - The remote meshing object catalog bounds actual files and preserves content identity. |

## determinism_across_layouts_and_platforms

Evidence kind: ExecutionEquivalence. Criterion: Canonical topology and digest are identical across worker counts, partition layouts and the supported platforms, on a matrix that includes macOS and Linux.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## budgets_and_cancellation

Evidence kind: ResourceAdmission. Criterion: Element, memory, scratch, search, recursion and time budgets and cooperative cancellation are enforced inside every stage with typed failures.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-execution[lib]::serial_tests::volume::execution_ceilings_do_not_change_the_published_mesh` (crates/runmat-meshing/execution/src/serial_tests/volume.rs:515) - Execution ceilings decide only whether a run completes; the same request meshed under 512 MiB and 8 GiB ceilings publishes the same artifact and digest.<br>`runmat-meshing-execution[lib]::serial_tests::serial_stage_reports_typed_cancellation_and_search_budget_failures` (crates/runmat-meshing/execution/src/serial_tests.rs:810) - A serial stage reports typed cancellation and search-budget failures.<br>`runmat-meshing-tetrahedron[lib]::cdt::insertion::tests::insertion_enforces_search_budgets_and_cancellation` (crates/runmat-meshing/tetrahedron/src/cdt/insertion/tests.rs:375) - Volume insertion enforces search budgets and cancellation. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## bounded_publication_and_cold_materialization

Evidence kind: PublicationIntegrity. Criterion: Mesh publication and cold materialization are bounded and compact at representative scale (millions of elements) and never exceed declared codec or cache caps.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cdt::volume_mesh::codec::admission::tests::cold_current_and_legacy_decode_admit_before_materialization_and_release_on_failure` (crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/codec/admission/tests.rs:11) - Cold volume decode admits before materializing and releases on failure; million-element scale is not asserted here. |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## single_scheduler_and_browser_policy

Evidence kind: RejectionInvariant. Criterion: Meshing reuses the execution configuration with no second scheduler, and the browser either selects a native or remote mesher or fails explicitly, never meshing silently.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## performance_slo_and_governance

Evidence kind: PerformanceMeasurement. Criterion: A versioned performance and resource SLO covers the corpus parts, current reports are regenerated each release, and the meshing governance validators (corpus, conformance, differential, reliability, release evidence) pass.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | FAIL | `runmat-analysis-qualification[test:release_checks]::meshing_conformance_catalog_validates` (crates/runmat-analysis/qualification/tests/release_checks.rs:73) - The meshing conformance catalog validator passes (15 cases covering the 16 required feature classes); the SLO report freshness and the other validators are sepa...<br>`runmat-analysis-qualification[test:release_checks]::meshing_corpus_catalog_validates` (crates/runmat-analysis/qualification/tests/release_checks.rs:82) - The meshing corpus validator passes: the fixture inventory matches the registered corpus.<br>`runmat-analysis-qualification[test:release_checks]::meshing_differential_catalog_validates` (crates/runmat-analysis/qualification/tests/release_checks.rs:91) - The meshing differential catalog validator passes (independent mesher cases dispositioned).<br>`runmat-analysis-qualification[test:release_checks]::meshing_reliability_catalog_anchors_exist` (crates/runmat-analysis/qualification/tests/release_checks.rs:101) - The meshing reliability catalog validator passes: every reliability control (budgets, cancellation, atomic writes, restart, worker loss, stale publication) anch... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

