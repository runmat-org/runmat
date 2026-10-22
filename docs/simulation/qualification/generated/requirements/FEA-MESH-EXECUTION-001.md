# FEA-MESH-EXECUTION-001: Deterministic, distributed and bounded meshing

0 PASS, 0 FAIL, 22 OPEN, 0 TAGGED. Generated; do not edit.

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
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

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
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## bounded_publication_and_cold_materialization

Evidence kind: PublicationIntegrity. Criterion: Mesh publication and cold materialization are bounded and compact at representative scale (millions of elements) and never exceed declared codec or cache caps.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
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
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

