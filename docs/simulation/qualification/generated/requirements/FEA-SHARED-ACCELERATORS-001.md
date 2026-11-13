# FEA-SHARED-ACCELERATORS-001: Operation-level accelerator qualification and honest fallback

0 PASS, 2 PARTIAL, 0 FAIL, 22 OPEN, 0 TAGGED. Generated; do not edit.

## provider_owned_native_sparse_work

Evidence kind: ExecutionEquivalence. Criterion: Extend `runmat-accelerate-api` with provider-neutral sparse descriptors, opaque sparse handles, capabilities, feasibility queries, and costs. Do not put WGPU objects or `GpuTensorHandle` values into the canonical analysis operator. The accelerator provider owns device buffers and exposes only capability-checked operations. Native CUDA sparse libraries and WGPU implementations may sit behind that s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::planning::resource::tests::general::general_host_solver_demand_does_not_authorize_accelerator_execution` (crates/runmat-analysis/fea/src/planning/resource/tests/general.rs:101) - Host general-solver demand plus a synthetic GPU requirement rejects planning; no actual accelerator kernel capability or performance proof. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## operation_level_promotion

Evidence kind: ExecutionEquivalence. Criterion: GPU capability is promoted by operation and solver envelope, not as one boolean claim: resident vector operations and reductions; native sparse matrix-vector application; a complete real-SPD Krylov path; a useful production preconditioner path; complex sparse operations and compatible Krylov methods; nonsymmetric, indefinite, and block systems; accelerator assembly and recovery where measurements ...

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

## resource_scoped_local_parallelism

Evidence kind: ExecutionEquivalence. Criterion: Local parallelism uses execution-resource-aware, scoped pools. It must not depend on an unconstrained global thread pool or oversubscribe CPUs through nested sweep, assembly, BLAS, solver, and provider concurrency. Plans account for memory bandwidth, bounded contribution buffers, accelerator queues, and NUMA where material. Deterministic mode fixes contribution and reduction order; throughput mode...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | PARTIAL | `runmat-analysis-fea[lib]::solve::linear::tests::scoped_pcg_matches_serial_and_rejects_nested_execution` (crates/runmat-analysis/fea/src/solve/linear/tests.rs:284) - A scoped pool solve equals the serial solve and nested pools are rejected; global pool independence and CPU resource accounting are covered in the resources req... |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

