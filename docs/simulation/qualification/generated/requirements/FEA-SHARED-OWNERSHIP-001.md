# FEA-SHARED-OWNERSHIP-001: Domain boundaries and shared platform interoperability

0 PASS, 0 PARTIAL, 2 FAIL, 42 OPEN, 0 TAGGED. Generated; do not edit.

## domain_ownership

Evidence kind: SemanticRealization. Criterion: Meshing owns geometric and physical topology. It does not own materials, constitutive laws, loads, boundary conditions, or physics behavior. RunMat's general numerical, signal-processing, control-system, data, callable, and accelerator facilities own general-purpose signal construction and execution. FEA consumes their typed outputs and does not fork those libraries. `runmat-analysis-core` owns so...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## code_quality

Evidence kind: SemanticRealization. Criterion: Do not add god files. A module that combines contracts, topology, assembly, solution, recovery, and diagnostics must be split before it grows further. Do not duplicate a rule at multiple boundaries. Region resolution, quadrature, material lookup, field topology, and identity derivation each have one owner. Strings may identify authored data such as names and IDs, but must not encode constraint kin...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | FAIL | `runmat-analysis-qualification[test:release_checks]::architecture_boundary_check_passes` (crates/runmat-analysis/qualification/tests/release_checks.rs:36) - scripts/development/check-architecture-boundaries.mjs passes: dependency direction, forbidden imports and module-size debt ceilings; it does not review code qua... |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## numerical_truth

Evidence kind: SemanticRealization. Criterion: A deterministic output is not evidence of a correct formulation. A small algebraic residual is not enough when the assembled operator does not represent the stated equations. Moving completed arrays to a device is not GPU solver execution. Generated graphs, implicit channels, scalar profiles, extrema-based terminals, and global averages must not substitute for canonical topology in production anal...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## bridge_dependency_boundary

Evidence kind: SemanticRealization. Criterion: Keep analysis identity and numerical-object schemas in `runmat-analysis-fea`, but place their projection into generic artifacts, tasks, and host workloads in the narrow `runmat-analysis-execution` peer crate. This mirrors the proven meshing split and keeps the portable bridge reusable by native, remote, and browser hosts without moving it into any one host. Typed contract structure: runmat-analysi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | FAIL | `runmat-analysis-qualification[test:release_checks]::architecture_boundary_check_enforces_dependency_edges` (crates/runmat-analysis/qualification/tests/release_checks.rs:49) - The same architecture script enforces the dependency edges between the analysis crates and the shared platform; the full edge list in the obligation is not comp... |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## shared_dependency_graph

Evidence kind: SemanticRealization. Criterion: This lets us move runtime-owned acoustic, CFD, CHT, and FSI code into `runmat-analysis-fea` without a dependency cycle. Analysis execution identities and numerical-object schemas remain with the analysis domain; task construction and the portable host envelope live in `runmat-analysis-execution`; host-specific process and network behavior remains in `runmat-execution-runner-native`. The `.fea` fro...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## extraction_policy

Evidence kind: SemanticRealization. Criterion: Reuse means extending the existing generic substrate where the missing capability is genuinely cross-domain. FEA must not copy RunMat's placement planner, value transport, content-addressed artifacts, provider capability discovery, device residency, worker pools, retries, cancellation, or publication fencing. Conversely, solver internals must not depend on MATLAB-facing `SparseTensor` or runtime `...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## public_platform_convergence

Evidence kind: SemanticRealization. Criterion: FEA development overlaps a broad RM-1064 runtime program. The implementation must move forward without either freezing on that work or recreating it. The integration policy is: Merge the current public `origin/dev` into the FEA branch before new implementation begins, resolving CI/release and native-runner overlaps by composing both products' requirements. The `origin/dev` CI and release workflow ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## platform_capability_ownership

Evidence kind: SemanticRealization. Criterion: Before remote, distributed, or GPU promotion, perform a mandatory convergence merge from the then-current `dev`, delete superseded compatibility ports, and prove there is one authority for every shared contract. Known incoming contracts that FEA should expect to consume include: the sole `runmat-value::Value` and typed dense, sparse, real, complex, logical, and integer storage model; dependency-li...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## interoperable_public_storage

Evidence kind: SemanticRealization. Criterion: A direct pre-`dev` integration is acceptable only if the owning workstream publishes a green, stable checkpoint, confirms that its commit ancestry will be preserved when it reaches `dev`, and agrees on the shared contract surface. Otherwise FEA consumes the current dependency-neutral types, continues analysis-owned work, and merges the completed platform work through `dev`. Do not cherry-pick a ha...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## platform_fact_ownership

Evidence kind: SemanticRealization. Criterion: FEA uses the RunMat type system directly where a type describes a platform-wide fact, and keeps analysis semantics in the analysis domain. This is a deliberate split: `runmat-analysis-core` may depend on the dependency-neutral `runmat-types` crate. `runmat-analysis-fea` may consume those types directly or through analysis-core. Neither analysis crate depends on `runmat-runtime` or `runmat-value`. ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## explicit_value_adapters

Evidence kind: SemanticRealization. Criterion: The compiler and runtime should therefore understand the public FEA surface through `ValueFact` and typed class identities, while the solver receives validated analysis types. This gives MATLAB code, AOT execution, remote execution, native extensions, and future language frontends one interoperable boundary without turning the numerical kernel into a dynamic-value interpreter.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## shared_execution_authorities

Evidence kind: SemanticRealization. Criterion: The current public baseline and the in-flight RM-1064 platform work must be reconciled before creating any of these surfaces. RM-1064 already defines typed SPMD gangs, collective identities and sequencing, distributed values and codistributors, remote encrypted gang execution, typed accelerator inventories and assignments, and provider-bound execution capabilities. FEA consumes and, where necessar...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## bulk_collective_contract

Evidence kind: SemanticRealization. Criterion: The existing language-level collective path coordinates typed `ValuePayload` objects and may route through a coordinator. FEA must first measure it for numerical use. If high-frequency Krylov reductions or halo exchanges require a bulk or device-resident data plane, extend the same gang, epoch, identity, security, failure, and transport authorities with typed numerical buffer operations. Do not ad...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## authored_intent_only

Evidence kind: SemanticRealization. Criterion: This crate owns authored intent only. It must not contain element matrices, mesh indexing, scheduler tasks, runtime values, filesystem paths, or host execution details. The study model must separate: `PhysicsFamily`: structural, thermal, electromagnetic, acoustic, fluid, or coupled composition; `AnalysisProcedure`: static, modal extraction, prestressed modal extraction, eigenvalue buckling, transi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## numerical_domain_owners

Evidence kind: SemanticRealization. Criterion: There must not be both `pipeline` and `orchestration` implementations for the same family after migration. Move a family, update all callers and tests, and remove its old module in the same measurable slice. Shared topology must be physics-neutral where possible, but it may expose deliberate projections such as structural boundary integration data or finite-volume face metrics. Those projections b...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## portable_bridge_only

Evidence kind: SemanticRealization. Criterion: This crate is only the portable adapter between analysis-owned semantics and generic execution. It maps analysis numerical manifests to `LogicalObject`/`ValueRef` closures, maps typed analysis capabilities and resources to generic worker requirements, creates the `AnalysisWorkload` executable form and host envelope, and projects validated semantic graphs into generic task graphs. It must not imple...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## runtime_adaptation_only

Evidence kind: SemanticRealization. Criterion: `analysis/mod.rs` should become a small module declaration and re-export surface. Runtime operation functions should translate runtime-facing values into analysis contracts, invoke the analysis orchestrator or execution provider, and translate results back. They must not assemble matrices, generate physical fields, or perform coupling iterations. The current runtime-owned `solver_mesh_flow_topolog...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## native_generic_lifecycle

Evidence kind: SemanticRealization. Criterion: The existing generic driver, durable store, remote transport, worker pool, encryption, and supervisor stay shared. Analysis and meshing adapters translate domain workloads to the generic execution substrate. They must not fork their own schedulers or storage systems.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## shared_cli_placement

Evidence kind: SemanticRealization. Criterion: `execution_target.rs` owns the shared placement arguments used by `runmat run`, `runmat mesh`, and future domain entrypoints. Script arguments and FEA placement arguments must remain unambiguous under Clap parsing.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## content_blind_presentation

Evidence kind: SemanticRealization. Criterion: Desktop FEA orchestration consumes the shared execution lifecycle and maps it into UI sessions. No Desktop physics implementation is added. No FEA-specific Server module, database schema, plaintext request body, or result decoder is added.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## platform_value_fact_contract

Evidence kind: SemanticRealization. Criterion: Numeric class and domain; Reuse `runmat-types::NumericClass`, `NumericDomain`, and related facts. FEA validates the subset supported by each formulation and backend.. Shape, storage kind, layout, residency, aliasing, and mutation facts; Reuse `runmat-types` when describing public values, compiler facts, workload requirements, and adapter capabilities.. Live MATLAB values and dense/sparse COW buffe...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## exact_transfer_and_evaluation_owners

Evidence kind: SemanticRealization. Criterion: `topology/transfer_geometry.rs`; Geometry, identities, weights, and measures from which a transfer can be built; no field or coupling policy.. `excitation/`; Interpolation, kinematics, modal participation, response combination, and conservative realization of already admitted typed signals; no VM dispatch or project resolution.. `execution/excitation_materialization.rs`; Analysis-owned materializa...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

