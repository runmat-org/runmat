# FEA-DYNAMICS-GOVERNANCE-001: Governed dynamic procedures, checkpoints and distributed evidence

0 PASS, 2 PARTIAL, 0 FAIL, 133 OPEN, 0 TAGGED. Generated; do not edit.

## modal_preload_buckling

Evidence kind: ExecutionEquivalence. Criterion: Complete modal analysis on physical mass/stiffness. Implement typed linear-static structural base-state transitions, tangent/geometric-stiffness construction, and linearization evidence; leave the validated extension point for selected nonlinear producer states. Implement eigenvalue buckling for the supported preload and load-scaling envelope; explicitly reject nonlinear post-buckling implications...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::physical_compressive_preload_buckling_has_exact_residual_and_inverse_load_scaling` (crates/runmat-analysis/fea/src/tests.rs:2134) - Single-tetrahedron preload buckling with exact residual and inverse load scaling. |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## prestress_direct_dynamics

Evidence kind: ExecutionEquivalence. Criterion: Implement prestressed modal extraction and verify stress stiffening/softening against unstressed references. Implement second-order direct structural transient dynamics. Implement direct harmonic response with complex dynamic stiffness, documented damping, and frequency-dependent excitation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## participation_modal_kinematics

Evidence kind: ExecutionEquivalence. Criterion: Publish modal participation factors, effective modal mass by direction, cumulative captured mass, truncation evidence, and reusable modal-basis artifacts. Implement shared time-history force and base-motion excitation with homogeneous modal constraint rules and time-dependent affine constraint kinematics. Implement modal transient and modal harmonic response as procedures distinct from modal extra...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## spectra_random_input

Evidence kind: ExecutionEquivalence. Criterion: Implement SRS construction from a base-acceleration history as a general signal operation and typed response-spectrum admission as analysis input. Implement modal response-spectrum analysis with governed SRSS/CQC, directional combination, residual-mode, rigid-response, and missing-mass policies. Implement random vibration from typed PSD/cross-PSD inputs with correlation, modal truncation, and outp...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## checkpoint_procedure_references

Evidence kind: ExecutionEquivalence. Criterion: Add restart/checkpoint, transient continuation, immutable base-state branching, and temporal-block execution. Add governed buckling, prestressed, modal, direct/modal time-domain, direct/modal harmonic, SRS, response-spectrum, and random-response evidence. Add distributed frequency/load-case/spectrum-direction/sequence-branch parallelism; add operator partitioning only if measurements justify it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## distributed_modal_gates

Evidence kind: ExecutionEquivalence. Criterion: Define cooperative worker-group, partition epoch, halo, collective, and group-restart contracts before the first distributed operator implementation. Modal residual, orthogonality, and convergence gates pass. Buckling residual, load-scaling, sign, mesh-convergence, base-state, and tangent/geometric-stiffness gates pass within the supported envelope.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## prestress_combination_equivalence

Evidence kind: ExecutionEquivalence. Criterion: Prestressed modal and harmonic results retain complete base-state lineage and pass independent frequency/response references. Participation, effective-mass, truncation, modal/directional combination, and missing-mass evidence passes for each promoted modal-response procedure. Direct and modal transient/harmonic procedures remain distinct, and their shared validity cases agree within governed toler...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## dynamic_balance_generated_inputs

Evidence kind: ExecutionEquivalence. Criterion: Dynamic equilibrium, energy, time-step, frequency-grid, and complex power/phase gates pass. Explicit tables, RunMat closures, and named .fea project generators produce equivalent governed histories and responses. Prescribed motion produces consistent displacement, velocity, acceleration, reactions, and work.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## motion_restart_branches

Evidence kind: ExecutionEquivalence. Criterion: Durable and remote restart preserve logical results. Uninterrupted and restarted continuations agree, and independent branches cannot mutate or relabel the producer state. Cooperative distributed evidence distinguishes task distribution from a tightly coupled solve and fences stale group epochs.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::transient_physical::modal_support_motion_matches_direct_physics_and_restarts_exactly` (crates/runmat-analysis/fea/src/tests/transient_physical.rs:551) - Modal support motion matches direct physics and restarts exactly; durable and remote restart are not covered. |
| linear_structural_dynamics | tet10 | numerical_kernel | OPEN |  |
| linear_structural_dynamics | beam | numerical_kernel | OPEN |  |
| linear_structural_dynamics | shell | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | public_study | OPEN |  |
| linear_structural_dynamics | tet10 | public_study | OPEN |  |
| linear_structural_dynamics | beam | public_study | OPEN |  |
| linear_structural_dynamics | shell | public_study | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | durable | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_mixed_topology | claimed_accelerator | OPEN |  |

