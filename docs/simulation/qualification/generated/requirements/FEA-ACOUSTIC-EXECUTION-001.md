# FEA-ACOUSTIC-EXECUTION-001: Governed acoustic frequency execution and original-field publication

0 PASS, 3 PARTIAL, 0 FAIL, 42 OPEN, 0 TAGGED. Generated; do not edit.

## inventory

Evidence kind: PublicationIntegrity. Criterion: Exact frequency fan-in preserves all original entries, roots, pressure/velocity fields and metadata independent of completion order or chunking, rejecting missing, duplicate or substituted entries.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::acoustic_frequency_assembly::tests::native_admitted_frequency_assembly_reopens_original_complete_fields_and_metadata` (crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_frequency_assembly/tests.rs:87) - Tet4/Tet10, two/three serial-native frequencies, reversed completion and cold filesystem fields; no actual parallel/distributed/GPU or producer-termination clai... |
| acoustic_harmonic | tet10 | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::acoustic_frequency_assembly::tests::native_admitted_frequency_assembly_reopens_original_complete_fields_and_metadata` (crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_frequency_assembly/tests.rs:87) - Tet4/Tet10, two/three serial-native frequencies, reversed completion and cold filesystem fields; no actual parallel/distributed/GPU or producer-termination clai... |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## selection

Evidence kind: PublicationIntegrity. Criterion: Cold full/partial complex reads use exact sample and original partitions without solver reruns, nearest-frequency selection or aggregate-as-single-state substitution; corruption and incomplete roots reject.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | OPEN |  |
| acoustic_harmonic | tet10 | native_process | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## admission

Evidence kind: ResourceAdmission. Criterion: Worker rederivation rejects altered workload/uniqueness controls or understated storage/work plans before work; original authored ceilings remain enforced. Whole-task measured peak usage, cancellation and terminal accounting match claimed envelopes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::acoustic_harmonic::tests::native_acoustic_rejects_valid_but_substituted_resource_plan` (crates/runmat-execution-runner-native/src/analysis_kernel/acoustic_harmonic/tests.rs:388) - Tet4 native worker plan substitution rejection; not measured RSS, whole-family lifetime or complete cancellation evidence. |
| acoustic_harmonic | tet10 | native_process | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## placement

Evidence kind: ExecutionEquivalence. Criterion: Every claimed execution placement has actual end-to-end equivalence, failure/cancellation and resource evidence. Durable producer termination, encrypted remote and partitioned/GPU behavior receive separate proof rather than local transport/fanout credit.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | OPEN |  |
| acoustic_harmonic | tet10 | native_process | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## release

Evidence kind: AccuracyAcceptance. Criterion: Release evidence and usable examples document supported boundaries, sources, material/mesh/frequency limits, diagnostics and accuracy; all blocking release criteria pass before advertisement.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | OPEN |  |
| acoustic_harmonic | tet10 | native_process | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | parallel_cpu | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | claimed_accelerator | OPEN |  |

