# FEA-DYNAMICS-RESPONSE-001: Distinct second-order direct and modal dynamic procedures

0 PASS, 8 PARTIAL, 0 FAIL, 67 OPEN, 0 TAGGED. Generated; do not edit.

## second_order_equilibrium_excitation

Evidence kind: IndependentReference. Criterion: Replace first-order relaxation with a documented second-order integrator, initially generalized-alpha or Newmark with explicit parameter policy. Solve M a + C v + K u = f(t) with physical mass, damping, initial displacement, initial velocity, and load history. Support force, moment, traction, pressure, body/base displacement, velocity, and acceleration histories through the shared typed excitation...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::dynamics::modal_transient::tests::average_acceleration_modal_integration_is_second_order` (crates/runmat-analysis/fea/src/solve/dynamics/modal_transient.rs:493) - Average-acceleration modal integration is second order.<br>`runmat-analysis-fea[lib]::solve::transient::newmark::tests::undamped_free_vibration_converges_at_second_order` (crates/runmat-analysis/fea/src/solve/transient/newmark.rs:461) - Undamped free vibration under Newmark converges at second order; damped and forced cases are other tests. |
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

## direct_modal_procedures

Evidence kind: IndependentReference. Criterion: Implement modal transient response, modal harmonic response, and modal response-spectrum analysis as procedures distinct from eigenmode extraction. Implement direct harmonic response from the documented complex dynamic-stiffness operator, physical damping model, frequency-dependent excitation, and reusable pattern/value identities; do not route it through modal reduction implicitly. Support direct...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::curved_tet10_tests::harmonic::physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references` (crates/runmat-analysis/fea/src/pipeline/modal/curved_tet10_tests/harmonic.rs:89) - Direct and modal harmonic procedures agree on a curved solid.<br>`runmat-analysis-fea[lib]::pipeline::modal::rod_qualification::solid_rod_harmonic_face_traction_converges_to_independent_continuum` (crates/runmat-analysis/fea/src/pipeline/modal/rod_qualification.rs:192) - Original axial Tet4 rod harmonic traction matrix at three frequencies on both sides of resonance; independent complex displacement/reaction refinement and phase...<br>`runmat-analysis-fea[lib]::solve::dynamics::modal_projection::tests::applied_load_static_correction_collapses_the_modal_truncation_balance` (crates/runmat-analysis/fea/src/solve/dynamics/modal_projection.rs:620) - applied-load static correction cuts the full-space truncation balance of a one-mode modal-reduced harmonic solve of the physical Tet4 slab rod below its first m... |
| linear_structural_dynamics | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::curved_tet10_tests::harmonic::physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references` (crates/runmat-analysis/fea/src/pipeline/modal/curved_tet10_tests/harmonic.rs:89) - Direct and modal harmonic procedures agree on a curved solid. |
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

## spectrum_random_recovery

Evidence kind: IndependentReference. Criterion: Implement typed response spectra, modal and directional combination, participation/mass capture, residual-mode and missing-mass policy, and attributable combination evidence. Support random vibration only from governed PSD/cross-PSD inputs with documented correlation and output statistics. Recover velocity and acceleration from the integration state, not post-hoc finite differences alone.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::dynamics::random_response::tests::full_cross_spectrum_can_cancel_identical_base_motion_channels` (crates/runmat-analysis/fea/src/solve/dynamics/random_response/tests.rs:223) - A full cross-spectrum can cancel identical base-motion channels; PSD integration and residual-mode terms are other tests.<br>`runmat-analysis-fea[lib]::solve::dynamics::response_spectrum::qualification::modal_combination_matches_independent_quadratic_form_references` (crates/runmat-analysis/fea/src/solve/dynamics/response_spectrum/qualification.rs:13) - Scalar signed synthetic modal vectors verify SRSS, absolute sum and CQC against independent formulas. Mesh-independent arithmetic only; no missing-mass, directi... |
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

## time_policy_energy_checkpoints

Evidence kind: IndependentReference. Criterion: Support adaptive time stepping only through documented local error or convergence criteria. Report energy behavior, numerical dissipation, stability parameters, rejected steps, and solver convergence. Use checkpointed temporal blocks for long durable/remote runs.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::transient::rod_qualification::affine_forcing::affine_physical_forcing_qualifies_tet4_tet10_temporal_fields_and_work_energy` (crates/runmat-analysis/fea/src/pipeline/transient/rod_qualification/affine_forcing.rs:213) - Affine physical forcing qualifies Tet4 and Tet10 temporal fields and work-energy; adaptive time stepping and checkpoints are not covered. |
| linear_structural_dynamics | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::transient::rod_qualification::affine_forcing::affine_physical_forcing_qualifies_tet4_tet10_temporal_fields_and_work_energy` (crates/runmat-analysis/fea/src/pipeline/transient/rod_qualification/affine_forcing.rs:213) - Affine physical forcing qualifies Tet4 and Tet10 temporal fields and work-energy; adaptive time stepping and checkpoints are not covered. |
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

## references

Evidence kind: IndependentReference. Criterion: Evidence includes undamped and damped oscillators, impulse and base-motion response, callable/table equivalence, modal superposition comparisons, SRS construction references, SRSS/CQC and directional-combination references, missing-mass cases, random-response references where supported, energy behavior, and time-step convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural_dynamics | mesh_independent | numerical_kernel | OPEN |  |
| linear_structural_dynamics | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::curved_tet10_tests::harmonic::physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references` (crates/runmat-analysis/fea/src/pipeline/modal/curved_tet10_tests/harmonic.rs:89) - Direct and modal harmonic match independent phase and power references on a curved solid; oscillators, impulse and base-motion references are not covered. |
| linear_structural_dynamics | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::pipeline::modal::curved_tet10_tests::harmonic::physical_solid_direct_and_modal_harmonic_match_independent_phase_and_power_references` (crates/runmat-analysis/fea/src/pipeline/modal/curved_tet10_tests/harmonic.rs:89) - Direct and modal harmonic match independent phase and power references on a curved solid; oscillators, impulse and base-motion references are not covered. |
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

