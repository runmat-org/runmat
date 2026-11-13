# FEA-SCALE-DISTRIBUTED-001: Remote and cooperative execution for every family

0 PASS, 0 PARTIAL, 0 FAIL, 118 OPEN, 0 TAGGED. Generated; do not edit.

## remote_whole_study_every_family

Evidence kind: ExecutionEquivalence. Criterion: `runmat run --cluster` runs the family's published study remotely with encrypted inputs, no unsupported-physics rejection, and results equal to the local run under the equivalence policy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| modal_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | remote_execution | OPEN |  |
| structural_stability | canonical_tetrahedra | remote_execution | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| electrostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrical_conduction | canonical_tetrahedra | remote_execution | OPEN |  |
| magnetostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | remote_execution | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | remote_execution | OPEN |  |
| thermoelastic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrothermal | canonical_tetrahedra | remote_execution | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | remote_execution | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_modal | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | remote_execution | OPEN |  |

## sweep_and_frequency_fanout

Evidence kind: ExecutionEquivalence. Criterion: Sweep cases and independent frequency points run concurrently on local workers or a cluster, overlap in time, publish per-case status and return results in authored order identical to serial execution.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | remote_execution | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | remote_execution | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | remote_execution | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | remote_execution | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | remote_execution | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | remote_execution | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | remote_execution | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | remote_execution | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | remote_execution | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | remote_execution | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | remote_execution | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | remote_execution | OPEN |  |

## domain_decomposition

Evidence kind: ExecutionEquivalence. Criterion: One problem partitions into owned and ghost subdomains over at least 4 workers; the cooperative solve equals the serial reference within tolerance, fails collectively with epoch fencing, and is demonstrated at a model size beyond one worker's memory share.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | domain_decomposed | OPEN |  |
| modal_structural | canonical_tetrahedra | domain_decomposed | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | domain_decomposed | OPEN |  |
| structural_stability | canonical_tetrahedra | domain_decomposed | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | domain_decomposed | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | domain_decomposed | OPEN |  |
| thermal_steady | canonical_tetrahedra | domain_decomposed | OPEN |  |
| thermal_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |
| electrostatic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| electrical_conduction | canonical_tetrahedra | domain_decomposed | OPEN |  |
| magnetostatic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | domain_decomposed | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | domain_decomposed | OPEN |  |
| thermoelastic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| electrothermal | canonical_tetrahedra | domain_decomposed | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | domain_decomposed | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_modal | canonical_tetrahedra | domain_decomposed | OPEN |  |
| acoustic_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |

## checkpoint_and_mid_phase_restart

Evidence kind: ProcessLifecycle. Criterion: A long run checkpoints and restarts mid-phase after worker loss or cancellation to the same final result.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | durable | OPEN |  |
| modal_structural | canonical_tetrahedra | durable | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | durable | OPEN |  |
| structural_stability | canonical_tetrahedra | durable | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | durable | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | durable | OPEN |  |
| thermal_steady | canonical_tetrahedra | durable | OPEN |  |
| thermal_transient | canonical_tetrahedra | durable | OPEN |  |
| electrostatic | canonical_tetrahedra | durable | OPEN |  |
| electrical_conduction | canonical_tetrahedra | durable | OPEN |  |
| magnetostatic | canonical_tetrahedra | durable | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | durable | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | durable | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | durable | OPEN |  |
| thermoelastic | canonical_tetrahedra | durable | OPEN |  |
| electrothermal | canonical_tetrahedra | durable | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | durable | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | durable | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | durable | OPEN |  |
| acoustic_modal | canonical_tetrahedra | durable | OPEN |  |
| acoustic_transient | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## shared_distributed_contracts

Evidence kind: ExecutionEquivalence. Criterion: The shared remote, partitioned and thread-parallel contracts are one implementation reached by every family: no family-specific placement table, and unsupported placements fail typed before result commitment.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |

