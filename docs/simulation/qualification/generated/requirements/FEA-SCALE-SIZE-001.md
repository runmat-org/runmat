# FEA-SCALE-SIZE-001: Millions of degrees of freedom at stated time and memory

0 PASS, 0 PARTIAL, 0 FAIL, 94 OPEN, 0 TAGGED. Generated; do not edit.

## million_dof_public_study

Evidence kind: PerformanceMeasurement. Criterion: Each family solves a model of at least one million degrees of freedom (cells for flow) through the public command within a stated wall time and memory, on stated hardware, with the result checked against the same reference as the small case.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |

## published_size_and_time_per_example

Evidence kind: PerformanceMeasurement. Criterion: For every physics page example the node count, degrees of freedom, wall time, peak memory and hardware are generated from the run and published.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |

## whole_task_resource_measurement

Evidence kind: ResourceAdmission. Criterion: Whole-task peak resident memory is measured across worker processes and compared with the admitted estimate; admission never over-reserves a practical small model (buckling at 3,000 nodes, modal output, blade) and never admits a model that then exceeds its bound.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |

## usage_reports_and_failure_classification

Evidence kind: ResourceAdmission. Criterion: The plan reports cores, memory, wall time and core hours before a run; each run records per-stage usage and classifies failures (resource, numerical, input, infrastructure) in typed form, and the stated core-hour figure per guide is measured on the release.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | canonical_tetrahedra | public_study | OPEN |  |
| modal_structural | canonical_tetrahedra | public_study | OPEN |  |
| linear_structural_dynamics | canonical_tetrahedra | public_study | OPEN |  |
| structural_stability | canonical_tetrahedra | public_study | OPEN |  |
| nonlinear_structural | canonical_tetrahedra | public_study | OPEN |  |
| penalty_coulomb_contact | canonical_tetrahedra | public_study | OPEN |  |
| thermal_steady | canonical_tetrahedra | public_study | OPEN |  |
| thermal_transient | canonical_tetrahedra | public_study | OPEN |  |
| electrostatic | canonical_tetrahedra | public_study | OPEN |  |
| electrical_conduction | canonical_tetrahedra | public_study | OPEN |  |
| magnetostatic | canonical_tetrahedra | public_study | OPEN |  |
| harmonic_maxwell | canonical_tetrahedra | public_study | OPEN |  |
| electromagnetic_transient | canonical_tetrahedra | public_study | OPEN |  |
| saturating_magnetics | canonical_tetrahedra | public_study | OPEN |  |
| thermoelastic | canonical_tetrahedra | public_study | OPEN |  |
| electrothermal | canonical_tetrahedra | public_study | OPEN |  |
| conjugate_heat_transfer | canonical_tetrahedra | public_study | OPEN |  |
| fluid_structure_interaction | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_harmonic | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_modal | canonical_tetrahedra | public_study | OPEN |  |
| acoustic_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |

## fluid_performance_targets

Evidence kind: PerformanceMeasurement. Criterion: Steady flow on 1 million cells on a 16-core workstation completes in under 3 minutes and 4 GB, 10 million cells in under 45 minutes and 40 GB; transient flow sustains at least 2 million cell-steps per second per CPU node; strong-scaling efficiency is at least 70 percent down to 50,000 cells per core.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |

