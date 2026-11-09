# FEA-EM-MAXWELL-001: Integrated harmonic edge-element Maxwell analysis

0 PASS, 1 PARTIAL, 0 FAIL, 2 OPEN, 0 TAGGED. Generated; do not edit.

## basis

Evidence kind: IndependentReference. Criterion: Complete oriented tetrahedral edge topology, integrated Nedelec basis and material-interface continuity; assemble physical curl-curl, mass/loss, source, boundary and gauge terms per material region.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |

## frequency

Evidence kind: ExecutionEquivalence. Criterion: Support complex coefficients and deterministic frequency-sweep composition, complex source phasors, incident and supported port excitation with reusable pattern/value identities and documented frequency conventions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::maxwell::tests::terminal::current_driven_strip_impedance_converges_to_the_skin_effect_solution` (crates/runmat-analysis/fea/src/physics/maxwell/tests/terminal.rs:253) - Three frequencies are solved with complex coefficients and each matches its closed form; composition of several frequencies in one study is not asserted. |

## reconstruction

Evidence kind: PhysicalBalance. Criterion: Use documented projection/reconstruction for vector recovery; retain original divergence, energy, power, boundary and linear-solver balance diagnostics.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |

