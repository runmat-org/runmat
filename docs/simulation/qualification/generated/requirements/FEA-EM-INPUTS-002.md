# FEA-EM-INPUTS-002: Every declared electromagnetic input runs

0 PASS, 4 PARTIAL, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## scalar_potential_inputs

Evidence kind: PhysicalBalance. Criterion: terminal_voltage, electric current terminals, electric_charge, equipotential, floating and integral-current constraints run for electrostatics and steady conduction, with total current and charge balances closed.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:499) - A terminal_voltage load holds its terminal like a prescribed potential for electrostatics and steady conduction on two-cell Tet4 and Tet10 slabs; terminal curre...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::integral_current_and_floating_terminals_find_their_common_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:596) - Electric current terminal and floating inputs close the terminal current balance on Tet4 and Tet10 slabs; no charged floating conductor or equipotential set. |
| electrostatic | tet4 | public_study | OPEN |  |
| electrostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:499) - A terminal_voltage load holds its terminal like a prescribed potential for electrostatics and steady conduction on two-cell Tet4 and Tet10 slabs; terminal curre...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::integral_current_and_floating_terminals_find_their_common_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:596) - Electric current terminal and floating inputs close the terminal current balance on Tet4 and Tet10 slabs; no charged floating conductor or equipotential set. |
| electrostatic | tet10 | public_study | OPEN |  |
| electrical_conduction | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:499) - A terminal_voltage load holds its terminal like a prescribed potential for electrostatics and steady conduction on two-cell Tet4 and Tet10 slabs; terminal curre...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::integral_current_and_floating_terminals_find_their_common_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:596) - Electric current terminal and floating inputs close the terminal current balance on Tet4 and Tet10 slabs; no charged floating conductor or equipotential set. |
| electrical_conduction | tet4 | public_study | OPEN |  |
| electrical_conduction | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::electrical::tests::terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:499) - A terminal_voltage load holds its terminal like a prescribed potential for electrostatics and steady conduction on two-cell Tet4 and Tet10 slabs; terminal curre...<br>`runmat-analysis-fea[lib]::physics::electrical::tests::integral_current_and_floating_terminals_find_their_common_potential` (crates/runmat-analysis/fea/src/physics/electrical/tests.rs:596) - Electric current terminal and floating inputs close the terminal current balance on Tet4 and Tet10 slabs; no charged floating conductor or equipotential set. |
| electrical_conduction | tet10 | public_study | OPEN |  |

## magnetic_inputs

Evidence kind: SemanticRealization. Criterion: current_density, magnetic_vector_potential, magnetic.insulation and periodic constraints run for magnetostatics and harmonic Maxwell with typed rank-correct constraints.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| magnetostatic | edge_elements | numerical_kernel | OPEN |  |
| magnetostatic | edge_elements | public_study | OPEN |  |
| harmonic_maxwell | edge_elements | numerical_kernel | OPEN |  |
| harmonic_maxwell | edge_elements | public_study | OPEN |  |

