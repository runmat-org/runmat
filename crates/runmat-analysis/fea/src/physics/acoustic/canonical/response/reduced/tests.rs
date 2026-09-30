//! Independent dense complex oracle for genuinely free phased pressure DOFs.

use super::{solve, AcousticAssembly, AcousticHarmonicPolicy};
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, RealSparseMatrix},
    constraints::{build_phase_reduction, UnitPhaseRelation},
    excitation::ProjectionScalar,
    topology::AnalysisProjection,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

type Complex = (f64, f64);

const STIFFNESS: [[f64; 4]; 4] = [
    [6.0, -1.1, 0.3, -0.4],
    [-1.1, 5.0, -0.7, 0.2],
    [0.3, -0.7, 4.0, -0.5],
    [-0.4, 0.2, -0.5, 3.0],
];
const MASS: [[f64; 4]; 4] = [
    [0.8, 0.1, 0.02, 0.04],
    [0.1, 0.7, 0.03, 0.05],
    [0.02, 0.03, 0.9, 0.06],
    [0.04, 0.05, 0.06, 0.6],
];
const IMPEDANCE: [[f64; 4]; 4] = [
    [0.4, 0.03, 0.02, 0.01],
    [0.03, 0.3, 0.04, 0.02],
    [0.02, 0.04, 0.5, 0.03],
    [0.01, 0.02, 0.03, 0.2],
];
const OMEGA: f64 = 0.7;
const SOURCE: [Complex; 4] = [(0.9, -0.3), (-0.5, 1.2), (0.4, 0.7), (1.1, 0.2)];
const PRESCRIBED: Complex = (0.35, -0.27);

fn matrix(values: [[f64; 4]; 4], role: OperatorRole) -> RealSparseMatrix {
    let mut assembler =
        RealSparseAssembler::square(4, role, MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE);
    for (row, values) in values.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            assembler.add(row, column, *value);
        }
    }
    assembler.finish().unwrap()
}

fn assembly(phase: f64, amplitude: f64) -> AcousticAssembly {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    assert_eq!(projection.nodes.len(), 4);
    let prescribed = vec![
        None,
        None,
        Some(ProjectionScalar {
            real: amplitude * PRESCRIBED.0,
            imaginary: amplitude * PRESCRIBED.1,
        }),
        None,
    ];
    let reduction = build_phase_reduction(
        4,
        &[UnitPhaseRelation {
            primary: 0,
            secondary: 1,
            phase_radians: phase,
        }],
        &prescribed,
    )
    .unwrap();
    assert_eq!(reduction.free_dof_indices, [0, 3, 4, 7]);
    AcousticAssembly {
        projection,
        stiffness: matrix(STIFFNESS, OperatorRole::Stiffness),
        mass: matrix(MASS, OperatorRole::Mass),
        impedance: matrix(IMPEDANCE, OperatorRole::Damping),
        omega: OMEGA,
        density: vec![1.0],
        compressibility: vec![1.0],
        source: SOURCE
            .iter()
            .map(|value| ProjectionScalar {
                real: amplitude * value.0,
                imaginary: amplitude * value.1,
            })
            .collect(),
        prescribed,
        source_evidence: Vec::new(),
        reduction: Some(reduction),
    }
}

fn add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}
fn multiply(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn conjugate(a: Complex) -> Complex {
    (a.0, -a.1)
}

/// Construct T and lifting literally, without inspecting the production recovery map.
fn independent_pressure(phase: f64) -> [Complex; 4] {
    let phasor = (phase.cos(), phase.sin());
    let transform = [
        [(1.0, 0.0), (0.0, 0.0)],
        [phasor, (0.0, 0.0)],
        [(0.0, 0.0), (0.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)],
    ];
    let operator: [[Complex; 4]; 4] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (
                STIFFNESS[row][column] - OMEGA * OMEGA * MASS[row][column],
                OMEGA * IMPEDANCE[row][column],
            )
        })
    });
    let lifting = [(0.0, 0.0), (0.0, 0.0), PRESCRIBED, (0.0, 0.0)];
    let rhs: [Complex; 4] = std::array::from_fn(|row| {
        let applied = (0..4).fold((0.0, 0.0), |sum, column| {
            add(sum, multiply(operator[row][column], lifting[column]))
        });
        (SOURCE[row].0 - applied.0, SOURCE[row].1 - applied.1)
    });
    let reduced: [[Complex; 2]; 2] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..4)
                .flat_map(|i| (0..4).map(move |j| (i, j)))
                .fold((0.0, 0.0), |sum, (i, j)| {
                    add(
                        sum,
                        multiply(
                            conjugate(transform[i][row]),
                            multiply(operator[i][j], transform[j][column]),
                        ),
                    )
                })
        })
    });
    let reduced_rhs: [Complex; 2] = std::array::from_fn(|row| {
        (0..4).fold((0.0, 0.0), |sum, node| {
            add(sum, multiply(conjugate(transform[node][row]), rhs[node]))
        })
    });
    // Independent stacked complex-to-real conversion and pivoted dense elimination.
    let mut dense = [[0.0; 5]; 4];
    for (row, values) in reduced.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            dense[row][column] = value.0;
            dense[row][column + 2] = -value.1;
            dense[row + 2][column] = value.1;
            dense[row + 2][column + 2] = value.0;
        }
        dense[row][4] = reduced_rhs[row].0;
        dense[row + 2][4] = reduced_rhs[row].1;
    }
    let solution = dense_solve(dense);
    let primary = (solution[0], solution[2]);
    [
        primary,
        multiply(phasor, primary),
        PRESCRIBED,
        (solution[1], solution[3]),
    ]
}

fn dense_solve(mut augmented: [[f64; 5]; 4]) -> [f64; 4] {
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|a, b| {
                augmented[*a][column]
                    .abs()
                    .total_cmp(&augmented[*b][column].abs())
            })
            .unwrap();
        augmented.swap(column, pivot);
        let diagonal = augmented[column][column];
        assert!(diagonal.abs() > 0.1);
        for value in &mut augmented[column][column..] {
            *value /= diagonal;
        }
        let pivot_row = augmented[column];
        for (row, values) in augmented.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = values[column];
            for (value, pivot) in values[column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot;
            }
        }
    }
    std::array::from_fn(|row| augmented[row][4])
}

#[test]
fn free_phased_helmholtz_pressure_matches_independent_complex_dense_lifting() {
    for phase in [
        0.0,
        std::f64::consts::FRAC_PI_2,
        -std::f64::consts::FRAC_PI_2,
        0.731,
    ] {
        let expected = independent_pressure(phase);
        assert!(expected[0].0.hypot(expected[0].1) > 0.01);
        for amplitude in [1e-12, 1.0, 1e12] {
            let assembly = assembly(phase, amplitude);
            let (real, imaginary, iterations, residual) = solve(
                &assembly,
                AcousticHarmonicPolicy {
                    relative_tolerance: 1e-12,
                    restart: 8,
                    maximum_iterations: 64,
                },
            )
            .unwrap();
            assert!(iterations > 0, "free phased DOFs must actually be solved");
            assert!(residual < 1e-11);
            for (node, expected) in expected.iter().enumerate() {
                let error = (real[node] / amplitude - expected.0)
                    .hypot(imaginary[node] / amplitude - expected.1);
                assert!(
                    error < 2e-11,
                    "phase={phase}, amplitude={amplitude}, node={node}, error={error}"
                );
            }
            let phased = multiply(
                (phase.cos(), phase.sin()),
                (real[0] / amplitude, imaginary[0] / amplitude),
            );
            assert!(
                (real[1] / amplitude - phased.0).hypot(imaginary[1] / amplitude - phased.1) < 2e-12
            );
        }
    }
}
