//! Independent dense complex oracle for genuinely free phased pressure DOFs.
mod uniqueness;

use super::{solve, AcousticAssembly, AcousticHarmonicPolicy};
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, RealSparseMatrix},
    constraints::{build_phase_reduction, UnitPhaseRelation},
    excitation::ProjectionScalar,
    topology::AnalysisProjection,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

type Complex = (f64, f64);

#[test]
fn sparse_reduction_and_physical_warm_starts_match_independent_phased_solution() {
    for phase in [0.0, 0.731, std::f64::consts::FRAC_PI_2] {
        for amplitude in [1e-12, 1.0, 1e12] {
            let assembly = assembly(phase, amplitude);
            for preconditioner in [
                crate::contracts::SolverPreconditioner::None,
                crate::contracts::SolverPreconditioner::Ilu0,
            ] {
                let policy = AcousticHarmonicPolicy {
                    preconditioner,
                    relative_tolerance: 1e-12,
                    restart: 8,
                    maximum_iterations: 64,
                };
                let prepared = super::PreparedReduced::new(&assembly, policy).unwrap();
                for column in 0..prepared.rhs.len() {
                    let mut basis = vec![0.0; prepared.rhs.len()];
                    basis[column] = 1.0;
                    let full = prepared.reduction.expand_linear(&basis).unwrap();
                    let (mut real, imaginary) = assembly.apply(&full[..4], &full[4..]);
                    real.extend(imaginary);
                    let expected = prepared.reduction.transpose_apply(&real).unwrap();
                    let sparse = crate::algebra::spmv(&prepared.operator, &basis).unwrap();
                    for (actual, expected) in sparse.iter().zip(expected) {
                        assert!((actual - expected).abs() < 2e-14);
                    }
                }
                let warm = [0.08, -0.12, 0.04, 0.17].map(|value| amplitude * value);
                let result = prepared.solve(&assembly, policy, Some(&warm)).unwrap();
                let expected = independent_pressure(phase);
                for (node, expected) in expected.iter().enumerate() {
                    let error = (result.real[node] / amplitude - expected.0)
                        .hypot(result.imaginary[node] / amplitude - expected.1);
                    assert!(error < 2e-11, "phase={phase} amplitude={amplitude} method={preconditioner:?} error={error}");
                }
                let same = prepared
                    .solve(&assembly, policy, Some(&result.coordinates))
                    .unwrap();
                assert_eq!(same.iterations, 0);
                assert!(same.relative_residual <= policy.relative_tolerance);
                assert!(prepared.solve(&assembly, policy, Some(&[0.0])).is_err());
            }
        }
    }
}

#[test]
fn cancellation_never_accepts_a_converged_warm_start_or_setup() {
    let assembly = assembly(0.731, 1.0);
    let policy = AcousticHarmonicPolicy::default();
    let prepared = super::PreparedReduced::new(&assembly, policy).unwrap();
    let result = prepared.solve(&assembly, policy, None).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            handler: None,
            cancellation: token,
            attempt_id: None,
        });
    assert!(matches!(
        super::PreparedReduced::new(&assembly, policy),
        Err(super::AcousticHarmonicError::Cancelled)
    ));
    assert!(matches!(
        prepared.solve(&assembly, policy, Some(&result.coordinates)),
        Err(super::AcousticHarmonicError::Cancelled)
    ));
    assert!(matches!(
        super::ilu_error(crate::solve::krylov::right_ilu::SparseRightIluError::Cancelled),
        super::AcousticHarmonicError::Cancelled
    ));
}

#[test]
fn zero_effective_rhs_keeps_physical_relative_acceptance_across_load_scales() {
    for amplitude in [1e-12, 1.0, 1e12] {
        for homogeneous in [false, true] {
            let mut assembly = assembly(0.0, amplitude);
            assembly.omega = 1.0;
            assembly.stiffness = matrix(
                [
                    [2.0, -1.0, 0.0, 0.0],
                    [-1.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ],
                OperatorRole::Stiffness,
            );
            assembly.mass = matrix([[0.0; 4]; 4], OperatorRole::Mass);
            assembly.impedance = matrix(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                OperatorRole::Damping,
            );
            let fixed = if homogeneous {
                (0.0, 0.0)
            } else {
                (amplitude, 0.5 * amplitude)
            };
            assembly.prescribed = vec![
                Some(ProjectionScalar {
                    real: fixed.0,
                    imaginary: fixed.1,
                }),
                None,
                Some(ProjectionScalar {
                    real: 0.0,
                    imaginary: 0.0,
                }),
                Some(ProjectionScalar {
                    real: 0.0,
                    imaginary: 0.0,
                }),
            ];
            assembly.reduction = Some(build_phase_reduction(4, &[], &assembly.prescribed).unwrap());
            // Literal independent A*c for A=K+iI, not production assembly.apply.
            let source = if homogeneous {
                [(0.0, 0.0); 4]
            } else {
                [
                    (1.5 * amplitude, 2.0 * amplitude),
                    (-amplitude, -0.5 * amplitude),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ]
            };
            assembly.source = source
                .into_iter()
                .map(|(real, imaginary)| ProjectionScalar { real, imaginary })
                .collect();
            for preconditioner in [
                crate::contracts::SolverPreconditioner::None,
                crate::contracts::SolverPreconditioner::Ilu0,
            ] {
                let policy = AcousticHarmonicPolicy {
                    preconditioner,
                    relative_tolerance: 1e-12,
                    restart: 4,
                    maximum_iterations: 16,
                };
                let prepared = super::PreparedReduced::new(&assembly, policy).unwrap();
                assert!(prepared.rhs.iter().all(|value| *value == 0.0));
                let physical_reference = if homogeneous {
                    0.0
                } else {
                    1.25_f64.sqrt() * amplitude
                };
                assert!(
                    (prepared.residual_reference - physical_reference).abs()
                        <= 2e-15 * physical_reference
                );
                let cold = prepared.solve(&assembly, policy, None).unwrap();
                assert_eq!(cold.iterations, 0);
                assert_eq!(cold.relative_residual, 0.0);
                let result = prepared
                    .solve(
                        &assembly,
                        policy,
                        Some(&[3.0 * amplitude, -2.0 * amplitude]),
                    )
                    .unwrap();
                let error = result.real[1].hypot(result.imaginary[1]);
                let reference = if homogeneous {
                    65.0_f64.sqrt() * amplitude
                } else {
                    physical_reference
                };
                // Original free residual is literally (2+i)*p1.
                let independent_relative = 5.0_f64.sqrt() * error / reference;
                assert!((result.relative_residual - independent_relative).abs() < 1e-14);
                assert!(independent_relative <= policy.relative_tolerance);
                assert!(error / amplitude < 4e-12);
                assert_eq!(result.real[0], fixed.0);
                assert_eq!(result.imaginary[0], fixed.1);
            }
        }
    }
}

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
        incident_faces: Vec::new(),
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
            let super::ReducedSolution {
                real,
                imaginary,
                iterations,
                relative_residual: residual,
                ..
            } = solve(
                &assembly,
                AcousticHarmonicPolicy {
                    preconditioner: crate::contracts::SolverPreconditioner::Ilu0,
                    relative_tolerance: 1e-12,
                    restart: 8,
                    maximum_iterations: 64,
                },
                None,
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
