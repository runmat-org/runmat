use super::*;
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler},
    excitation::ProjectionScalar,
    topology::AnalysisProjection,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn assembly(amplitude: f64) -> AcousticAssembly {
    let matrix = |values: [f64; 4], role| {
        let mut builder =
            RealSparseAssembler::square(4, role, MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE);
        for (row, value) in values.into_iter().enumerate() {
            builder.add(row, row, value);
        }
        builder.finish().unwrap()
    };
    AcousticAssembly {
        projection: AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(
            ElementOrder::Tet4,
        ))
        .unwrap(),
        stiffness: matrix([1., 2., 3., 4.], OperatorRole::Stiffness),
        mass: matrix([0.; 4], OperatorRole::Mass),
        impedance: matrix([0.1, 0.5, 0.8, 1.], OperatorRole::Damping),
        omega: 1.,
        density: vec![1.],
        compressibility: vec![1.],
        source: (1..=4)
            .map(|i| ProjectionScalar {
                real: amplitude * i as f64,
                imaginary: 0.,
            })
            .collect(),
        prescribed: vec![None; 4],
        source_evidence: vec![],
        incident_faces: Vec::new(),
        reduction: None,
    }
}

#[test]
fn nonfinite_apparent_power_cannot_certify_zero_relative_error() {
    let mut a = assembly(1.0);
    a.impedance = RealSparseAssembler::square(
        4,
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .finish()
    .unwrap();
    a.source.fill(ProjectionScalar {
        real: 1e200,
        imaginary: 0.0,
    });
    // Exact signed real source-work is zero but apparent-power multiplication overflows.
    assert!(matches!(
        power::evaluate(&a, &[1e200; 4], &[0.0; 4]),
        Err(AcousticHarmonicError::Operator(_))
    ));
}

#[test]
fn loose_linear_stop_does_not_authorize_power_and_refinement_matches_complex_division() {
    for amplitude in [1., 1e-12] {
        let assembly = assembly(amplitude);
        let policy = AcousticHarmonicPolicy {
            preconditioner: crate::contracts::SolverPreconditioner::None,
            relative_tolerance: 0.99,
            restart: 8,
            maximum_iterations: 32,
        };
        let coarse = reduced::solve(&assembly, policy, None).unwrap();
        let coarse_power = power::evaluate(&assembly, &coarse.real, &coarse.imaginary).unwrap();
        assert!(coarse.relative_residual <= policy.relative_tolerance);
        assert!(coarse_power.power_relative_error > 1e-10);
        let (actual, closure) = solve(&assembly, policy, 1e-10).unwrap();
        assert!(actual.iterations > coarse.iterations);
        assert!(actual.iterations as usize <= policy.maximum_iterations);
        assert!(closure.power_relative_error <= 1e-10);
        // Independent diagonal complex division p_i=b_i/(K_i+iB_i).
        for (i, loss) in [0.1, 0.5, 0.8, 1.].into_iter().enumerate() {
            let k = (i + 1) as f64;
            let denominator = k * k + loss * loss;
            let expected = [
                amplitude * k * k / denominator,
                -amplitude * k * loss / denominator,
            ];
            assert!((actual.real[i] - expected[0]).abs() / amplitude < 1e-9);
            assert!((actual.imaginary[i] - expected[1]).abs() / amplitude < 1e-9);
        }
    }
}

#[test]
fn power_refinement_cannot_reset_or_enlarge_the_authored_iteration_budget() {
    let error = solve(
        &assembly(1.),
        AcousticHarmonicPolicy {
            preconditioner: crate::contracts::SolverPreconditioner::None,
            relative_tolerance: 0.99,
            restart: 8,
            maximum_iterations: 1,
        },
        1e-10,
    )
    .err()
    .expect("one loose iteration cannot close passive power");
    assert!(matches!(
        error,
        AcousticHarmonicError::PowerNotConverged { iterations: 1, .. }
    ));
    for tolerance in [0., -1., f64::NAN, f64::INFINITY, 1.] {
        assert!(solve(&assembly(1.), AcousticHarmonicPolicy::default(), tolerance).is_err());
    }
}
