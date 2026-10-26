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
        crate::physics::acoustic::canonical::quantity_layout::pressure_operator(
            builder.finish().unwrap(),
            match role {
                OperatorRole::Stiffness => 2,
                OperatorRole::Mass => 0,
                OperatorRole::Damping => 1,
                _ => unreachable!(),
            },
        )
        .unwrap()
    };
    AcousticAssembly {
        projection: AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(
            ElementOrder::Tet4,
        ))
        .unwrap(),
        stiffness: matrix([1., 2., 3., 4.], OperatorRole::Stiffness),
        mass: matrix([0.; 4], OperatorRole::Mass),
        impedance: matrix([0.1, 0.5, 0.8, 1.], OperatorRole::Damping),
        derivative_rate: runmat_types::physical::AngularFrequency::new(1.0)
            .phasor_derivative_rate(),
        density:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::MassDensity, _>::new(
                vec![1.],
            ),
        compressibility: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Compressibility,
            _,
        >::new(vec![1.]),
        source: runmat_types::physical::Quantity::try_from_descriptor(
            crate::physics::acoustic::canonical::quantity_layout::equation_descriptor(2),
            (1..=4)
                .map(|i| ProjectionScalar {
                    real: amplitude * i as f64,
                    imaginary: 0.,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        prescribed: runmat_types::physical::Quantity::new(vec![None; 4]),
        source_evidence: vec![],
        incident_faces: Vec::new(),
        reduction: None,
    }
}

#[test]
fn nonfinite_apparent_power_cannot_certify_zero_relative_error() {
    let mut a = assembly(1.0);
    a.impedance = crate::physics::acoustic::canonical::quantity_layout::pressure_operator(
        RealSparseAssembler::square(
            4,
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .finish()
        .unwrap(),
        1,
    )
    .unwrap();
    a.source = a
        .source
        .clone()
        .try_map_canonical_storage(|mut values| {
            values.fill(ProjectionScalar {
                real: 1e200,
                imaginary: 0.0,
            });
            values
        })
        .unwrap();
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
            uniqueness: None,
            preconditioner: crate::contracts::SolverPreconditioner::None,
            relative_tolerance: 0.99,
            restart: 8,
            maximum_iterations: 32,
        };
        let coarse = reduced::solve(&assembly, policy, None).unwrap();
        let coarse_power = power::evaluate(
            &assembly,
            coarse.real.canonical_values(),
            coarse.imaginary.canonical_values(),
        )
        .unwrap();
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
            assert!((actual.real.canonical_values()[i] - expected[0]).abs() / amplitude < 1e-9);
            assert!(
                (actual.imaginary.canonical_values()[i] - expected[1]).abs() / amplitude < 1e-9
            );
        }
    }
}

#[test]
fn power_refinement_cannot_reset_or_enlarge_the_authored_iteration_budget() {
    let error = solve(
        &assembly(1.),
        AcousticHarmonicPolicy {
            uniqueness: None,
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
