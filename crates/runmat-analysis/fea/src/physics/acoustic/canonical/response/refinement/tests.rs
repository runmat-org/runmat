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
        volume_stiffness: matrix([1., 2., 3., 4.], OperatorRole::Stiffness),
        mass: matrix([0.; 4], OperatorRole::Mass),
        impedance: matrix([0.1, 0.5, 0.8, 1.], OperatorRole::Damping),
        volume_admittance: matrix([0.; 4], OperatorRole::Damping),
        lossy: false,
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
        port_faces: Vec::new(),
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
fn certified_direct_solve_matches_complex_division_and_closes_power() {
    for amplitude in [1., 1e-12] {
        let assembly = assembly(amplitude);
        let policy = AcousticHarmonicPolicy::default();
        let (actual, closure) = solve(&assembly, policy, 1e-10).unwrap();
        assert!(actual.relative_residual <= policy.relative_tolerance);
        assert!(closure.power_relative_error <= 1e-10);
        assert!(actual.certificate.contraction_bound < actual.certificate.limit);
        assert_eq!(actual.certificate.dimension, 8);
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
fn singular_operator_fails_closed_and_invalid_power_tolerances_are_rejected() {
    // A lossless rigid cavity at a resonance: K − ω²M singular in one mode.
    let mut singular = assembly(1.);
    singular.impedance = crate::physics::acoustic::canonical::quantity_layout::pressure_operator(
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
    let mut stiffness = RealSparseAssembler::square(
        4,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    for (row, value) in [1., 2., 0., 4.].into_iter().enumerate() {
        stiffness.add(row, row, value);
    }
    singular.stiffness = crate::physics::acoustic::canonical::quantity_layout::pressure_operator(
        stiffness.finish().unwrap(),
        2,
    )
    .unwrap();
    assert!(matches!(
        solve(&singular, AcousticHarmonicPolicy::default(), 1e-10),
        Err(AcousticHarmonicError::Certificate(_))
    ));
    for tolerance in [0., -1., f64::NAN, f64::INFINITY, 1.] {
        assert!(solve(&assembly(1.), AcousticHarmonicPolicy::default(), tolerance).is_err());
    }
}
