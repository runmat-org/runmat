use super::*;

fn relation(primary: usize, secondary: usize, phase_radians: f64) -> UnitPhaseRelation {
    UnitPhaseRelation {
        primary,
        secondary,
        phase_radians: runmat_types::physical::Angle::new(phase_radians),
    }
}

fn scalar(real: f64, imaginary: f64) -> ProjectionScalar {
    ProjectionScalar { real, imaginary }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-14 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn free_unit_phase_recovery_and_restriction_use_conjugate_transpose() {
    for phase in [
        0.0,
        std::f64::consts::FRAC_PI_2,
        -std::f64::consts::FRAC_PI_2,
        0.731,
    ] {
        let map = build_phase_reduction(3, &[relation(0, 1, phase)], &[None; 3]).unwrap();
        assert_eq!(map.free_dof_indices, [0, 2, 3, 5]);
        assert_eq!(map.pivot_dof_indices, [1, 4]);
        assert_eq!(map.reduced_dof_count, 4);
        // The representative and unrelated node remain genuinely free.
        let recovered = map.expand(&[2.0, 5.0, -3.0, 7.0]).unwrap();
        let (sine, cosine) = phase.sin_cos();
        let expected = [
            2.0,
            2.0 * cosine + 3.0 * sine,
            5.0,
            -3.0,
            2.0 * sine - 3.0 * cosine,
            7.0,
        ];
        for (actual, expected) in recovered.iter().zip(expected) {
            close(*actual, expected);
        }
        let restricted = map
            .transpose_apply(&[11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
            .unwrap();
        // conj(exp(i phi)) * (13 + 23i), independently in real coordinates.
        let expected = [
            11.0 + 13.0 * cosine + 23.0 * sine,
            17.0,
            19.0 - 13.0 * sine + 23.0 * cosine,
            29.0,
        ];
        for (actual, expected) in restricted.iter().zip(expected) {
            close(*actual, expected);
        }
        assert!(map
            .recovery_row_offsets
            .windows(2)
            .all(|range| range[1] - range[0] <= 2));
        map.validate().unwrap();
    }
}

#[test]
fn arbitrary_free_phase_operator_reduction_matches_independent_dense_congruence() {
    let phase: f64 = -0.382;
    let map = build_phase_reduction(2, &[relation(0, 1, phase)], &[None; 2]).unwrap();
    let full = [
        [4.0, 1.0, -2.0, 3.0],
        [1.0, 5.0, 2.0, -1.0],
        [-2.0, 2.0, 6.0, 1.0],
        [3.0, -1.0, 1.0, 7.0],
    ];
    let (sine, cosine) = phase.sin_cos();
    let transform = [[1.0, 0.0], [cosine, -sine], [0.0, 1.0], [sine, cosine]];
    let flattened: Vec<_> = full.iter().flatten().copied().collect();
    let (reduced, _) = map.reduce_dense_operator(&flattened, &[0.0; 4]).unwrap();
    for row in 0..2 {
        for column in 0..2 {
            let expected: f64 = (0..4)
                .flat_map(|i| (0..4).map(move |j| (i, j)))
                .map(|(i, j)| transform[i][row] * full[i][j] * transform[j][column])
                .sum();
            close(reduced[row * 2 + column], expected);
        }
    }
}

#[test]
fn canonical_components_are_independent_of_relation_order_and_direction() {
    let forward = [
        relation(2, 1, 0.3),
        relation(1, 0, 0.2),
        relation(0, 2, -0.5),
    ];
    let reversed = [
        relation(2, 0, 0.5),
        relation(0, 1, -0.2),
        relation(1, 2, -0.3),
    ];
    let left = build_phase_reduction(3, &forward, &[None; 3]).unwrap();
    let right = build_phase_reduction(3, &reversed, &[None; 3]).unwrap();
    assert_eq!(left.free_dof_indices, [0, 3]);
    assert_eq!(left.recovery_row_offsets, right.recovery_row_offsets);
    assert_eq!(left.recovery_column_indices, right.recovery_column_indices);
    for (actual, expected) in left
        .recovery_coefficients
        .iter()
        .zip(right.recovery_coefficients)
    {
        close(*actual, expected);
    }
    let redundant = [
        relation(0, 1, 0.25),
        relation(0, 1, 0.25),
        relation(1, 1, 0.0),
    ];
    assert_eq!(
        build_phase_reduction(2, &redundant, &[None; 2]).unwrap(),
        build_phase_reduction(2, &[redundant[0]], &[None; 2]).unwrap()
    );
}

#[test]
fn cycles_and_scaled_prescriptions_fail_closed_without_a_dimensional_floor() {
    assert!(matches!(
        build_phase_reduction(
            3,
            &[
                relation(0, 1, 0.2),
                relation(1, 2, 0.3),
                relation(0, 2, 0.6)
            ],
            &[None; 3]
        ),
        Err(UnitPhaseConstraintError::InconsistentCycle { .. })
    ));
    assert!(matches!(
        build_phase_reduction(1, &[relation(0, 0, 0.1)], &[None]),
        Err(UnitPhaseConstraintError::InconsistentCycle { .. })
    ));
    for amplitude in [1e-200, 1e-20, 1.0, 1e200] {
        let relations = [
            relation(0, 1, std::f64::consts::FRAC_PI_2),
            relation(1, 2, -std::f64::consts::FRAC_PI_2),
        ];
        let prescriptions = [
            None,
            Some(scalar(3.0 * amplitude, 2.0 * amplitude)),
            Some(scalar(2.0 * amplitude, -3.0 * amplitude)),
        ];
        let map = build_phase_reduction(3, &relations, &prescriptions).unwrap();
        assert_eq!(map.reduced_dof_count, 0);
        let expected = [2.0, 3.0, 2.0, -3.0, 2.0, -3.0];
        for (actual, expected) in map.expand(&[]).unwrap().iter().zip(expected) {
            close(*actual / amplitude, expected);
        }
        let conflict = [
            Some(scalar(amplitude, 0.0)),
            Some(scalar(0.0, 1.01 * amplitude)),
            None,
        ];
        assert!(matches!(
            build_phase_reduction(3, &relations, &conflict),
            Err(UnitPhaseConstraintError::ConflictingPrescription { .. })
        ));
        // A tiny nonzero value cannot be identified with exactly zero.
        let zero_conflict = [Some(scalar(0.0, 0.0)), Some(scalar(0.0, amplitude)), None];
        assert!(matches!(
            build_phase_reduction(3, &relations, &zero_conflict),
            Err(UnitPhaseConstraintError::ConflictingPrescription { .. })
        ));
    }
}

#[test]
fn invalid_dimensions_indices_and_nonfinite_inputs_are_rejected() {
    assert!(matches!(
        build_phase_reduction(1, &[], &[]),
        Err(UnitPhaseConstraintError::InvalidDimensions)
    ));
    for relation in [
        relation(0, 1, 0.0),
        relation(0, 0, f64::NAN),
        relation(0, 0, f64::INFINITY),
    ] {
        assert!(matches!(
            build_phase_reduction(1, &[relation], &[None]),
            Err(UnitPhaseConstraintError::InvalidRelation { .. })
        ));
    }
    assert!(matches!(
        build_phase_reduction(1, &[], &[Some(scalar(f64::INFINITY, 0.0))]),
        Err(UnitPhaseConstraintError::InvalidPrescription { .. })
    ));
    let empty = build_phase_reduction(0, &[], &[]).unwrap();
    assert_eq!(empty.full_dof_count, 0);
    assert_eq!(empty.recovery_row_offsets, [0]);
    empty.validate().unwrap();
}
