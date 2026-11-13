use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
fn matrix(values: [[f64; 3]; 3], scale: f64) -> RealSparseMatrix {
    let mut a = RealSparseAssembler::new(
        3,
        3,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for (r, row) in values.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            a.add(r, c, v * scale);
        }
    }
    a.finish().unwrap()
}
fn policy() -> GeneralSolvePolicy {
    GeneralSolvePolicy::bounded(1e-10)
}
// witness: FEA-SHARED-OPERATORS-001/operator_algorithm_contract @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Independent nonsymmetric and indefinite references preserve the general operator
//   path; SPD and complex cases are covered by sibling tests.
#[test]
fn independent_nonsymmetric_and_indefinite_references_preserve_general_operator() {
    for entries in [
        [[4., 2., -1.], [-3., 2., 1.], [1., -2., 5.]],
        [[-2., 1., 0.], [1., 3., 1.], [0., 1., -4.]],
    ] {
        for scale in [1e-80, 1., 1e80] {
            let a = matrix(entries, scale);
            let original = a.clone();
            let x = [1.5, -2., 0.25];
            let b = entries.map(|row| row.iter().zip(x).map(|(a, b)| a * b * scale).sum::<f64>());
            let result = solve_general_system(
                &a,
                &b,
                policy(),
                GeneralSolveEnvelope::derive(&a, policy()).unwrap(),
            )
            .unwrap();
            for (actual, reference) in result.solution.iter().zip(x) {
                assert!((actual - reference).abs() < 1e-9);
            }
            assert!(result.relative_residual < 1e-10);
            assert_eq!(
                result.certificate.operator_identity,
                a.numerical_identity().unwrap()
            );
            assert!(result.certificate.contraction_bound < 0.5);
            assert_eq!(a, original);
        }
    }
}
#[test]
fn zero_load_cannot_hide_an_unverified_nullspace() {
    let a = matrix([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]], 1.);
    assert!(matches!(
        solve_general_system(
            &a,
            &[0.; 3],
            policy(),
            GeneralSolveEnvelope::derive(&a, policy()).unwrap()
        ),
        Err(GeneralSolveError::NonsingularityNotEstablished(_))
    ));
    let a = matrix([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.]], 1.);
    let result = solve_general_system(
        &a,
        &[0.; 3],
        policy(),
        GeneralSolveEnvelope::derive(&a, policy()).unwrap(),
    )
    .unwrap();
    assert_eq!(result.solution, vec![0.; 3]);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.relative_residual, 0.);
}
#[test]
fn actual_counts_rederive_admission_and_both_budget_axes_reject() {
    let a = matrix([[4., 2., -1.], [-3., 2., 1.], [1., -2., 5.]], 1.);
    let required = GeneralSolveEnvelope::derive(&a, policy()).unwrap();
    assert_eq!(
        required,
        GeneralSolveEnvelope::for_counts(3, 9, policy()).unwrap()
    );
    for admitted in [
        GeneralSolveEnvelope {
            resident_bytes: required.resident_bytes - 1,
            ..required
        },
        GeneralSolveEnvelope {
            scalar_operations: required.scalar_operations - 1,
            ..required
        },
    ] {
        assert_eq!(
            solve_general_system(&a, &[1.; 3], policy(), admitted).unwrap_err(),
            GeneralSolveError::Admission
        );
    }
    assert!(GeneralSolveEnvelope::for_counts(3, 10, policy()).is_err());
    assert!(GeneralSolveEnvelope::for_counts(usize::MAX, usize::MAX, policy()).is_err());
}
#[test]
fn original_rhs_and_invalid_policies_fail_closed() {
    let a = matrix([[4., 2., -1.], [-3., 2., 1.], [1., -2., 5.]], 1.);
    let envelope = GeneralSolveEnvelope::derive(&a, policy()).unwrap();
    for rhs in [&[1., 2.][..], &[1., f64::NAN, 2.][..]] {
        assert_eq!(
            solve_general_system(&a, rhs, policy(), envelope).unwrap_err(),
            GeneralSolveError::Invalid
        );
    }
    for tolerance in [0., 1., -1., f64::NAN] {
        assert_eq!(
            GeneralSolveEnvelope::derive(
                &a,
                GeneralSolvePolicy {
                    relative_tolerance: tolerance,
                    ..policy()
                }
            ),
            Err(GeneralSolveError::Invalid)
        );
    }
    for limit in [0., 1., f64::NAN] {
        assert!(GeneralSolveEnvelope::derive(
            &a,
            GeneralSolvePolicy {
                contraction_limit: limit,
                ..policy()
            }
        )
        .is_err());
    }
    let mut encoded = serde_json::to_value(policy()).unwrap();
    encoded
        .as_object_mut()
        .unwrap()
        .insert("fallback".into(), true.into());
    assert!(serde_json::from_value::<GeneralSolvePolicy>(encoded).is_err());
}
#[test]
fn precancellation_is_typed_before_numerical_work() {
    let a = matrix([[4., 2., -1.], [-3., 2., 1.], [1., -2., 5.]], 1.);
    let envelope = GeneralSolveEnvelope::derive(&a, policy()).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            handler: None,
            cancellation: token,
            attempt_id: None,
        });
    assert_eq!(
        solve_general_system(&a, &[1.; 3], policy(), envelope).unwrap_err(),
        GeneralSolveError::Cancelled
    );
}

#[test]
fn ill_conditioned_operator_is_not_certified_and_returns_no_solution() {
    // Nonsingular in exact arithmetic but with condition ~1e17: the certificate's
    // n·ε·κ term exceeds the contraction limit, so no solution is returned.
    let a = matrix([[1., 1., 0.], [1., 1. + 1e-16, 0.], [0., 0., 1.]], 1.);
    assert!(matches!(
        solve_general_system(
            &a,
            &[1., 2., 3.],
            policy(),
            GeneralSolveEnvelope::derive(&a, policy()).unwrap()
        ),
        Err(GeneralSolveError::NonsingularityNotEstablished(_))
    ));
}

#[test]
fn pivoting_handles_a_zero_leading_diagonal() {
    // The retired unpivoted ILU could not produce a candidate here; sparse LU pivots.
    let a = matrix([[0., 1., 0.], [1., 0., 0.], [0., 0., 2.]], 1.);
    let result = solve_general_system(
        &a,
        &[1., 2., 3.],
        policy(),
        GeneralSolveEnvelope::derive(&a, policy()).unwrap(),
    )
    .unwrap();
    for (actual, expected) in result.solution.iter().zip([2., 1., 1.5]) {
        assert!((actual - expected).abs() < 1e-14);
    }
}
