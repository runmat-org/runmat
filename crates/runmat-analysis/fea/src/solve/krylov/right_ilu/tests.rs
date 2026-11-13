use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, SparsePattern};
use crate::solve::krylov::{restarted_gmres, GmresPolicy};
use std::cell::Cell;

fn full(values: &[f64], n: usize) -> RealSparseMatrix {
    RealSparseMatrix::try_new(
        SparsePattern::try_new(
            n as u64,
            n as u64,
            (0..=n).map(|row| (row * n) as u64).collect(),
            (0..n)
                .flat_map(|_| (0..n).map(|column| column as u64))
                .collect(),
        )
        .unwrap(),
        values.to_vec(),
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
        Vec::new(),
    )
    .unwrap()
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-13 * expected.abs().max(1.0),
        "actual={actual} expected={expected}"
    );
}

#[test]
fn signed_nonsymmetric_lu_matches_independent_literal_factors_not_ldlt() {
    // Independently multiply L=[[1,0,0],[2,1,0],[-1,1/2,1]] and
    // U=[[-2,3,1],[0,4,-2],[0,0,-5]]. Explicit zero support is retained.
    let a = full(&[-2., 3., 1., -4., 10., 0., 2., -1., -7.], 3);
    let ilu = SparseRightIlu::factor(&a, || false).unwrap();
    assert_eq!(ilu.factors(), vec![-2., 3., 1., 2., 4., -2., -1., 0.5, -5.]);
    let actual = ilu.apply(&[-5., -24., -17.], || false).unwrap();
    for (value, expected) in actual.into_iter().zip([1., -2., 3.]) {
        near(value, expected);
    }
    // L D L^T gives a different first column [−2,−4,2] but second row
    // [−4,−4,6], rather than A's [−4,10,0]; it cannot solve this RHS.
    let ldlt_second_row_applied = -4. * 1. + -4. * -2. + 6. * 3.;
    assert_ne!(ldlt_second_row_applied, -24.);
}

#[test]
fn missing_fill_is_dropped_and_original_csr_support_preserved() {
    // Eliminating (2,0) would create (2,1)=−1/4, which is absent and must drop.
    let a = RealSparseMatrix::try_new(
        SparsePattern::try_new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 0, 2]).unwrap(),
        vec![4., 1., 1., 4., 1., 1., 4.],
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
        Vec::new(),
    )
    .unwrap();
    let original = a.clone();
    let ilu = SparseRightIlu::factor(&a, || false).unwrap();
    assert_eq!(a, original);
    assert_eq!(ilu.row_offsets(), vec![0, 2, 5, 7]);
    assert_eq!(ilu.column_indices(), vec![0, 1, 0, 1, 2, 0, 2]);
    assert_eq!(ilu.factors(), vec![4., 1., 0.25, 3.75, 1., 0.25, 4.]);
    let answer = ilu.apply(&[3., 7., 9.], || false).unwrap();
    // Literal forward substitution y=[3,25/4,33/4], followed by true U solve.
    let x2 = 33. / 16.;
    let x1 = (25. / 4. - x2) / (15. / 4.);
    let x0 = (3. - x1) / 4.;
    for (value, expected) in answer.into_iter().zip([x0, x1, x2]) {
        near(value, expected);
    }
}

#[test]
fn signed_tiny_and_large_pivots_have_no_dimensional_floor() {
    for scale in [1e-200, 1e-20, 1., 1e20, 1e200] {
        let a = RealSparseMatrix::diagonal(
            vec![scale, -2. * scale],
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )
        .unwrap();
        let ilu = SparseRightIlu::factor(&a, || false).unwrap();
        let answer = ilu.apply(&[3. * scale, 8. * scale], || false).unwrap();
        near(answer[0], 3.);
        near(answer[1], -4.);
    }
    let empty = full(&[], 0);
    assert_eq!(
        SparseRightIlu::factor(&empty, || false)
            .unwrap()
            .apply(&[], || false)
            .unwrap(),
        Vec::<f64>::new()
    );
}

#[test]
fn invalid_support_pivots_and_nonfinite_factor_arithmetic_fail_closed() {
    let rectangular = RealSparseMatrix::try_new(
        SparsePattern::try_new(1, 2, vec![0, 1], vec![0]).unwrap(),
        vec![1.],
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        SparseRightIlu::factor(&rectangular, || false),
        Err(SparseRightIluError::NonSquare)
    );
    let missing = RealSparseMatrix::try_new(
        SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 0]).unwrap(),
        vec![1., 1.],
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        SparseRightIlu::factor(&missing, || false),
        Err(SparseRightIluError::MissingDiagonal { row: 1 })
    );
    for value in [0., -0., f64::from_bits(1)] {
        assert_eq!(
            SparseRightIlu::factor(&full(&[value], 1), || false),
            Err(SparseRightIluError::InvalidPivot { row: 0 })
        );
    }
    assert_eq!(
        SparseRightIlu::factor(&full(&[1., 1., 1., 1.], 2), || false),
        Err(SparseRightIluError::InvalidPivot { row: 1 })
    );
    // Nonfinite source values are already impossible for validated CSR input.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(RealSparseMatrix::diagonal(
            vec![value],
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL
        )
        .is_err());
    }
    for entries in [
        [1e-200, 1., 1e200, 1.],
        [1., 1e308, 2., 1.],
        [1., 1e308, 1., -1e308],
    ] {
        assert!(matches!(
            SparseRightIlu::factor(&full(&entries, 2), || false),
            Err(SparseRightIluError::NonfiniteFactor { .. })
        ));
    }
}

#[test]
fn application_dimensions_nonfinite_overflow_and_cancellation_fail_closed() {
    let a = full(&[1., 2., 3., 5.], 2);
    let ilu = SparseRightIlu::factor(&a, || false).unwrap();
    assert_eq!(
        ilu.apply(&[1.], || false),
        Err(SparseRightIluError::DimensionMismatch)
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            ilu.apply(&[1., value], || false),
            Err(SparseRightIluError::NonfiniteInput { row: 1 })
        );
    }
    let tiny = SparseRightIlu::factor(&full(&[1e-200], 1), || false).unwrap();
    assert_eq!(
        tiny.apply(&[1e200], || false),
        Err(SparseRightIluError::NonfiniteOutput { row: 0 })
    );
    let lower = SparseRightIlu::factor(&full(&[1., 0., 1e308, 1.], 2), || false).unwrap();
    assert_eq!(
        lower.apply(&[2., 1.], || false),
        Err(SparseRightIluError::NonfiniteOutput { row: 0 })
    );
    let upper = SparseRightIlu::factor(&full(&[1., 1e308, 0., 1.], 2), || false).unwrap();
    assert_eq!(
        upper.apply(&[1., 2.], || false),
        Err(SparseRightIluError::NonfiniteOutput { row: 0 })
    );
    assert_eq!(
        SparseRightIlu::factor(&a, || true),
        Err(SparseRightIluError::Cancelled)
    );
    assert_eq!(
        ilu.apply(&[1., 2.], || true),
        Err(SparseRightIluError::Cancelled)
    );
    // Cancellation is observed after the factorization and the application, not only at entry.
    for limit in [1] {
        let calls = Cell::new(0);
        assert_eq!(
            SparseRightIlu::factor(&a, || {
                calls.set(calls.get() + 1);
                calls.get() > limit
            }),
            Err(SparseRightIluError::Cancelled)
        );
    }
    for limit in [1] {
        let calls = Cell::new(0);
        assert_eq!(
            ilu.apply(&[1., 2.], || {
                calls.set(calls.get() + 1);
                calls.get() > limit
            }),
            Err(SparseRightIluError::Cancelled)
        );
    }
}

#[test]
fn right_gmres_recovers_original_solution_and_residual_across_load_scales() {
    let a = full(&[-2., 3., 1., -4., 10., 0., 2., -1., -7.], 3);
    let ilu = SparseRightIlu::factor(&a, || false).unwrap();
    for scale in [1e-100, 1e-12, 1., 1e12, 1e100] {
        let rhs = [-5. * scale, -24. * scale, -17. * scale];
        let apply = |x: &[f64]| -> Result<Vec<f64>, SparseRightIluError> {
            Ok(vec![
                -2. * x[0] + 3. * x[1] + x[2],
                -4. * x[0] + 10. * x[1],
                2. * x[0] - x[1] - 7. * x[2],
            ])
        };
        let result = restarted_gmres(
            &rhs,
            &[0.; 3],
            GmresPolicy {
                relative_tolerance: 1e-12,
                restart: 1,
                maximum_iterations: 4,
            },
            |y| apply(&ilu.apply(y, || false)?),
            || false,
        )
        .unwrap()
        .unwrap();
        assert!(result.converged && !result.breakdown && !result.cancelled);
        assert_eq!(result.iterations, 1);
        let x = ilu.apply(&result.solution, || false).unwrap();
        for (&value, expected) in x.iter().zip([1., -2., 3.]) {
            near(value / scale, expected);
        }
        let ax = apply(&x).unwrap();
        let relative = rhs
            .iter()
            .zip(ax)
            .map(|(b, ax)| ((b - ax) / scale).powi(2))
            .sum::<f64>()
            .sqrt()
            / 890.0_f64.sqrt();
        assert!(relative <= 1e-12);
    }
}
