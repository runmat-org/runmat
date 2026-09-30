use super::*;
use crate::solve::krylov::{restarted_gmres, GmresPolicy};

fn dot(left: &[f64], right: &[f64]) -> Result<f64, DiagonalRightPreconditionerError> {
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum())
}

#[test]
fn indefinite_nonsymmetric_right_scaling_preserves_original_residual_acceptance() {
    // A = B diag(d), B = [[1,e,0], [0,1,-e], [e,0,1]].
    // Negative d[1] makes A indefinite; its unequal off-diagonals are nonsymmetric.
    // All reference algebra below is literal, independent of the scaling implementation.
    let diagonal = [1e-10, -1.0, 1e10];
    let coupling = 1e-3;
    let apply = |input: &[f64]| {
        Ok::<_, DiagonalRightPreconditionerError>(vec![
            diagonal[0] * input[0] + coupling * diagonal[1] * input[1],
            diagonal[1] * input[1] - coupling * diagonal[2] * input[2],
            coupling * diagonal[0] * input[0] + diagonal[2] * input[2],
        ])
    };
    let preconditioner = DiagonalRightPreconditioner::new(&diagonal).unwrap();
    let policy = GmresPolicy {
        relative_tolerance: 1e-10,
        restart: 1,
        maximum_iterations: 8,
    };
    for amplitude in [1e-100, 1e-12, 1.0, 1e12, 1e100] {
        let rhs = [amplitude, 2.0 * amplitude, -amplitude];
        let original = restarted_gmres(&rhs, &[0.0; 3], policy, apply, dot, || false)
            .unwrap()
            .unwrap();
        assert!(
            !original.converged,
            "unscaled restart-one system must expose stagnation"
        );
        let conditioned = restarted_gmres(
            &rhs,
            &[0.0; 3],
            policy,
            |input| apply(&preconditioner.apply(input)?),
            dot,
            || false,
        )
        .unwrap()
        .unwrap();
        assert!(
            conditioned.converged,
            "amplitude={amplitude}: {conditioned:?}"
        );
        assert!(!conditioned.breakdown && !conditioned.cancelled);
        assert!(conditioned.iterations > 0 && conditioned.iterations < 8);
        let physical = preconditioner.apply(&conditioned.solution).unwrap();
        let applied = apply(&physical).unwrap();
        let residual = rhs
            .iter()
            .zip(&applied)
            .map(|(rhs, applied)| ((rhs - applied) / amplitude).powi(2))
            .sum::<f64>()
            .sqrt();
        let reference_norm = 6.0_f64.sqrt();
        assert!(residual / reference_norm <= policy.relative_tolerance);
        assert!((conditioned.residual_norm / amplitude - residual).abs() <= 1e-13);
        // Solve B y = [1,2,-1] by direct substitution, then x = diag(d)^-1 y.
        let y0 =
            (1.0 - 2.0 * coupling + coupling * coupling) / (1.0 - coupling * coupling * coupling);
        let y2 = -1.0 - coupling * y0;
        let y1 = 2.0 + coupling * y2;
        let expected = [y0 / diagonal[0], y1 / diagonal[1], y2 / diagonal[2]];
        for (actual, expected) in physical.iter().zip(expected) {
            assert!((actual / amplitude / expected - 1.0).abs() < 3e-10);
        }
    }
}

#[test]
fn finite_negative_and_tiny_diagonals_have_no_unit_sized_floor() {
    for scale in [1e-200, 1e-20, 1.0, 1e20, 1e200] {
        let map = DiagonalRightPreconditioner::new(&[scale, -2.0 * scale]).unwrap();
        assert!(map
            .apply(&[scale, -2.0 * scale])
            .unwrap()
            .iter()
            .all(|value| (value - 1.0).abs() < 1e-14));
        let recovered = map.apply(&[3.0 * scale, 8.0 * scale]).unwrap();
        assert!((recovered[0] - 3.0).abs() < 1e-14);
        assert!((recovered[1] + 4.0).abs() < 1e-14);
    }
    assert_eq!(
        DiagonalRightPreconditioner::new(&[])
            .unwrap()
            .apply(&[])
            .unwrap(),
        Vec::<f64>::new()
    );
}

#[test]
fn noninvertible_diagonals_and_nonfinite_vector_operations_fail_closed() {
    for invalid in [
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(1),
    ] {
        assert_eq!(
            DiagonalRightPreconditioner::new(&[1.0, invalid]),
            Err(DiagonalRightPreconditionerError::InvalidDiagonal { row: 1 })
        );
    }
    let map = DiagonalRightPreconditioner::new(&[1.0, 1e-200]).unwrap();
    assert_eq!(
        map.apply(&[1.0]),
        Err(DiagonalRightPreconditionerError::DimensionMismatch)
    );
    assert_eq!(
        map.apply(&[f64::NAN, 1.0]),
        Err(DiagonalRightPreconditionerError::NonfiniteInput { row: 0 })
    );
    assert_eq!(
        map.apply(&[1.0, f64::INFINITY]),
        Err(DiagonalRightPreconditionerError::NonfiniteInput { row: 1 })
    );
    assert_eq!(
        map.apply(&[1.0, 1e200]),
        Err(DiagonalRightPreconditionerError::NonfiniteOutput { row: 1 })
    );
}
