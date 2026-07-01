#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityTolerance {
    pub abs: f64,
    pub rel: f64,
}

impl ParityTolerance {
    pub const fn strict() -> Self {
        Self {
            abs: 1e-12,
            rel: 1e-12,
        }
    }
}

pub fn assert_vectors_within_tolerance(left: &[f64], right: &[f64], tol: ParityTolerance) {
    assert_eq!(left.len(), right.len(), "vector length mismatch");
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        let diff = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let threshold = tol.abs.max(tol.rel * scale);
        assert!(
            diff <= threshold,
            "parity mismatch lhs={lhs} rhs={rhs} diff={diff} threshold={threshold}"
        );
    }
}
