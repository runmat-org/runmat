//! Componentwise original-operator backward-error scale, not a convergence
//! tolerance or a Hessian inferred from cancelled computed products.
#[derive(Clone, Copy)]
pub(crate) struct StationarityScale {
    pub rhs: f64,
    pub curvature: f64,
    pub diagonal: f64,
}
impl StationarityScale {
    /// Each bound is formed from absolute original rectangular contributions.
    /// The maximum is stricter than their sum and avoids an overflowing
    /// addition. A nonrepresentable formation bound is never an infinite
    /// denominator that could make an arbitrary residual pass.
    pub(crate) fn qualifies(self, gradient: f64, product: f64, shifted: f64, n: usize) -> bool {
        if !gradient.is_finite()
            || !product.is_finite()
            || !shifted.is_finite()
            || [self.rhs, self.curvature, self.diagonal]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.)
        {
            return false;
        }
        let scale = gradient
            .abs()
            .max(product.abs())
            .max(shifted.abs())
            .max(self.rhs)
            .max(self.curvature)
            .max(self.diagonal);
        scale == 0.
            || (gradient / scale + product / scale + shifted / scale).abs() <= super::tolerance(n)
    }
}
