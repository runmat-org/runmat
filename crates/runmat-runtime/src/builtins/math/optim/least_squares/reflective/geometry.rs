//! Original fixed coordinates are excluded from the numerical trust subspace.
use super::*;
pub(super) struct Geometry {
    pub(super) axes: Vec<usize>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) variables: usize,
}
pub(super) struct Frame {
    pub(super) x: Vec<f64>,
    pub(super) gradient: Vec<f64>,
    pub(super) optimality: f64,
}
impl Geometry {
    pub(super) async fn new(
        bounds: &LeastSquaresBounds,
        evaluator: &mut impl LeastSquaresEvaluator,
    ) -> BuiltinResult<Self> {
        let mut axes = runmat_value::exact_capacity_vec(bounds.lower.len()).map_err(allocation)?;
        let mut lower = runmat_value::exact_capacity_vec(bounds.lower.len()).map_err(allocation)?;
        let mut upper = runmat_value::exact_capacity_vec(bounds.lower.len()).map_err(allocation)?;
        for (i, (&l, &u)) in bounds.lower.iter().zip(&bounds.upper).enumerate() {
            if l < u {
                axes.push(i);
                lower.push(l);
                upper.push(u);
            }
            checkpoint(evaluator, 1).await?;
        }
        Ok(Self {
            axes,
            lower,
            upper,
            variables: bounds.lower.len(),
        })
    }
    pub(super) async fn frame(
        &self,
        x: &[f64],
        residual: &[f64],
        products: &mut derivative::Products<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Frame>> {
        if x.len() != self.variables || products.dimensions() != (residual.len(), self.variables) {
            return Err(invalid(
                "original Jacobian geometry disagrees with reflective point".into(),
            ));
        }
        let mut full_gradient = zeros_with(self.variables, c).await?;
        if products
            .apply(
                crate::builtins::math::linalg::ops::linear_operator::ProductOrientation::Adjoint,
                residual,
                &mut full_gradient,
                c,
            )
            .await?
            .is_none()
        {
            return Ok(None);
        }
        self.frame_from_gradient(x, &full_gradient, c).await
    }
    pub(super) async fn frame_from_gradient(
        &self,
        x: &[f64],
        full_gradient: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Frame>> {
        if x.len() != self.variables || full_gradient.len() != self.variables {
            return Err(invalid(
                "accepted original gradient geometry disagrees".into(),
            ));
        }
        let mut selected = zeros_with(self.axes.len(), c).await?;
        let mut gradient = zeros_with(self.axes.len(), c).await?;
        let mut optimality = 0f64;
        for (column, &axis) in self.axes.iter().enumerate() {
            selected[column] = x[axis];
            let g = full_gradient[axis];
            let distance = if g < 0. && self.upper[column].is_finite() {
                self.upper[column] - x[axis]
            } else if g >= 0. && self.lower[column].is_finite() {
                x[axis] - self.lower[column]
            } else {
                1.
            };
            let measure = (distance * g).abs();
            if !g.is_finite() || !measure.is_finite() {
                return Ok(None);
            }
            gradient[column] = g;
            optimality = optimality.max(measure);
            c.completed(1).await?;
        }
        Ok(Some(Frame {
            x: selected,
            gradient,
            optimality,
        }))
    }
    pub(super) async fn metric(
        &self,
        frame: &Frame,
        options: &LeastSquaresOptions,
        evaluator: &mut impl LeastSquaresEvaluator,
    ) -> BuiltinResult<Option<trust_quadratic::bounds::Metric>> {
        trust_quadratic::bounds::Metric::new(
            &frame.x,
            &frame.gradient,
            trust_quadratic::bounds::Bounds {
                lower: &self.lower,
                upper: &self.upper,
            },
            |i| options.scaling.typical_value(self.axes[i]),
            policy(),
            cooperation(evaluator)?,
        )
        .await
    }
}
