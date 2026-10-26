//! Effective stopping criteria. Legacy equation/curve callers retain their
//! own residual criterion; nonlinear least squares separates the three roles.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Criteria {
    pub function: f64,
    pub optimality: f64,
    pub step: f64,
}
impl LeastSquaresOptions {
    pub(super) fn criteria(&self, method: Method) -> Criteria {
        match self.criteria {
            Some(mut criteria) => {
                if method == Method::LevenbergMarquardt {
                    criteria.optimality = 1e-4 * criteria.function;
                }
                criteria
            }
            None => Criteria {
                function: self.tol_fun,
                optimality: self.tol_fun,
                step: self.tol_x,
            },
        }
    }
}
/// Bound-projected LM stationarity in original coordinates. Evaluate the
/// squared projected-gradient ratio without squaring a potentially huge norm.
pub(super) async fn lm_optimality(
    jacobian: &[f64],
    residual: &[f64],
    x: &[f64],
    bounds: &LeastSquaresBounds,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<f64> {
    let n = x.len();
    if residual.len().checked_mul(n) != Some(jacobian.len()) {
        return Err(optim_error(
            "lsqnonlin",
            "Jacobian geometry disagrees with accepted residual",
        ));
    }
    let mut projection = Projection::default();
    for (column, &coordinate) in x.iter().enumerate() {
        let mut gradient = 0f64;
        for (row, &value) in residual.iter().enumerate() {
            gradient = jacobian[row * n + column].mul_add(value, gradient);
            checkpoint(evaluator, 1).await?;
        }
        if projection
            .include(
                coordinate,
                gradient,
                bounds.lower[column],
                bounds.upper[column],
            )
            .is_none()
        {
            return Ok(f64::NAN);
        }
    }
    Ok(projection.measure())
}

#[derive(Default)]
struct Projection {
    gradient: f64,
    projected: f64,
}
impl Projection {
    fn include(&mut self, x: f64, gradient: f64, lower: f64, upper: f64) -> Option<()> {
        if !gradient.is_finite() {
            return None;
        }
        let distance = if gradient > 0. { x - lower } else { upper - x };
        let projected = gradient.abs().min(distance);
        if projected.is_nan() || projected < 0. {
            return None;
        }
        self.gradient = self.gradient.max(gradient.abs());
        self.projected = self.projected.max(projected);
        Some(())
    }
    fn measure(&self) -> f64 {
        if self.gradient == 0. {
            0.
        } else {
            (self.projected / self.gradient) * self.projected
        }
    }
}
pub(super) async fn from_gradient(
    gradient: &[f64],
    x: &[f64],
    bounds: &LeastSquaresBounds,
    c: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    if gradient.len() != x.len() || bounds.lower.len() != x.len() || bounds.upper.len() != x.len() {
        return Err(optim_error(
            "lsqnonlin",
            "accepted gradient geometry disagrees with point",
        ));
    }
    let mut projection = Projection::default();
    for (axis, (&x, &gradient)) in x.iter().zip(gradient).enumerate() {
        if projection
            .include(x, gradient, bounds.lower[axis], bounds.upper[axis])
            .is_none()
        {
            return Ok(f64::NAN);
        }
        c.completed(1).await?;
    }
    Ok(projection.measure())
}
