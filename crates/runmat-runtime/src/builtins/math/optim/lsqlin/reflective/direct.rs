//! Same rectangular QR/COD input; shared secular trust-ball selection.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares;
use trust_quadratic::residual::{self, ShiftedResidual};
mod augmented;
use augmented::Augmented;
pub(super) async fn step(
    model: &Model,
    point: &Point,
    metric: &trust_quadratic::bounds::Metric,
    radius: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut augmented = match Augmented::new(model, point, metric, c).await? {
        Some(value) => value,
        None => return Ok(None),
    };
    residual::step(&mut augmented, &metric.gradient, radius, c).await
}
impl ShiftedResidual for Augmented<'_> {
    async fn solve(
        &mut self,
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        Augmented::solve(self, shift, c).await
    }
}
