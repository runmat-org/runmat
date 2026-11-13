//! Actual Steihaug CG with only the selected band preconditioner retained.
use super::*;
use trust_quadratic::cg;
pub(super) async fn step(
    geometry: &Geometry,
    jacobian: &Jacobian,
    metric: &trust_quadratic::bounds::Metric,
    radius: f64,
    bandwidth: usize,
    maximum: usize,
    tolerance: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<cg::Step>> {
    let products = jacobian.products(c).await?;
    let mut operator = normal::Normal::new(geometry, metric, products, c).await?;
    let Some(preconditioner) =
        cg::Preconditioner::new(&mut operator, geometry.axes.len(), bandwidth, c).await?
    else {
        return Ok(None);
    };
    cg::step(
        &mut operator,
        &preconditioner,
        &metric.gradient,
        radius,
        maximum,
        tolerance,
        c,
    )
    .await
}
