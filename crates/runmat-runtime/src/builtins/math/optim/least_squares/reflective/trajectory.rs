//! A reflected original-point proposal and its unchanged local residual model.
use super::*;
pub(super) struct Proposal {
    pub x: Vec<f64>,
    pub predicted: f64,
    pub length: f64,
    pub step: f64,
}
pub(super) async fn propose(
    geometry: &Geometry,
    point: &initial::Seed,
    jacobian: &Jacobian,
    metric: &trust_quadratic::bounds::Metric,
    direction: &[f64],
    radius: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Proposal>> {
    let mut x = runmat_value::exact_capacity_vec(point.x.len()).map_err(allocation)?;
    for chunk in point.x.chunks(1024) {
        x.extend_from_slice(chunk);
        c.completed(chunk.len() as u64).await?;
    }
    let mut length = 0f64;
    let mut step = 0f64;
    for (column, (&axis, &p)) in geometry.axes.iter().zip(direction).enumerate() {
        let Some(mut value) = trust_quadratic::bounds::folded(
            point.x[axis],
            p,
            geometry.lower[column],
            geometry.upper[column],
        ) else {
            return Ok(None);
        };
        if value <= geometry.lower[column] {
            value = geometry.lower[column].next_up();
        }
        if value >= geometry.upper[column] {
            value = geometry.upper[column].next_down();
        }
        if !value.is_finite() || value <= geometry.lower[column] || value >= geometry.upper[column]
        {
            return Ok(None);
        }
        let delta = value - point.x[axis];
        x[axis] = value;
        step = step.max(delta.abs());
        length = length.hypot(delta / metric.scale[column]);
        c.completed(1).await?;
    }
    if step == 0.
        || !length.is_finite()
        || length > radius * (1. + 128. * f64::EPSILON * geometry.axes.len().max(1) as f64)
    {
        return Ok(None);
    }
    let mut delta = zeros_with(point.x.len(), c).await?;
    for ((to, &trial), &old) in delta.iter_mut().zip(&x).zip(&point.x) {
        *to = trial - old;
        c.completed(1).await?;
    }
    let mut product = zeros_with(point.residual.len(), c).await?;
    let mut products = jacobian.products(c).await?;
    if products
        .apply(
            crate::builtins::math::linalg::ops::linear_operator::ProductOrientation::Normal,
            &delta,
            &mut product,
            c,
        )
        .await?
        .is_none()
    {
        return Ok(None);
    }
    let mut linear = 0f64;
    let mut quadratic = 0f64;
    for (&r, &product) in point.residual.iter().zip(&product) {
        linear = r.mul_add(product, linear);
        quadratic = product.mul_add(product, quadratic);
        c.completed(1).await?;
    }
    let predicted = -(linear + 0.5 * quadratic);
    if !(predicted > 0. && predicted.is_finite()) {
        return Ok(None);
    }
    Ok(Some(Proposal {
        x,
        predicted,
        length,
        step,
    }))
}
