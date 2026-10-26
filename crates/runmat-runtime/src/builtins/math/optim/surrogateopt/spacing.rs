//! Final modeled-refinement spacing in the actual full normalized box.
use super::{domain::BoxView, invalid, rbf::Training};
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};
pub(super) async fn admissible(
    point: &[f64],
    domain: &BoxView<'_>,
    evaluated: &impl Training,
    minimum: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if !evaluated.valid()
        || evaluated.dimensions() != domain.dimensions()
        || point.len() != domain.variables()
        || !minimum.is_finite()
        || minimum < 0.0
    {
        return Err(invalid("invalid refined-point spacing geometry"));
    }
    for axis in 0..domain.dimensions() {
        let value = domain.unit_coordinate(point, axis);
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    if minimum == 0.0 {
        return Ok(true);
    }
    for row in 0..evaluated.rows().len() {
        let mut distance = 0.0_f64;
        for axis in 0..domain.dimensions() {
            distance = distance
                .hypot(domain.unit_coordinate(point, axis) - evaluated.coordinate(row, axis));
            cooperation.completed(1).await?;
        }
        if distance < minimum {
            return Ok(false);
        }
    }
    Ok(true)
}
