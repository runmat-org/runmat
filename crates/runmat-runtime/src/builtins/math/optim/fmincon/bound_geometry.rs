//! Shared representable strict-bound initialization; fixed bounds retain their exact value.
use super::model::Geometry;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(super) async fn interior_start(
    x: &mut [f64],
    geometry: &Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for (index, value) in x.iter_mut().enumerate() {
        let lower = geometry.lower[index];
        let upper = geometry.upper[index];
        if lower == upper {
            *value = lower;
            cooperation.completed(1).await?;
            continue;
        }
        if *value <= lower {
            let candidate = if upper.is_finite() {
                0.5 * lower + 0.5 * upper
            } else {
                lower + 0.01 * lower.abs().max(1.0)
            };
            *value = if candidate > lower && candidate < upper && candidate.is_finite() {
                candidate
            } else {
                lower.next_up()
            };
        }
        if *value >= upper {
            let candidate = if lower.is_finite() {
                0.5 * lower + 0.5 * upper
            } else {
                upper - 0.01 * upper.abs().max(1.0)
            };
            *value = if candidate > lower && candidate < upper && candidate.is_finite() {
                candidate
            } else {
                upper.next_down()
            };
        }
        if !value.is_finite() || *value <= lower || *value >= upper {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
