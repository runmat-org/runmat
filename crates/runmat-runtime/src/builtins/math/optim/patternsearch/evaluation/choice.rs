//! Single original-merit improvement comparison for poll and auxiliary search.
use super::Point;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(in crate::builtins::math::optim::patternsearch) async fn improving(
    point: Point,
    best_score: f64,
    target: f64,
    merit: &mut impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(Point, f64)>> {
    let Some(score) = merit(&point, cooperation).await? else {
        return Ok(None);
    };
    Ok(if score < best_score && score < target {
        Some((point, score))
    } else {
        None
    })
}
