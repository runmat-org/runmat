//! Unconstrained adapter to the shared cooperative trust-ball kernel.
use super::{model::Products, Controls, ScalarCooperation};
pub(super) use crate::builtins::math::optim::common::trust_quadratic::cg::{Preconditioner, Step};
use crate::BuiltinResult;
pub(super) async fn step(
    model: &mut Products<'_>,
    preconditioner: &Preconditioner,
    gradient: &[f64],
    radius: f64,
    controls: &Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Step>> {
    crate::builtins::math::optim::common::trust_quadratic::cg::step(
        model,
        preconditioner,
        gradient,
        radius,
        controls.maximum_cg_iterations,
        controls.cg_tolerance,
        cooperation,
    )
    .await
}
