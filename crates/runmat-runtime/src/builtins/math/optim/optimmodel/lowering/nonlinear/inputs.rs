//! Preflight every original starting field before conversion; preserve bounds.
use super::*;
pub(super) async fn prepare(
    model: &Model,
    starting: Option<&Value>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<compiled::Problem> {
    let starting = starting.ok_or_else(|| {
        policy.invalid(
            "a nonlinear model requires an initial-point structure of original variable values",
        )
    })?;
    let crate::builtins::math::optim::least_squares::LeastSquaresBounds { lower, upper } =
        layout::continuous_bounds(&model.layout, policy, c).await?;
    let x = layout::initial_point(&model.source, &model.layout, starting, policy, c).await?;
    Ok(compiled::Problem {
        x,
        lower,
        upper,
        inequalities: model.inequalities,
        equalities: model.equalities,
    })
}
