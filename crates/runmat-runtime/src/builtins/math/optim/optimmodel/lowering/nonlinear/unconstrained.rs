//! Original model scalar probes through the actual unconstrained controller.
use super::*;
use crate::builtins::math::optim::fminunc::compiled as unconstrained;
pub(super) async fn solve(
    model: &Model,
    formulation: &formulation::Formulation,
    starting: Option<&Value>,
    options: Option<&Value>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<unconstrained::Outcome> {
    let options = unconstrained::prepare(options, model.layout.columns, c).await?;
    let starting = starting.ok_or_else(|| policy.invalid(
        "an unconstrained nonlinear model requires an original variable-name initial-point structure",
    ))?;
    let initial = layout::initial_point(&model.source, &model.layout, starting, policy, c).await?;
    let mut oracle = oracle::Original {
        model,
        formulation,
        policy,
    };
    let outcome = unconstrained::solve(initial, options, &mut oracle, c)?.await;
    outcome
}
