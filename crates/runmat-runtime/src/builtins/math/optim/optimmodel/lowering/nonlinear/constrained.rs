//! Same constrained controller and original scalar oracle.
use super::*;
pub(super) async fn solve(
    model: &Model,
    formulation: &formulation::Formulation,
    starting: Option<&Value>,
    options: Option<&Value>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<compiled::Outcome> {
    let options =
        compiled::prepare_options(options, model.inequalities != 0 || model.equalities != 0, c)
            .await?;
    let problem = inputs::prepare(model, starting, policy, c).await?;
    let mut oracle = oracle::Original {
        model,
        formulation,
        policy,
    };
    let numeric = crate::builtins::math::optim::common::constrained_numeric::Policy {
        owner: policy.identity,
        invalid: Some(policy.invalid),
        numeric: None,
        single: None,
        resident: None,
        sparse: None,
    };
    let outcome = compiled::solve(problem, options, &mut oracle, numeric, c)?.await;
    outcome
}
