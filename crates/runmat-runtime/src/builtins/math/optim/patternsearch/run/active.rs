//! Invocation controller and terminal publication after preparation retires.
use super::*;
pub(super) async fn run(
    input: &Input<'_>,
    ready: Ready,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ResultTuple> {
    let Ready {
        engine,
        options,
        linear,
        domain,
        policy,
        shape,
        scalar,
        mut state,
        mut deadline,
        mut observers,
        rng,
    } = ready;
    let mut evaluator = evaluation::Evaluator {
        context: evaluation::Context {
            execution: None,
            demand: evaluation::ObservationDemand::MaySkipVisited,
            objective: input.objective,
            constraints: input.constraints,
            shape: &shape,
            scalar,
            policy,
        },
        state: &mut state,
        cooperation,
    };
    // One exact-layout owner for the invocation's controller, not one per
    // evaluation. Its concrete search/observer alternatives do not inflate
    // the preparation and canonical builtin forwarding poll frames.
    let runtime = evaluator.cooperation.runtime().cloned();
    let (engine, reason, options) =
        crate::execution::rooted_continuation::admit_optional_hostcall_with_contextual_error(
            runtime.as_ref(),
            || {
                controller::drive(
                    engine,
                    options,
                    &linear,
                    &domain,
                    input,
                    &mut evaluator,
                    &mut deadline,
                    &mut observers,
                )
            },
        )?
        .await?;
    let violation = match linear
        .violation(&engine.point.x, evaluator.cooperation)
        .await?
    {
        Some(linear) => Some(linear.max(
            super::super::state::nonlinear_violation(&engine.point, evaluator.cooperation).await?,
        )),
        None => None,
    };
    Ok((
        Outcome {
            engine: Some(engine),
            initial: None,
            reason,
            objective_calls: evaluator.state.objective_calls,
            constraint_calls: evaluator.state.constraint_calls,
            violation,
        },
        rng,
        options,
    ))
}
