//! Barrier/Newton controller. Initialization, positivity/globalization and
//! accepted-state transfer retain their own owners rather than a nested loop.
use super::super::{
    evaluation::Evaluator,
    model::Geometry,
    outcome::{Outcome, Parameters, Termination},
};
use super::{globalization, initialization, newton, state::State, transition};
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::BuiltinResult;

async fn small_direction(
    direction: &newton::Direction,
    state: &State,
    x: &[f64],
    tolerance: f64,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<bool> {
    for (&delta, &value) in direction.x.iter().zip(x) {
        if delta.abs() > tolerance * value.abs().max(1.0) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    for (&delta, &value) in direction
        .s
        .iter()
        .zip(&state.s)
        .chain(direction.z.iter().zip(&state.z))
        .chain(direction.y.iter().zip(&state.y))
    {
        if delta.abs() > tolerance * value.abs().max(1.0) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}

pub(in crate::builtins::math::optim::fmincon) async fn minimize(
    x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let initialization::Initialized {
        outcome: mut out,
        barrier,
    } = initialization::prepare(x, geometry, parameters, evaluator).await?;
    if out
        .observe(evaluator, parameters, OutputState::Init, &[], None)
        .await?
    {
        out.termination = Termination::Observer;
    } else if let Some(mut barrier) = barrier {
        let mut penalty = 1.0;
        loop {
            if evaluator.time_limit_reached() {
                out.termination = Termination::Time;
                break;
            }
            if transition::assess(&mut out, parameters, evaluator).await? {
                out.multipliers_available = true;
                out.termination = Termination::Optimality;
                break;
            }
            if out.iterations >= parameters.max_iterations || evaluator.budget.remaining() == 0 {
                out.termination = Termination::Limit;
                break;
            }
            let Some(mean) = barrier
                .state
                .mean_complementarity(evaluator.cooperation)
                .await?
            else {
                out.termination = Termination::Numerical;
                break;
            };
            // The target is genuinely monotone and uses the actual positive
            // complementarity generation, never a requested-tolerance floor.
            barrier.state.mu = barrier.state.mu.min(0.1 * mean);
            let accuracy = parameters.optimality.min(parameters.feasibility).min(1e-8);
            let Some(direction) = newton::solve(
                out.hessian.as_ref().unwrap(),
                out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
                out.constraints.as_ref().unwrap(),
                &barrier.layout,
                &barrier.state,
                &out.multipliers,
                accuracy,
                evaluator.policy.owner,
                evaluator.cooperation,
            )
            .await?
            else {
                // A local lifted-system refusal is not original nonlinear
                // infeasibility or permission to dispatch another solver.
                out.termination = Termination::Numerical;
                break;
            };
            let accepted = match globalization::search(
                &out.x,
                out.point.as_ref().unwrap(),
                out.constraints.as_ref().unwrap(),
                geometry,
                &barrier.layout,
                &barrier.state,
                &direction,
                parameters,
                &mut penalty,
                evaluator,
            )
            .await?
            {
                Ok(value) => value,
                Err(globalization::SearchStop::Time) => {
                    out.termination = Termination::Time;
                    break;
                }
                Err(globalization::SearchStop::Limit) => {
                    out.termination = Termination::Limit;
                    break;
                }
                Err(globalization::SearchStop::NoDescent) => {
                    out.termination = if out
                        .violation
                        .is_some_and(|value| value <= parameters.feasibility)
                        && small_direction(
                            &direction,
                            &barrier.state,
                            &out.x,
                            parameters.step,
                            evaluator.cooperation,
                        )
                        .await?
                    {
                        Termination::Step
                    } else {
                        Termination::Numerical
                    };
                    break;
                }
            };
            let alpha = accepted.alpha;
            let state_step = accepted.state_step;
            transition::accept(&mut out, accepted, &mut barrier, geometry, evaluator).await?;
            let optimal = transition::assess(&mut out, parameters, evaluator).await?;
            if out
                .observe(
                    evaluator,
                    parameters,
                    OutputState::Iter,
                    &direction.x,
                    Some(alpha),
                )
                .await?
            {
                out.termination = Termination::Observer;
                break;
            }
            if optimal {
                out.termination = Termination::Optimality;
                break;
            }
            let scale = out.x.iter().map(|value| value.abs()).fold(1.0, f64::max);
            if out.step <= parameters.step * scale
                && state_step <= parameters.step
                && out
                    .violation
                    .is_some_and(|value| value <= parameters.feasibility)
            {
                out.termination = Termination::Step;
                break;
            }
        }
    }
    out.counts = evaluator.budget.counts();
    let _ = out
        .observe(evaluator, parameters, OutputState::Done, &[], None)
        .await?;
    Ok(out)
}
