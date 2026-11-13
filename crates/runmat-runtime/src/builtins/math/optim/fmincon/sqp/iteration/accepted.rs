//! The single accepted-state SQP loop, with its original measurements,
//! transitions, nonlinear checks and observer ordering conserved.
use super::super::super::validation;
use super::super::{bfgs, merit, restoration, subproblem, Termination};
use super::{Evaluator, Geometry, GeometryView, Outcome, Parameters};
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::BuiltinResult;

pub(super) async fn run(
    out: Outcome,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::super::Strategy,
) -> BuiltinResult<Outcome> {
    run_view(out, &geometry.view(), parameters, evaluator, strategy).await
}

pub(super) async fn run_view(
    mut out: Outcome,
    geometry: &GeometryView<'_>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::super::Strategy,
) -> BuiltinResult<Outcome> {
    let owner = evaluator.policy.owner;
    let mut penalty = 1.0;
    if out
        .observe(evaluator, parameters, OutputState::Init, &[], None)
        .await?
    {
        out.termination = Termination::Observer;
    } else if out.constraints.is_some() {
        loop {
            if evaluator.time_limit_reached() {
                out.termination = Termination::Time;
                break;
            }
            let constraints = out
                .constraints
                .as_ref()
                .expect("complete accepted constraints");
            let point = out.point.as_ref().expect("complete accepted point");
            let gradient = point.gradient.as_ref().expect("complete accepted gradient");
            super::super::super::semi_infinite::require_current(evaluator, point, constraints)?;
            let checked = validation::check(
                gradient,
                constraints,
                &out.multipliers,
                parameters.feasibility,
                parameters.optimality,
                owner,
                evaluator.cooperation,
            )
            .await?;
            out.optimality = checked
                .as_ref()
                .filter(|_| point.full_derivatives())
                .map(|value| value.optimality);
            if point.full_derivatives() && checked.as_ref().is_some_and(|value| value.valid) {
                out.multipliers_available = true;
                out.termination = Termination::Optimality;
                break;
            }
            if out.iterations >= parameters.max_iterations || evaluator.budget.remaining() == 0 {
                out.termination = Termination::Limit;
                break;
            }
            if out.hessian.is_none() {
                out.hessian = Some(evaluator.initial_model(out.x.len()).await?);
            }
            let hessian = out.hessian.as_ref().expect("certified accepted Hessian");
            let subproblem_tolerance = evaluator.semi_infinite.as_ref().map_or(
                parameters.optimality.min(parameters.feasibility),
                |domain| domain.controls.inner_tolerance,
            );
            let proposed = super::proposal::solve(
                hessian,
                gradient,
                constraints,
                subproblem_tolerance,
                evaluator,
                strategy,
            )
            .await?;
            let mut elastic = false;
            let step = match proposed {
                subproblem::Result::Step(step) => step,
                _ => {
                    if out
                        .violation
                        .is_some_and(|value| value <= parameters.feasibility)
                    {
                        out.termination = Termination::Numerical;
                        break;
                    }
                    elastic = true;
                    let Some(step) = restoration::solve(
                        hessian,
                        gradient,
                        constraints,
                        penalty,
                        subproblem_tolerance,
                        owner,
                        evaluator.cooperation,
                    )
                    .await?
                    else {
                        out.termination = Termination::NoFeasiblePoint;
                        break;
                    };
                    step
                }
            };
            // At an initially constrained optimum a zero QP step can supply
            // the missing real multipliers without inventing an accepted step.
            if !elastic {
                if let Some(checked) = validation::check(
                    gradient,
                    constraints,
                    &step.multipliers,
                    parameters.feasibility,
                    parameters.optimality,
                    owner,
                    evaluator.cooperation,
                )
                .await?
                {
                    if checked.valid && point.full_derivatives() {
                        out.multipliers = step.multipliers;
                        out.multipliers_available = true;
                        out.optimality = Some(checked.optimality);
                        out.termination = Termination::Optimality;
                        break;
                    }
                }
            }
            if evaluator.semi_infinite.is_some()
                && !elastic
                && out
                    .violation
                    .is_some_and(|value| value <= parameters.feasibility)
            {
                let mut norm = 0.0f64;
                let mut directional = 0.0;
                for (&gradient, &direction) in gradient.iter().zip(&step.direction) {
                    norm = norm.max(direction.abs());
                    directional += gradient * direction;
                    evaluator.cooperation.completed(1).await?;
                }
                if norm <= parameters.step {
                    out.termination = Termination::Step;
                    out.step = 0.0;
                    out.alpha = Some(0.0);
                    break;
                }
                if directional.is_finite()
                    && parameters
                        .function
                        .is_some_and(|tolerance| directional.abs() <= tolerance)
                {
                    out.termination = Termination::Function;
                    out.step = 0.0;
                    out.alpha = Some(0.0);
                    break;
                }
            }
            let result = {
                let mut search = merit::Search {
                    evaluator: &mut *evaluator,
                    geometry,
                    differences: parameters.differences,
                    penalty,
                    iteration: out.iterations,
                };
                let result = search
                    .search(
                        &out.x,
                        point,
                        constraints,
                        &step.direction,
                        &step.multipliers,
                    )
                    .await?;
                penalty = search.penalty;
                result
            };
            let accepted = match result {
                Ok(value) => value,
                Err(merit::SearchStop::Time) => {
                    out.termination = Termination::Time;
                    break;
                }
                Err(merit::SearchStop::Limit) => {
                    out.termination = Termination::Limit;
                    break;
                }
                Err(merit::SearchStop::StepBound) => {
                    out.termination = if out.violation.is_some_and(|v| v <= parameters.feasibility)
                    {
                        Termination::Step
                    } else {
                        Termination::NoFeasiblePoint
                    };
                    out.step = 0.0;
                    out.alpha = Some(0.0);
                    break;
                }
                Err(merit::SearchStop::NoDescent) => {
                    let mut norm = 0.0f64;
                    for &value in &step.direction {
                        norm = norm.max(value.abs());
                        evaluator.cooperation.completed(1).await?;
                    }
                    out.termination = if out
                        .violation
                        .is_some_and(|value| value <= parameters.feasibility)
                    {
                        if norm <= parameters.step {
                            Termination::Step
                        } else {
                            Termination::Numerical
                        }
                    } else {
                        Termination::NoFeasiblePoint
                    };
                    break;
                }
            };
            let update = super::super::transition::accept(
                &mut out,
                accepted,
                step.multipliers,
                elastic,
                geometry,
                parameters,
                evaluator,
            )
            .await?;
            if let Some(stop) = update.interruption {
                out.termination = stop;
                break;
            }
            if out
                .observe(
                    evaluator,
                    parameters,
                    OutputState::Iter,
                    &step.direction,
                    Some(update.alpha),
                )
                .await?
            {
                out.termination = Termination::Observer;
                break;
            }
            if update.optimal {
                out.termination = Termination::Optimality;
                break;
            }
            if update.function_small
                && out
                    .violation
                    .is_some_and(|value| value <= parameters.feasibility)
            {
                out.termination = Termination::Function;
                break;
            }
            if out.step <= parameters.step
                && out
                    .violation
                    .is_some_and(|value| value <= parameters.feasibility)
            {
                out.termination = Termination::Step;
                break;
            }
            if out.iterations >= parameters.max_iterations || evaluator.budget.remaining() == 0 {
                out.termination = Termination::Limit;
                break;
            }
            if let Some(secant) = update.secant {
                if let Some(next) = bfgs::update(
                    out.hessian.as_ref().expect("last-used accepted model"),
                    &secant.displacement,
                    &secant.change,
                    owner,
                    evaluator.cooperation,
                )
                .await?
                {
                    out.hessian = Some(next);
                }
            }
        }
    }
    out.counts = evaluator.budget.counts();
    let _ = out
        .observe(evaluator, parameters, OutputState::Done, &[], None)
        .await?;
    Ok(out)
}
