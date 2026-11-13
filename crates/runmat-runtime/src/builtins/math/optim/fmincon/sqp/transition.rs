//! Commit one constrained generation, retaining an existing original-x point
//! only under typed attainment reuse. Build a successor Hessian only if needed.
use super::super::{evaluation::Evaluator, model::GeometryView, observation, validation};
use super::{bfgs, merit, Outcome, Parameters};
use crate::BuiltinResult;
pub(super) struct Update {
    pub secant: Option<bfgs::Secant>,
    pub alpha: f64,
    pub optimal: bool,
    pub function_small: bool,
    pub interruption: Option<super::Termination>,
}
pub(super) async fn accept(
    out: &mut Outcome,
    accepted: merit::Accepted,
    multipliers: Vec<f64>,
    elastic: bool,
    geometry: &GeometryView<'_>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Update> {
    let owner = evaluator.policy.owner;
    if accepted.point.is_none() && !evaluator.reuses_generation(&out.x, &accepted.x).await? {
        return Err(super::super::invalid(
            "retained point has a different original-x generation",
        ));
    }
    let next_point = accepted
        .point
        .as_ref()
        .unwrap_or_else(|| out.point.as_ref().expect("accepted point"));
    let function_small = if evaluator.semi_infinite.is_some() {
        false
    } else if let Some(tolerance) = parameters.function {
        let previous = evaluator
            .reported_objective(out.point.as_ref().unwrap())
            .await?;
        let next = if evaluator.attainment.is_some() && !evaluator.uses_multiobjective_merit() {
            accepted.x.last().copied()
        } else {
            evaluator.reported_objective(next_point).await?
        };
        match (previous, next) {
            (Some(previous), Some(next)) => {
                (next - previous).abs() <= tolerance * previous.abs().max(next.abs()).max(1.0)
            }
            _ => false,
        }
    } else {
        false
    };
    let next_constraints = geometry
        .linearize_with_values(
            &accepted.x,
            next_point,
            accepted.values,
            owner,
            evaluator.cooperation,
        )
        .await?;
    // Elastic duals describe another objective, never this objective's lambda.
    let multipliers = if elastic {
        observation::copy(&out.multipliers, owner, evaluator.cooperation).await?
    } else {
        multipliers
    };
    let secant = if evaluator.semi_infinite.is_some() {
        None
    } else {
        bfgs::between(
            bfgs::LagrangianPoint {
                x: &out.x,
                gradient: out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
                constraints: out.constraints.as_ref().unwrap(),
            },
            bfgs::LagrangianPoint {
                x: &accepted.x,
                gradient: next_point.gradient.as_ref().unwrap(),
                constraints: &next_constraints,
            },
            &multipliers,
            owner,
            evaluator.cooperation,
        )
        .await?
    };
    if let Some(point) = accepted.point {
        out.point = Some(point);
    } else {
        evaluator
            .accept_retained(
                &out.x,
                &accepted.x,
                out.point.as_mut().unwrap(),
                &next_constraints,
            )
            .await?;
    }
    out.x = accepted.x;
    out.constraints = Some(next_constraints);
    out.multipliers = multipliers;
    out.multipliers_available = out
        .point
        .as_ref()
        .is_some_and(|point| point.full_derivatives())
        && (out.multipliers_available || !elastic);
    out.iterations += 1;
    out.step = accepted.step_norm;
    out.alpha = Some(accepted.alpha);
    if evaluator.semi_infinite.is_some() {
        if let super::super::semi_infinite::Refresh::Stopped(stop) =
            super::super::semi_infinite::refresh(
                out,
                geometry.require_owned(owner)?,
                parameters,
                evaluator,
            )
            .await?
        {
            return Ok(Update {
                secant: None,
                alpha: accepted.alpha,
                optimal: false,
                function_small: false,
                interruption: Some(stop),
            });
        }
    }
    super::super::semi_infinite::require_current(
        evaluator,
        out.point.as_ref().unwrap(),
        out.constraints.as_ref().unwrap(),
    )?;
    let checked = validation::check(
        out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
        out.constraints.as_ref().unwrap(),
        &out.multipliers,
        parameters.feasibility,
        parameters.optimality,
        owner,
        evaluator.cooperation,
    )
    .await?;
    out.violation = checked.as_ref().map(|value| value.violation);
    out.optimality = checked
        .as_ref()
        .filter(|_| out.point.as_ref().unwrap().full_derivatives())
        .map(|value| value.optimality);
    let optimal = out.point.as_ref().unwrap().full_derivatives()
        && checked.as_ref().is_some_and(|value| value.valid);
    if evaluator.semi_infinite.is_some() {
        out.multipliers_available = optimal;
    }
    Ok(Update {
        interruption: None,
        secant,
        alpha: accepted.alpha,
        optimal,
        function_small,
    })
}
