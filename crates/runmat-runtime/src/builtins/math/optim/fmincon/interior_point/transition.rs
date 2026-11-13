//! Accepted primal-dual generation transfer. Same-point centering retains the
//! existing evaluated point; a changed point moves its original value stage.
use super::super::{evaluation::Evaluator, hessian, model::Geometry, outcome::Outcome, validation};
use super::{globalization::Accepted, initialization::Barrier};
use crate::BuiltinResult;
pub(super) async fn accept(
    out: &mut Outcome,
    accepted: Accepted,
    barrier: &mut Barrier,
    geometry: &Geometry,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<()> {
    let owner = evaluator.policy.owner;
    match accepted.generation {
        None => {
            let Some(multipliers) = barrier
                .layout
                .original_multipliers(
                    out.constraints.as_ref().unwrap(),
                    out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
                    &accepted.state,
                    owner,
                    evaluator.cooperation,
                )
                .await?
            else {
                return Err(super::super::invalid(
                    "accepted barrier dual generation is not finite",
                ));
            };
            out.multipliers = multipliers;
        }
        Some(trial) => {
            let original = geometry
                .linearize_with_values(
                    &trial.x,
                    &trial.point,
                    trial.constraints,
                    owner,
                    evaluator.cooperation,
                )
                .await?;
            let Some(multipliers) = barrier
                .layout
                .original_multipliers(
                    &original,
                    trial.point.gradient.as_ref().unwrap(),
                    &accepted.state,
                    owner,
                    evaluator.cooperation,
                )
                .await?
            else {
                return Err(super::super::invalid(
                    "accepted barrier dual generation is not finite",
                ));
            };
            let secant = hessian::between(
                hessian::LagrangianPoint {
                    x: &out.x,
                    gradient: out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
                    constraints: out.constraints.as_ref().unwrap(),
                },
                hessian::LagrangianPoint {
                    x: &trial.x,
                    gradient: trial.point.gradient.as_ref().unwrap(),
                    constraints: &original,
                },
                &multipliers,
                owner,
                evaluator.cooperation,
            )
            .await?;
            out.x = trial.x;
            out.point = Some(trial.point);
            out.constraints = Some(original);
            out.multipliers = multipliers;
            if let Some(value) = secant {
                if let Some(next) = hessian::update(
                    out.hessian.as_ref().expect("accepted IP model"),
                    &value.displacement,
                    &value.change,
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
    barrier.state = accepted.state;
    out.multipliers_available = out.point.as_ref().unwrap().full_derivatives();
    out.iterations += 1;
    out.step = accepted.step;
    out.alpha = Some(accepted.alpha);
    // Actual point/callback counts are sampled only from the one original budget.
    out.counts = evaluator.budget.counts();
    Ok(())
}

pub(super) async fn assess(
    out: &mut Outcome,
    parameters: &super::super::outcome::Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<bool> {
    let checked = validation::check(
        out.point.as_ref().unwrap().gradient.as_ref().unwrap(),
        out.constraints.as_ref().unwrap(),
        &out.multipliers,
        parameters.feasibility,
        parameters.optimality,
        evaluator.policy.owner,
        evaluator.cooperation,
    )
    .await?;
    out.violation = checked.as_ref().map(|value| value.violation);
    out.optimality = checked
        .as_ref()
        .filter(|_| out.point.as_ref().unwrap().full_derivatives())
        .map(|value| value.optimality);
    Ok(out.point.as_ref().unwrap().full_derivatives()
        && checked.as_ref().is_some_and(|value| value.valid))
}
