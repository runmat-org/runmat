//! Rejected radii reuse one accepted model/preconditioner. Only actual numeric
//! undefinedness shrinks a trial; callback/service failures leave unchanged.
use super::super::evaluation::EvaluationStop;
use super::*;
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::execution::RootedValue;
pub(super) struct Accepted {
    pub raw: Option<RootedValue>,
    pub function_small: bool,
    pub radius_exhausted: bool,
    pub direction: Vec<f64>,
}
fn shrink(info: &mut publication::Info) -> bool {
    let next = info.radius * 0.25;
    if next > 0.0 && next < info.radius {
        info.radius = next;
        true
    } else {
        false
    }
}
fn relative_step(p: &[f64], x: &[f64]) -> f64 {
    let scale = x
        .iter()
        .chain(p)
        .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
    let numerator = p
        .iter()
        .fold(0.0_f64, |length, &value| length.hypot(value / scale));
    let denominator = x
        .iter()
        .fold(0.0_f64, |length, &value| length.hypot(value / scale))
        + 1.0 / scale;
    numerator / denominator
}
pub(super) async fn search(
    out: &mut Outcome,
    space: &mut subproblem::Space<'_, '_, '_>,
    gradient: &[f64],
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    controls: &Controls,
    info: &mut publication::Info,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<std::result::Result<Accepted, Termination>> {
    let Some(prepared) =
        subproblem::Prepared::new(space, out.x.len(), controls, evaluator.cooperation).await?
    else {
        return Ok(Err(Termination::Numerical));
    };
    loop {
        if out.iterations >= parameters.max_iterations || evaluator.budget.remaining() == 0 {
            return Ok(Err(Termination::Limit));
        }
        let Some(step) = prepared
            .step(
                space,
                gradient,
                info.radius,
                controls,
                evaluator.cooperation,
            )
            .await?
        else {
            return Ok(Err(Termination::Numerical));
        };
        info.cg_iterations = info
            .cg_iterations
            .checked_add(step.iterations)
            .ok_or_else(|| super::super::invalid("PCG iteration count overflow"))?;
        let Some(proposal) = space
            .proposal(
                step,
                &out.x,
                out.point
                    .as_ref()
                    .expect("accepted point")
                    .gradient
                    .as_ref()
                    .expect("analytic gradient"),
                geometry,
                parameters.feasibility,
                info.radius,
                evaluator.cooperation,
            )
            .await?
        else {
            if !shrink(info) {
                return Ok(Err(Termination::Numerical));
            }
            continue;
        };
        if relative_step(&proposal.displacement, &out.x) <= parameters.step
            && out
                .violation
                .is_some_and(|value| value <= parameters.feasibility)
        {
            out.step = norm(&proposal.displacement);
            return Ok(Err(Termination::Step));
        }
        let previous = out.point.as_ref().expect("accepted point").f;
        let third = !matches!(
            controls.source,
            controls::Source::GradientDifferences { .. }
        );
        let result = evaluator.evaluate_reflective(&proposal.x, third).await?;
        if matches!(result, Err(EvaluationStop::Time)) {
            return Ok(Err(Termination::Time));
        }
        if matches!(result, Err(EvaluationStop::BudgetExhausted)) {
            return Ok(Err(Termination::Limit));
        }
        out.iterations = out
            .iterations
            .checked_add(1)
            .ok_or_else(|| super::super::invalid("reflective iteration count overflow"))?;
        out.step = norm(&proposal.displacement);
        let candidate = match result {
            Ok((point, raw)) => Some((point, raw)),
            Err(EvaluationStop::Undefined) => None,
            Err(EvaluationStop::Time) => unreachable!("checked selected deadline"),
            Err(EvaluationStop::BudgetExhausted) => unreachable!("checked point budget"),
        };
        let mut radius_exhausted = false;
        if let Some((point, raw)) = candidate {
            let reduction = previous - point.f;
            let ratio = if reduction.is_finite() {
                reduction / proposal.predicted
            } else {
                previous / proposal.predicted - point.f / proposal.predicted
            };
            if !ratio.is_finite() {
                return Ok(Err(Termination::Numerical));
            }
            if ratio < 0.25 {
                radius_exhausted = !shrink(info);
            } else if ratio > 0.75 && proposal.boundary {
                let next = info.radius * 2.0;
                if next.is_finite() {
                    info.radius = next;
                }
            }
            if ratio > 0.1 && point.f < previous {
                let scale = previous.abs().max(point.f.abs()).max(1.0);
                let function_small =
                    (previous / scale - point.f / scale).abs() <= controls.function_tolerance;
                out.x = proposal.x;
                out.point = Some(point);
                out.multipliers_available = false;
                out.violation = None;
                out.optimality = None;
                return Ok(Ok(Accepted {
                    raw,
                    function_small,
                    radius_exhausted,
                    direction: proposal.displacement,
                }));
            }
            // Rejected H/Hinfo retires with this exact trial generation.
        } else {
            radius_exhausted = !shrink(info);
        }
        if publication::observe(
            out,
            parameters,
            *info,
            OutputState::Iter,
            &proposal.displacement,
            evaluator,
        )
        .await?
        {
            return Ok(Err(Termination::Observer));
        }
        if radius_exhausted {
            return Ok(Err(Termination::Numerical));
        }
    }
}
