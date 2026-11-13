//! The first callback sees a genuine admitted bound/equality-feasible point.
use super::*;
use crate::execution::RootedValue;
pub(super) struct Started<'a> {
    pub outcome: Outcome,
    pub equality: Option<equality::Equality<'a>>,
    pub raw: Option<RootedValue>,
}
pub(super) async fn prepare<'a>(
    x: Vec<f64>,
    geometry: &'a Geometry,
    parameters: &Parameters<'_>,
    mode: Mode,
    third: bool,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Started<'a>> {
    let mut out = Outcome {
        algorithm: runmat_builtins::OptimAlgorithm::TrustRegionReflective,
        x,
        point: None,
        constraints: None,
        multipliers: Vec::new(),
        multipliers_available: false,
        hessian: None,
        iterations: 0,
        counts: evaluator.budget.counts(),
        termination: Termination::Limit,
        violation: None,
        optimality: None,
        step: 0.0,
        alpha: None,
    };
    let equality = match mode {
        Mode::Bounds => {
            if !super::super::bound_geometry::interior_start(
                &mut out.x,
                geometry,
                evaluator.cooperation,
            )
            .await?
            {
                out.termination = Termination::Numerical;
                return Ok(Started {
                    outcome: out,
                    equality: None,
                    raw: None,
                });
            }
            None
        }
        Mode::Equalities => {
            let Some(equality) =
                equality::Equality::new(&geometry.equality, evaluator.cooperation).await?
            else {
                out.termination = Termination::Numerical;
                return Ok(Started {
                    outcome: out,
                    equality: None,
                    raw: None,
                });
            };
            if !equality
                .correct(&mut out.x, parameters.feasibility, evaluator.cooperation)
                .await?
            {
                out.termination = Termination::Numerical;
                return Ok(Started {
                    outcome: out,
                    equality: Some(equality),
                    raw: None,
                });
            }
            Some(equality)
        }
    };
    let raw = match evaluator.evaluate_reflective(&out.x, third).await? {
        Ok((point, raw)) => {
            out.point = Some(point);
            raw
        }
        Err(super::super::evaluation::EvaluationStop::Time) => {
            out.termination = Termination::Time;
            None
        }
        Err(super::super::evaluation::EvaluationStop::BudgetExhausted) => None,
        Err(super::super::evaluation::EvaluationStop::Undefined) => {
            return Err(super::super::invalid(
                "objective or gradient is undefined at the initial reflective point",
            ))
        }
    };
    Ok(Started {
        outcome: out,
        equality,
        raw,
    })
}
