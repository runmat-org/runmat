//! A callback generation and positive barrier generation are prepared before
//! the controller enters Newton iterations. No synthetic infeasibility proof.
use super::super::{
    derivatives::{self, Completion},
    evaluation::{EvaluationStop, Evaluator},
    hessian,
    model::Geometry,
    outcome::{Outcome, Parameters, Termination},
};
use super::state::{self, Layout, State};
use crate::BuiltinResult;
pub(super) struct Barrier {
    pub layout: Layout,
    pub state: State,
}
pub(super) struct Initialized {
    pub outcome: Outcome,
    pub barrier: Option<Barrier>,
}
pub(super) async fn prepare(
    mut x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Initialized> {
    let owner = evaluator.policy.owner;
    let interior = state::interior_start(&mut x, geometry, evaluator.cooperation).await?;
    let mut out = Outcome {
        algorithm: runmat_builtins::OptimAlgorithm::InteriorPoint,
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
    if !interior {
        out.termination = Termination::Numerical;
        return Ok(Initialized {
            outcome: out,
            barrier: None,
        });
    }
    match evaluator.evaluate(&out.x).await? {
        Err(EvaluationStop::Time) => {
            out.termination = Termination::Time;
            return Ok(Initialized {
                outcome: out,
                barrier: None,
            });
        }
        Err(EvaluationStop::BudgetExhausted) => {
            return Ok(Initialized {
                outcome: out,
                barrier: None,
            })
        }
        Err(EvaluationStop::Undefined) => {
            return Err(super::super::invalid(
                "objective or nonlinear constraint is undefined at the initial interior point",
            ))
        }
        Ok(point) => out.point = Some(point),
    }
    let point = out.point.as_mut().expect("evaluated initial point");
    out.violation = geometry
        .violation(&out.x, point, owner, evaluator.cooperation)
        .await?
        .map(|value| value.0);
    match derivatives::complete(
        evaluator,
        &out.x,
        &geometry.lower,
        &geometry.upper,
        point,
        parameters.differences,
    )
    .await?
    {
        Completion::Time => {
            out.termination = Termination::Time;
            return Ok(Initialized {
                outcome: out,
                barrier: None,
            });
        }
        Completion::BudgetExhausted => {
            return Ok(Initialized {
                outcome: out,
                barrier: None,
            })
        }
        Completion::Undefined => {
            out.termination = Termination::Numerical;
            return Ok(Initialized {
                outcome: out,
                barrier: None,
            });
        }
        Completion::Complete => {}
    }
    let original = geometry
        .linearize(
            &out.x,
            out.point.as_ref().unwrap(),
            owner,
            evaluator.cooperation,
        )
        .await?;
    let layout = Layout::new(&original, geometry, owner, evaluator.cooperation).await?;
    let Some(state) = State::new(&layout, &original, owner, evaluator.cooperation).await? else {
        out.termination = Termination::Numerical;
        return Ok(Initialized {
            outcome: out,
            barrier: None,
        });
    };
    let gradient = out.point.as_ref().unwrap().gradient.as_ref().unwrap();
    let Some(multipliers) = layout
        .original_multipliers(&original, gradient, &state, owner, evaluator.cooperation)
        .await?
    else {
        out.termination = Termination::Numerical;
        return Ok(Initialized {
            outcome: out,
            barrier: None,
        });
    };
    out.multipliers = multipliers;
    out.constraints = Some(original);
    // IP's BFGS output is the chosen model at the final point, unlike SQP's
    // last-used step model. No additional objective probe is introduced.
    out.hessian = Some(hessian::initial(out.x.len(), owner, evaluator.cooperation).await?);
    Ok(Initialized {
        outcome: out,
        barrier: Some(Barrier { layout, state }),
    })
}
