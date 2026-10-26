//! SQP bound-feasible initial point and derivative/model ownership.
use super::super::{
    derivatives::{self, Completion},
    evaluation::{EvaluationStop, Evaluator},
    model::GeometryView,
    workspace,
};
use super::{Outcome, Parameters, Termination};
use crate::{build_runtime_error, BuiltinResult};
pub(super) async fn initialize(
    mut x: Vec<f64>,
    geometry: &GeometryView<'_>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    let owner = evaluator.policy.owner;
    for variable in 0..x.len() {
        x[variable] = x[variable]
            .max(geometry.lower[variable])
            .min(geometry.upper[variable]);
        evaluator.cooperation.completed(1).await?;
    }
    let mut out = Outcome {
        algorithm: runmat_builtins::OptimAlgorithm::Sqp,
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
    match evaluator.evaluate(&out.x).await? {
        Err(EvaluationStop::Time)=>out.termination=Termination::Time,
        Err(EvaluationStop::BudgetExhausted)=>{},
        Err(EvaluationStop::Undefined) if evaluator.internal.is_some()=>{
            out.termination=Termination::Numerical;
            return Ok(out);
        },
        Err(EvaluationStop::Undefined)=>return Err(build_runtime_error(format!("{}: objective or nonlinear constraint is undefined at the initial bound-feasible point",owner.name))
            .with_builtin(owner.name).with_gpu_gather_retry(crate::GpuGatherRetry::Never).build()),
        Ok(point)=>out.point=Some(point),
    }
    if evaluator.semi_infinite.is_some() {
        if let super::super::semi_infinite::Refresh::Stopped(stop) =
            super::super::semi_infinite::refresh(
                &mut out,
                geometry.require_owned(owner)?,
                parameters,
                evaluator,
            )
            .await?
        {
            out.termination = stop;
        }
        return Ok(out);
    }
    if let Some(point) = out.point.as_mut() {
        if !evaluator.seed_attainment(&mut out.x, point).await? {
            out.termination = Termination::Numerical;
            return Ok(out);
        }
        match derivatives::complete(
            evaluator,
            &out.x,
            geometry.lower,
            geometry.upper,
            point,
            parameters.differences,
        )
        .await?
        {
            Completion::Complete => {
                let constraints = geometry
                    .linearize(&out.x, point, owner, evaluator.cooperation)
                    .await?;
                out.multipliers = workspace::capacity(constraints.b.len(), owner)?;
                for _ in &constraints.b {
                    out.multipliers.push(0.0);
                    evaluator.cooperation.completed(1).await?;
                }
                out.constraints = Some(constraints);
            }
            Completion::Time => out.termination = Termination::Time,
            Completion::BudgetExhausted => {}
            Completion::Undefined => out.termination = Termination::Numerical,
        }
    }
    if let Some(point) = out.point.as_ref() {
        out.violation = geometry
            .violation(&out.x, point, owner, evaluator.cooperation)
            .await?
            .map(|value| value.0);
    }
    Ok(out)
}
