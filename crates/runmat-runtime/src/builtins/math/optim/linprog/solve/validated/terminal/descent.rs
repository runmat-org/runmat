//! Original selected status arithmetic and callback ordering.
use super::*;
use crate::builtins::math::optim::linprog::compiled;
pub(super) async fn run<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<LinprogOutcome> {
    let Prepared {
        model,
        point,
        cooperation,
        controls,
        validation,
        ..
    } = prepared;
    if !compiled::descent(&model, &point.x, cooperation, validation).await? {
        return Ok(LinprogOutcome::failure(
            -4,
            controls.iterations(),
            "Descent certificate could not be verified against original constraints.",
        ));
    }
    if compiled::feasible(&model, &model.seed, cooperation, validation).await? {
        return Ok(LinprogOutcome::unbounded(controls.iterations()));
    }
    // A dual-infeasibility ray does not establish primal feasibility:
    // both primal and dual can be infeasible. Retire the first kernel
    // before the genuine auxiliary zero-objective feasibility problem.
    drop(point);
    if controls.remaining() == 0 {
        return Ok(LinprogOutcome::failure(0,controls.iterations(),"Iteration limit reached before primal feasibility for the descent certificate was established."));
    }
    let auxiliary = super::super::run(&model, true, cooperation, controls).await?;
    if compiled::feasible(&model, &auxiliary.solution.x, cooperation, validation).await?
        && !matches!(
            auxiliary.solution.status,
            SolverStatus::PrimalInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::DualInfeasible
                | SolverStatus::AlmostDualInfeasible
        )
    {
        Ok(LinprogOutcome::unbounded(controls.iterations()))
    } else if matches!(
        auxiliary.solution.status,
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible
    ) && compiled::farkas(&model, &auxiliary.solution.z, cooperation, validation).await?
    {
        Ok(LinprogOutcome::failure(
            -2,
            controls.iterations(),
            "Problem is infeasible; auxiliary original constraint certificate verified.",
        ))
    } else {
        let stopped = matches!(
            auxiliary.solution.status,
            SolverStatus::MaxIterations | SolverStatus::MaxTime | SolverStatus::CallbackTerminated
        );
        Ok(LinprogOutcome::failure(if stopped {0}else{-4},controls.iterations(),
                    "Primal feasibility for the descent certificate was not established; no unbounded conclusion is published."))
    }
}
