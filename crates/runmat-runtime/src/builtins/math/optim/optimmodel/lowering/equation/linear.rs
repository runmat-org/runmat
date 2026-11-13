//! The existing linear residual controller, with original equation evidence.
use super::*;
use crate::builtins::math::optim::{
    common::solver_timing::SolverTiming, lsqlin::compiled as solver,
};
use runmat_builtins::{OptimAlgorithm, OptimizationExitflag};
pub(in crate::builtins::math::optim::optimmodel) struct Outcome {
    pub(super) solver: report::Outcome,
    pub(super) flag: OptimizationExitflag,
    pub(super) point: Option<numerical::Point>,
    pub(super) nonroot: bool,
}
pub(super) async fn solve(
    program: &Program,
    starting: Option<&Value>,
    options: Option<&Value>,
    information: bool,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    c.completed(0).await?;
    let timing = if information {
        Some(SolverTiming::start(c.runtime())?)
    } else {
        None
    };
    let mut validation = validation::Validator::new(program, policy, c).await?;
    let problem = assembly::assemble(program, policy, c).await?;
    let prepared = solver::prepare(problem, options, c).await?;
    let root_tolerance = prepared.equation_tolerance();
    let initial = if matches!(
        prepared.algorithm(),
        OptimAlgorithm::TrustRegionReflective | OptimAlgorithm::ActiveSet
    ) {
        match starting {
            Some(starting) => {
                layout::initial_point(&program.source, &program.layout, starting, policy, c).await?
            }
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let result = solver::solve(
        prepared,
        initial,
        solver::Products::Primal,
        solver::Units::OriginalSquared,
        c,
        &mut validation,
    )
    .await?;
    let mut solver = report::Outcome::LeastSquares {
        outcome: result,
        timing,
    };
    let candidate = solver.point();
    let mut flag = solver.flag(policy)?;
    let point = match candidate {
        Some(candidate) => {
            let point = validation.point.take().ok_or_else(|| {
                policy.invalid("linear equation result has no original evaluated candidate")
            })?;
            if !point.matches(program, &candidate, policy, c).await? {
                return Err(policy.invalid(
                    "linear equation result differs from the original validated candidate",
                ));
            }
            Some(point)
        }
        None => None,
    };
    let mut nonroot = false;
    if matches!(
        flag,
        OptimizationExitflag::OptimalSolution
            | OptimizationExitflag::FunctionChangeBelowTolerance
            | OptimizationExitflag::StepSizeBelowTolerance
    ) {
        let point = point
            .as_ref()
            .ok_or_else(|| policy.invalid("positive equation stop has no original candidate"))?;
        let objective::ModelObjective::Equations(evidence) = &program.objective else {
            unreachable!("original equation program")
        };
        for value in evidence.residual(point, policy, c).await? {
            nonroot |= value.abs() > root_tolerance;
            c.completed(1).await?;
        }
        if nonroot {
            flag = OptimizationExitflag::NoFeasiblePointFound;
        }
    }
    if flag == OptimizationExitflag::OptimalSolution {
        flag = OptimizationExitflag::EquationSolved;
    }
    Ok(Outcome {
        solver,
        flag,
        point,
        nonroot,
    })
}
