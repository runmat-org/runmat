//! Actual residual controllers consume the same original equation oracle.
use super::*;
use crate::builtins::math::optim::{
    fsolve::compiled as system, least_squares::LeastSquaresResult, lsqnonlin::compiled as least,
};
use runmat_builtins::OptimSolver;
pub(super) enum Options {
    System(system::Options),
    LeastSquares(least::Options),
}
pub(in crate::builtins::math::optim::optimmodel) struct Outcome {
    pub(super) result: LeastSquaresResult,
    pub(super) options: Options,
    pub(super) point: super::super::residual::Candidate,
}
pub(super) async fn solve(
    model: &Model,
    solver: OptimSolver,
    starting: Option<&Value>,
    options: Option<&Value>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    c.completed(0).await?;
    let options = match solver {
        OptimSolver::Fsolve => Options::System(system::prepare(options, c).await?),
        OptimSolver::Lsqnonlin => Options::LeastSquares(least::equation_options(
            least::prepare(options, model.layout.columns, c).await?,
        )),
        _ => {
            return Err(
                policy.invalid("selected solver is not a nonlinear equation residual controller")
            )
        }
    };
    let bounds = layout::continuous_bounds(&model.layout, policy, c).await?;
    let starting = starting.ok_or_else(|| {
        policy.invalid(
            "nonlinear equations require an initial-point structure of original variable values",
        )
    })?;
    let x = layout::initial_point(&model.source, &model.layout, starting, policy, c).await?;
    let mut oracle = super::super::residual::Original {
        source: &model.source,
        layout: &model.layout,
        projection: super::super::residual::Projection::Equations(&model.evidence),
        roots: &model.evidence.formulation.roots,
        rows: model.evidence.rows,
        pending: None,
        accepted: None,
        policy,
        cooperation: c,
    };
    let result = match &options {
        Options::System(options) => system::solve(options, x, &bounds, &mut oracle)?.await?,
        Options::LeastSquares(options) => {
            let mut result = least::solve(options, x, &bounds, &mut oracle)?.await?;
            crate::builtins::math::optim::least_squares::equation::require_root(
                &mut result,
                least::equation_tolerance(options),
                Some(&mut *oracle.cooperation),
            )
            .await?;
            result
        }
    };
    let accepted = oracle.accepted.take();
    drop(oracle);
    let point =
        super::super::residual::Candidate::select(&model.layout, &result, accepted, policy, c)
            .await?;
    Ok(Outcome {
        result,
        options,
        point,
    })
}
