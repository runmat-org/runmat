//! Primal working-set quadratic subproblems. Feasibility is established before
//! objective steps; blocking inequalities enter and negative-dual rows leave.
//! Equality-constrained solves share the cooperative numerical substrate, not
//! the full-inequality objective solve used by the conic SQP strategy.
use super::{hessian, model::Linearization, quadratic, workspace};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;
mod rows;
#[cfg(test)]
mod tests;
mod working_set;

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub tolerance: f64,
    pub iterations: Option<usize>,
}
pub(super) async fn solve(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    model: &Linearization,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<quadratic::Result> {
    solve_with_limits(
        hessian,
        gradient,
        model,
        Limits {
            tolerance,
            iterations: None,
        },
        owner,
        cooperation,
    )
    .await
}
pub(super) async fn solve_with_limits(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    model: &Linearization,
    limits: Limits,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<quadratic::Result> {
    let tolerance = limits.tolerance;
    let mut seed = workspace::capacity(gradient.len(), owner)?;
    for _ in gradient {
        seed.push(0.0);
        cooperation.completed(1).await?;
    }
    if !rows::feasible(model, &seed, tolerance, owner, cooperation).await? {
        // Phase I minimizes distance from zero, not the original QP objective.
        // The original full rows are validated by the common QP checker.
        let metric = hessian::initial(gradient.len(), owner, cooperation).await?;
        match quadratic::regular(&metric, &seed, model, tolerance, owner, cooperation).await? {
            quadratic::Result::Step(step) => seed = step.direction,
            other => return Ok(other),
        }
    }
    working_set::minimize(hessian, gradient, model, seed, limits, owner, cooperation).await
}
