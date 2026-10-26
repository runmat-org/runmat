//! Nonlinear residual products borrow the original model and one controller.
use super::*;
use crate::builtins::math::optim::{
    least_squares::{LeastSquaresBounds, LeastSquaresResult},
    lsqnonlin::compiled as residual_solver,
};
mod publication;
pub(super) use publication::publish;
pub(super) struct Outcome {
    result: LeastSquaresResult,
    options: residual_solver::Options,
    bounds: LeastSquaresBounds,
    point: super::super::residual::Candidate,
}
pub(super) async fn solve(
    model: &Model,
    formulation: &formulation::Formulation,
    starting: Option<&Value>,
    options: Option<&Value>,
    _products: Products,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    c.completed(0).await?;
    if model.inequalities != 0 || model.equalities != 0 {
        return Err(policy.invalid("this nonlinear least-squares model consumer supports bounds only; comparison constraints require the constrained residual solver"));
    }
    if formulation.objective.as_ref().is_none_or(|value| {
        value.sense != runmat_value::optimization_model::OptimizationObjectiveSense::Minimize
    }) {
        return Err(policy.invalid("nonlinear least squares requires a minimization objective"));
    }
    let squares = model.squares.as_ref().ok_or_else(|| {
        policy.invalid("nonlinear residual solver requires original squared-residual evidence")
    })?;
    let mut rows = 0usize;
    let mut roots = runmat_value::exact_capacity_vec(
        formulation
            .roots
            .len()
            .checked_add(squares.terms.len())
            .ok_or_else(|| policy.invalid("residual root count overflow"))?,
    )
    .map_err(|e| policy.storage(e))?;
    roots.extend_from_slice(&formulation.roots);
    for term in &squares.terms {
        if !term.weight.exact.is_positive()
            || !term.weight.proposal.is_finite()
            || term.weight.proposal <= 0.
        {
            return Err(policy.invalid("nonlinear residual weights must be positive and representable in working precision"));
        }
        let node = model
            .source
            .nodes
            .get(&term.residual)
            .ok_or_else(|| policy.invalid("residual lost original source"))?
            .node();
        rows = rows
            .checked_add(evaluation::element_count(&node.shape, policy)?)
            .ok_or_else(|| policy.invalid("nonlinear residual count overflow"))?;
        roots.push(term.residual);
        c.completed(1).await?;
    }
    if rows == 0 {
        return Err(policy.invalid("nonlinear residual vector must not be empty"));
    }
    let options = residual_solver::model_options(
        residual_solver::prepare(options, model.layout.columns, c).await?,
    );
    let problem = inputs::prepare(model, starting, policy, c).await?;
    let bounds = LeastSquaresBounds {
        lower: problem.lower,
        upper: problem.upper,
    };
    let mut oracle = super::super::residual::Original {
        source: &model.source,
        layout: &model.layout,
        projection: super::super::residual::Projection::Squares {
            squares,
            objective: formulation
                .objective
                .as_ref()
                .expect("validated residual objective")
                .root,
        },
        roots: &roots,
        rows,
        pending: None,
        accepted: None,
        policy,
        cooperation: c,
    };
    let result = residual_solver::solve(&options, problem.x, &bounds, &mut oracle)?.await?;
    let accepted = oracle.accepted.take();
    drop(oracle);
    let point =
        super::super::residual::Candidate::select(&model.layout, &result, accepted, policy, c)
            .await?;
    Ok(Outcome {
        result,
        options,
        bounds,
        point,
    })
}
