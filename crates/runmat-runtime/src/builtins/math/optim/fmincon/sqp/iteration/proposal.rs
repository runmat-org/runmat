//! Retire quadratic proposal work before merit-search callbacks are polled.
use super::super::super::{active_set, model::Linearization, quadratic};
use super::super::Strategy;
use super::Evaluator;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;

pub(super) async fn solve(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    constraints: &Linearization,
    tolerance: f64,
    evaluator: &mut Evaluator<'_>,
    strategy: Strategy,
) -> BuiltinResult<quadratic::Result> {
    let owner = evaluator.policy.owner;
    match strategy {
        Strategy::Conic => {
            quadratic::regular(
                hessian,
                gradient,
                constraints,
                tolerance,
                owner,
                evaluator.cooperation,
            )
            .await
        }
        Strategy::WorkingSet => {
            if let Some(domain) = &evaluator.semi_infinite {
                let limits = domain
                    .controls
                    .pivots(gradient.len(), constraints, evaluator.cooperation)
                    .await?;
                active_set::solve_with_limits(
                    hessian,
                    gradient,
                    constraints,
                    limits,
                    owner,
                    evaluator.cooperation,
                )
                .await
            } else {
                active_set::solve(
                    hessian,
                    gradient,
                    constraints,
                    tolerance,
                    owner,
                    evaluator.cooperation,
                )
                .await
            }
        }
    }
}
