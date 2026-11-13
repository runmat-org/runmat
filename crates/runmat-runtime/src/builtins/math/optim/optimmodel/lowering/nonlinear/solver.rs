//! Canonical solver identities select only compatible original model geometry.
use super::*;
use runmat_builtins::OptimSolver;
use runmat_value::optimization_model::OptimizationObjectiveSense;
use runmat_value::optimization_variable::OptimizationVariableDomain;

use runmat_value::optimization_variable::{OptimizationBoundRole as Role, OptimizationBoundView};
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn select(
    model: &Model,
    formulation: &formulation::Formulation,
    requested: Option<OptimSolver>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<OptimSolver> {
    let rows = model.inequalities != 0 || model.equalities != 0;
    let mut bounded = false;
    for variable in &model.layout.variables {
        if variable.domain != OptimizationVariableDomain::Continuous {
            return Err(policy
                .invalid("this nonlinear model family cannot discard an integer variable domain"));
        }
        if variable.count == 0 {
            c.completed(1).await?;
            continue;
        }
        for role in [Role::Lower, Role::Upper] {
            let free = match role {
                Role::Lower => f64::NEG_INFINITY,
                Role::Upper => f64::INFINITY,
            };
            match variable.view.definition().bound(role) {
                OptimizationBoundView::Uniform(value) => {
                    bounded |= value != free;
                }
                OptimizationBoundView::PerElement(values) => {
                    for chunk in values.chunks(1024) {
                        bounded |= chunk.iter().any(|&value| value != free);
                        c.completed(chunk.len() as u64).await?;
                    }
                }
            }
            c.completed(1).await?;
        }
    }
    if let Some(requested) = requested {
        return match requested {
            OptimSolver::Fminunc if rows || bounded => Err(policy.invalid(
                "fminunc requires an original model without bounds or comparison constraints",
            )),
            OptimSolver::Fminunc | OptimSolver::Fmincon | OptimSolver::Lsqnonlin => Ok(requested),
            _ => {
                Err(policy
                    .invalid("selected solver does not support this original nonlinear model"))
            }
        };
    }
    if !rows
        && formulation
            .objective
            .as_ref()
            .is_some_and(|objective| objective.sense == OptimizationObjectiveSense::Minimize)
    {
        if let Some(squares) = &model.squares {
            let mut positive = !squares.terms.is_empty();
            let mut nonempty = false;
            for term in &squares.terms {
                positive &= term.weight.exact.is_positive();
                let node = model
                    .source
                    .nodes
                    .get(&term.residual)
                    .ok_or_else(|| policy.invalid("residual lost its original source"))?
                    .node();
                nonempty |= evaluation::element_count(&node.shape, policy)? != 0;
                c.completed(1).await?;
            }
            if positive && nonempty {
                return Ok(OptimSolver::Lsqnonlin);
            }
        }
    }
    Ok(if rows || bounded {
        OptimSolver::Fmincon
    } else {
        OptimSolver::Fminunc
    })
}
