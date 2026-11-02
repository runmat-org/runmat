//! Construct one selected solver future before polling numerical callbacks.
use super::*;
use futures::{future::Either, FutureExt};
use std::future::Future;

pub(super) struct Execution<'a> {
    pub model: &'a Model,
    pub formulation: &'a formulation::Formulation,
    pub starting: Option<&'a Value>,
    pub options: Option<&'a Value>,
    pub products: Products,
    pub policy: Policy,
    pub cooperation: &'a mut ScalarCooperation,
}

impl<'a> Execution<'a> {
    // Keep inactive branch construction out of the poll frame that remains
    // live while the selected solver calls back into original model evaluation.
    #[inline(never)]
    pub(super) fn select(
        self,
        solver: runmat_builtins::OptimSolver,
    ) -> impl Future<Output = BuiltinResult<Outcome>> + 'a {
        let Self {
            model,
            formulation,
            starting,
            options,
            products,
            policy,
            cooperation,
        } = self;
        match solver {
            runmat_builtins::OptimSolver::Fmincon => Either::Left(Either::Left(
                constrained::solve(model, formulation, starting, options, policy, cooperation)
                    .map(|result| result.map(Outcome::constrained)),
            )),
            runmat_builtins::OptimSolver::Fminunc => Either::Left(Either::Right(
                unconstrained::solve(model, formulation, starting, options, policy, cooperation)
                    .map(|result| result.map(Outcome::unconstrained)),
            )),
            runmat_builtins::OptimSolver::Lsqnonlin => Either::Right(Either::Left(
                residual::solve(
                    model,
                    formulation,
                    starting,
                    options,
                    products,
                    policy,
                    cooperation,
                )
                .map(|result| result.map(Outcome::residual)),
            )),
            _ => Either::Right(Either::Right(std::future::ready(Err(
                policy.invalid("unsupported nonlinear model solver family")
            )))),
        }
    }
}
