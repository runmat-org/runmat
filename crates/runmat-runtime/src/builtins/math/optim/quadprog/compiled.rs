//! Typed model entry into the existing convex QP engine. No public argument
//! wrapper, numerical probing, or second factorization/certificate authority.
use super::*;
mod geometry;
#[cfg(test)]
mod tests;
mod validation;
pub(in crate::builtins::math::optim) use model::{ActiveTermination, Method, Outcome};
pub(in crate::builtins::math::optim) use validation::{
    CertificateCandidate, RestrictedCurvature, Validation,
};
pub(super) use validation::{SuppliedModel, Terminal, WarmSupplied};

pub(in crate::builtins::math::optim) struct Problem {
    /// Canonical upper triangle of the full symmetric Hessian.
    pub p: Matrix,
    pub f: Vec<f64>,
    pub a_ineq: Matrix,
    pub b_ineq: Vec<f64>,
    pub a_eq: Matrix,
    pub b_eq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Products {
    Primal,
    WithMultipliers,
}
pub(in crate::builtins::math::optim) const fn algorithm(outcome: &Outcome) -> &'static str {
    match outcome.method {
        Method::InteriorPoint => ALGORITHM,
        Method::ActiveSet(_) => "active-set",
    }
}
pub(in crate::builtins::math::optim) struct Prepared {
    problem: Problem,
    controls: options::Controls,
}
impl Prepared {
    pub(in crate::builtins::math::optim) fn algorithm(&self) -> runmat_builtins::OptimAlgorithm {
        self.controls.algorithm
    }
}
pub(in crate::builtins::math::optim) async fn prepare(
    problem: Problem,
    options: Option<&Value>,
    timing: Option<&SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    cooperation.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(), NAME,
        "finite host-memory execution is unsupported; quadratic factorization and certificate workspace have no complete finite bound",
    )?;
    geometry::validate(&problem, cooperation).await?;
    let mut controls = options::Controls::parse_borrowed(options, cooperation).await?;
    controls.bind_timing(timing, cooperation)?;
    Ok(Prepared { problem, controls })
}
pub(in crate::builtins::math::optim) async fn solve<V: Validation>(
    prepared: Prepared,
    initial: Vec<f64>,
    products: Products,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let Prepared { problem, controls } = prepared;
    solve::solve_validated(
        model::Problem {
            p: problem.p,
            f: problem.f,
            a: problem.a_ineq,
            b: problem.b_ineq,
            aeq: problem.a_eq,
            beq: problem.b_eq,
            lb: problem.lb,
            ub: problem.ub,
            x0: initial,
            controls,
        },
        products,
        cooperation,
        validation,
    )
    .await
}
