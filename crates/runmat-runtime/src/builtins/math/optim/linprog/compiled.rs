//! Typed compiled-model entry into the existing LP numerical body. This port
//! owns no public Value argument/gather/callback wrapper or second solver.
pub(in crate::builtins::math::optim) use super::model::{
    LinearProgram as Problem, LinprogOutcome as Outcome,
};
use super::*;
mod validation;
pub(super) use validation::{descent, farkas, feasible, SuppliedRows};
pub(in crate::builtins::math::optim) use validation::{CertificateCandidate, RowRole, Validation};

pub(in crate::builtins::math::optim) const fn algorithm() -> &'static str {
    ALGORITHM
}
pub(in crate::builtins::math::optim) const fn validation_tolerance() -> f64 {
    TOL
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Products {
    Primal,
    WithMultipliers,
}
impl Products {
    pub(super) fn for_output_count(count: Option<usize>) -> Self {
        if count.is_some_and(|count| count >= 5) {
            Self::WithMultipliers
        } else {
            Self::Primal
        }
    }
}
mod preparation;
mod selected;
struct Ready<'a, V> {
    problem: Problem,
    controls: super::options::Controls,
    products: Products,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
}
struct Finished<'a> {
    outcome: Outcome,
    controls: super::options::Controls,
    cooperation: &'a mut ScalarCooperation,
}
pub(in crate::builtins::math::optim) fn solve<'a, V: Validation + 'a>(
    problem: Problem,
    options: Option<&'a Value>,
    products: Products,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    crate::execution::after_result(
        preparation::run(problem, options, products, cooperation, validation),
        selected::run,
    )
}

async fn validate(problem: &Problem, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
    let n = problem.f.len();
    if n == 0
        || problem.lb.len() != n
        || problem.ub.len() != n
        || problem.a_ineq.n != n
        || problem.a_eq.n != n
        || problem.a_ineq.m != problem.b_ineq.len()
        || problem.a_eq.m != problem.b_eq.len()
    {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            "compiled model dimensions disagree",
        ));
    }
    for values in [&problem.f, &problem.b_ineq, &problem.b_eq] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    "compiled costs and rows must be finite",
                ));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    for values in [&problem.lb, &problem.ub] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| value.is_nan()) {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    "compiled bounds cannot contain NaN",
                ));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    for matrix in [&problem.a_ineq, &problem.a_eq] {
        crate::builtins::math::optim::common::conic::matrix::validate_csc(
            matrix,
            cooperation,
            |message| linprog_error_with_detail(&LINPROG_ERROR_INVALID_INPUT, message),
        )
        .await?;
    }
    Ok(())
}
