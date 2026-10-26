//! Typed sparse residual port into the existing least-squares solver.
use super::*;
mod validation;
pub(in crate::builtins::math::optim) use model::{ActiveTermination, Method, Outcome};
pub(in crate::builtins::math::optim) use objective::{Units, Weights};
pub(in crate::builtins::math::optim) use validation::{Certificate, Rows, Validation};
pub(super) use validation::{Supplied, Terminal, WarmSupplied};
pub(in crate::builtins::math::optim) struct Problem {
    pub c: Matrix,
    pub d: Vec<f64>,
    pub a: Matrix,
    pub b: Vec<f64>,
    pub aeq: Matrix,
    pub beq: Vec<f64>,
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    pub weights: Weights,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Products {
    Primal,
    WithMultipliers,
}
impl Products {
    pub(super) fn for_output_count(count: Option<usize>) -> Self {
        if count.is_some_and(|n| n >= 6) {
            Self::WithMultipliers
        } else {
            Self::Primal
        }
    }
}
pub(in crate::builtins::math::optim) struct Prepared {
    problem: Problem,
    controls: options::Controls,
}
impl Prepared {
    pub(in crate::builtins::math::optim) fn equation_tolerance(&self) -> f64 {
        self.controls.feasibility
    }
    pub(in crate::builtins::math::optim) fn algorithm(&self) -> runmat_builtins::OptimAlgorithm {
        self.controls.algorithm
    }
}
pub(in crate::builtins::math::optim) async fn prepare(
    problem: Problem,
    options: Option<&Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    crate::context::host_allocation::require_unbounded_workspace(cooperation.runtime(),NAME,
        "finite host-memory execution is unsupported for sparse residual factorization and exact certificates")?;
    validate(&problem, cooperation).await?;
    let linear_constraints = problem.a.m != 0 || problem.aeq.m != 0;
    let controls =
        options::Controls::parse_borrowed(options, linear_constraints, cooperation).await?;
    Ok(Prepared { problem, controls })
}
pub(in crate::builtins::math::optim) async fn solve<V: Validation>(
    prepared: Prepared,
    starting: Vec<f64>,
    products: Products,
    units: Units,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let Prepared { problem, controls } = prepared;
    super::solve::solve_validated(
        model::Problem {
            c: problem.c,
            d: problem.d,
            a: problem.a,
            b: problem.b,
            aeq: problem.aeq,
            beq: problem.beq,
            lb: problem.lb,
            ub: problem.ub,
            weights: problem.weights,
            x0: starting,
            controls,
        },
        products,
        units,
        cooperation,
        validation,
    )?
    .await
}
pub(in crate::builtins::math::optim) fn algorithm(outcome: &Outcome) -> &'static str {
    outcome.method.name()
}
async fn validate(problem: &Problem, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    let n = problem.c.n;
    if n == 0
        || problem.c.m != problem.d.len()
        || problem.a.n != n
        || problem.aeq.n != n
        || problem.a.m != problem.b.len()
        || problem.aeq.m != problem.beq.len()
        || problem.lb.len() != n
        || problem.ub.len() != n
    {
        return Err(invalid("compiled residual dimensions disagree"));
    }
    for values in [&problem.d, &problem.b, &problem.beq] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Err(invalid("compiled residual rows must be finite"));
            }
            c.completed(chunk.len() as u64).await?;
        }
    }
    for values in [&problem.lb, &problem.ub] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| value.is_nan()) {
                return Err(invalid("compiled bounds cannot contain NaN"));
            }
            c.completed(chunk.len() as u64).await?;
        }
    }
    for matrix in [&problem.c, &problem.a, &problem.aeq] {
        conic::matrix::validate_csc(matrix, c, |message| invalid(message)).await?;
    }
    Ok(())
}
