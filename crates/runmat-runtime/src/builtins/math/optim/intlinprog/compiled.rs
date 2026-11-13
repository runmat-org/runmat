//! Typed original-model entry into the existing branch-and-bound owner.
pub(in crate::builtins::math::optim) use super::publication::{nonnegative, Evaluation};
use super::*;
mod geometry;
mod validation;
pub(in crate::builtins::math::optim) use super::domains::Semi;
pub(in crate::builtins::math::optim) use super::output::Published as Outcome;
pub(in crate::builtins::math::optim) use validation::{Node, Publication, Validation};
pub(in crate::builtins::math::optim::intlinprog) use validation::{Supplied, Terminal};
pub(in crate::builtins::math::optim) struct Problem {
    pub f: Vec<f64>,
    pub a_ineq: clarabel::algebra::CscMatrix<f64>,
    pub b_ineq: Vec<f64>,
    pub a_eq: clarabel::algebra::CscMatrix<f64>,
    pub b_eq: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub integers: Vec<usize>,
    pub semis: Vec<Semi>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Products {
    Primal,
    Information,
}
pub(in crate::builtins::math::optim) const fn algorithm() -> &'static str {
    "branch-and-bound"
}
pub(in crate::builtins::math::optim) struct Prepared {
    workspace: model::Workspace,
    controls: controls::Controls,
}
pub(in crate::builtins::math::optim) async fn prepare(
    problem: Problem,
    options: Option<&Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    cooperation.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(cooperation.runtime(), NAME,
        "finite host-memory execution is unsupported; tree, sparse factorization and rational-growth workspaces have no complete finite bound")?;
    let runtime = cooperation.runtime().cloned().ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "compiled intlinprog requires its original execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    geometry::validate(&problem, cooperation).await?;
    let controls = controls::Controls::parse(
        options.cloned().unwrap_or_else(empty),
        &runtime,
        cooperation,
    )
    .await?;
    let mut lower_rows =
        runmat_value::exact_capacity_vec(problem.b_ineq.len()).map_err(allocation)?;
    for _ in 0..problem.b_ineq.len() {
        lower_rows.push(f64::NEG_INFINITY);
        cooperation.completed(1).await?;
    }
    let (a, b, equalities) = normalize::constraints(
        problem.a_ineq,
        lower_rows,
        problem.b_ineq,
        problem.a_eq,
        problem.b_eq,
        cooperation,
    )
    .await?;
    let workspace = model::Workspace::new(
        model::Problem {
            a,
            b,
            equalities,
            f: problem.f,
            lower: problem.lower,
            upper: problem.upper,
            integers: problem.integers,
            semis: problem.semis,
            initial: None,
        },
        cooperation,
    )
    .await?;
    Ok(Prepared {
        workspace,
        controls,
    })
}
pub(in crate::builtins::math::optim) async fn solve<V: Validation>(
    prepared: Prepared,
    initial: Option<Vec<f64>>,
    products: Products,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    cooperation.completed(0).await?;
    let Prepared {
        mut workspace,
        mut controls,
    } = prepared;
    if let Some(point) = &initial {
        if point.len() != workspace.n() {
            return Err(invalid("compiled initial point geometry changed"));
        }
        for value in point {
            if !value.is_finite() {
                return Err(invalid("compiled initial point must be finite"));
            }
            cooperation.completed(1).await?;
        }
    }
    workspace.initial = initial;
    let (outcome, workspace) =
        engine::solve_validated(workspace, &mut controls, cooperation, validation)
            .await
            .map_err(no_replay)?;
    output::prepare(
        outcome,
        &workspace,
        &controls,
        products == Products::Information,
        validation,
        cooperation,
    )
    .await
    .map_err(no_replay)
}
fn no_replay(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
