//! Typed model adapters reuse actual solver owners and original validation.
use super::*;
use crate::builtins::math::optim::common::solver_timing::SolverTiming;
use crate::builtins::math::optim::{
    linprog::compiled as lp, lsqlin::compiled as lsq, quadprog::compiled as qp,
};
use futures::future::Either;
use runmat_builtins::OptimSolver;
pub(in crate::builtins::math::optim::optimmodel) type Solved =
    Either<AlgebraicSolved, EvaluatedSolved>;
pub(in crate::builtins::math::optim::optimmodel) type EvaluatedSolved =
    Either<NonlinearSolved, AlgebraicStopped>;
pub(in crate::builtins::math::optim::optimmodel) struct AlgebraicSolved {
    pub(super) program: Program,
    pub(super) formulation: formulation::Formulation,
    pub(super) rows: Vec<assembly::Rows>,
    pub(super) point: Option<numerical::Point>,
    pub(super) outcome: report::Outcome,
}
pub(in crate::builtins::math::optim::optimmodel) struct NonlinearSolved {
    pub(super) solved: nonlinear::Solved,
    pub(super) formulation: formulation::Formulation,
}
// Evaluated-only stops retain original values but confer no feasible/KKT proof.
pub(in crate::builtins::math::optim::optimmodel) struct AlgebraicStopped {
    pub(super) program: Program,
    pub(super) formulation: formulation::Formulation,
    pub(super) rows: Vec<assembly::Rows>,
    pub(super) candidate: validation::stopped::Candidate,
    pub(super) outcome: report::Outcome,
}
pub(super) fn unverified(policy: Policy, message: &str) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(policy.identity.name)
        .with_identifier(
            runmat_builtins::OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED
                .identifier
                .expect("declared numerical diagnostic"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
mod finish;
mod prepare;
mod selected;

struct Invocation<'a> {
    options: Option<&'a Value>,
    starting: Option<&'a Value>,
    products: Products,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
}
struct Algebraic {
    program: Program,
    formulation: formulation::Formulation,
    rows: Vec<assembly::Rows>,
    assembled: assembly::Assembled,
    solver: OptimSolver,
    quadratic_cost: Option<quadratic::Cost>,
    original_convexity: bool,
}
struct NonlinearPrepared {
    solver: OptimSolver,
    model: nonlinear::Model,
    formulation: formulation::Formulation,
}
struct Prepared<'a> {
    family: Either<Algebraic, NonlinearPrepared>,
    invocation: Invocation<'a>,
}
struct AlgebraicCompleted {
    program: Program,
    formulation: formulation::Formulation,
    rows: Vec<assembly::Rows>,
    retained: Option<numerical::Point>,
    outcome: report::Outcome,
}
struct Completed<'a> {
    family: Either<AlgebraicCompleted, EvaluatedSolved>,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
}
// Both transitions retire the completed child before polling its successor.
// They share the existing admitted call allocation and original cooperation.
pub(in crate::builtins::math::optim::optimmodel) fn solve<'a>(
    model: &'a runmat_value::optimization_model::OptimizationProblemData,
    options: Option<&'a Value>,
    products: Products,
    selected: Option<OptimSolver>,
    starting: Option<&'a Value>,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Solved>> {
    // The original model invocation owns one admitted continuation. Its large
    // compiler/solver state must not become every public forwarder's poll frame.
    // Reject unsupported finite workspace before constructing that state.
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        policy.identity.name,
        "model compilation, exact proof and numerical factorization workspace require unbounded host memory",
    )?;
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        crate::execution::after_result(
            prepare::run(
                model,
                options,
                products,
                selected,
                starting,
                policy,
                cooperation,
            ),
            |prepared| crate::execution::after_result(selected::run(prepared), finish::run),
        )
    })
}
