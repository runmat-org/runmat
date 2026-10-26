//! One admitted equation invocation selects one existing numerical controller.
use super::*;
use crate::builtins::math::optim::common::solver_timing::SolverTiming;
use futures::future::Either;
use runmat_builtins::OptimSolver;
pub(in crate::builtins::math::optim::optimmodel) enum Solved {
    Linear {
        program: Program,
        outcome: linear::Outcome,
    },
    Nonlinear {
        model: Model,
        outcome: nonlinear::Outcome,
        timing: Option<SolverTiming>,
    },
    Scalar {
        model: Model,
        outcome: scalar::Outcome,
        timing: Option<SolverTiming>,
    },
}
struct Invocation<'a> {
    source: Prepared,
    requested: Option<OptimSolver>,
    starting: Option<&'a Value>,
    options: Option<&'a Value>,
    information: bool,
    policy: Policy,
    c: &'a mut ScalarCooperation,
}
pub(in crate::builtins::math::optim::optimmodel) fn solve<'a>(
    problem: &'a runmat_value::optimization_model::EquationProblemData,
    requested: Option<OptimSolver>,
    starting: Option<&'a Value>,
    options: Option<&'a Value>,
    information: bool,
    policy: Policy,
    c: &'a mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Solved>> {
    let runtime = c.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        crate::execution::after_result(
            prepare(
                problem,
                requested,
                starting,
                options,
                information,
                policy,
                c,
            ),
            select,
        )
    })
}
async fn prepare<'a>(
    problem: &runmat_value::optimization_model::EquationProblemData,
    requested: Option<OptimSolver>,
    starting: Option<&'a Value>,
    options: Option<&'a Value>,
    information: bool,
    policy: Policy,
    c: &'a mut ScalarCooperation,
) -> BuiltinResult<Invocation<'a>> {
    c.completed(0).await?;
    let source = Model::prepare(problem, requested, policy, c).await?;
    Ok(Invocation {
        source,
        requested,
        starting,
        options,
        information,
        policy,
        c,
    })
}
fn select<'a>(
    invocation: Invocation<'a>,
) -> impl std::future::Future<Output = BuiltinResult<Solved>> + 'a {
    let Invocation {
        source,
        requested,
        starting,
        options,
        information,
        policy,
        c,
    } = invocation;
    match source {
        Prepared::Linear(program) => Either::Left(async move {
            let outcome =
                linear::solve(&program, starting, options, information, policy, c).await?;
            Ok(Solved::Linear { program, outcome })
        }),
        Prepared::Nonlinear(model) => Either::Right(crate::execution::after_result(
            select_nonlinear(model, requested, starting, options, information, policy, c),
            selected_nonlinear,
        )),
    }
}
struct Nonlinear<'a> {
    model: Model,
    solver: OptimSolver,
    starting: Option<&'a Value>,
    options: Option<&'a Value>,
    timing: Option<SolverTiming>,
    policy: Policy,
    c: &'a mut ScalarCooperation,
}
async fn select_nonlinear<'a>(
    model: Model,
    requested: Option<OptimSolver>,
    starting: Option<&'a Value>,
    options: Option<&'a Value>,
    information: bool,
    policy: Policy,
    c: &'a mut ScalarCooperation,
) -> BuiltinResult<Nonlinear<'a>> {
    let mut bounded = false;
    use runmat_value::optimization_variable::{
        OptimizationBoundRole as Role, OptimizationBoundView,
    };
    for variable in &model.layout.variables {
        if variable.domain
            != runmat_value::optimization_variable::OptimizationVariableDomain::Continuous
        {
            return Err(policy
                .invalid("equation solvers cannot discard an original integer variable domain"));
        }
        if variable.count == 0 {
            continue;
        }
        for role in [Role::Lower, Role::Upper] {
            let free = if matches!(role, Role::Lower) {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            match variable.view.definition().bound(role) {
                OptimizationBoundView::Uniform(value) => bounded |= value != free,
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
    let solver = requested.unwrap_or(if bounded {
        OptimSolver::Lsqnonlin
    } else if model.layout.columns == 1 && model.evidence.rows == 1 {
        OptimSolver::Fzero
    } else {
        OptimSolver::Fsolve
    });
    if solver == OptimSolver::Lsqlin {
        return Err(policy.invalid("lsqlin requires affine original equation expressions"));
    }
    if bounded && matches!(solver, OptimSolver::Fsolve | OptimSolver::Fzero) {
        return Err(
            policy.invalid("the selected equation solver cannot discard original variable bounds")
        );
    }
    if !matches!(
        solver,
        OptimSolver::Fsolve | OptimSolver::Fzero | OptimSolver::Lsqnonlin
    ) {
        return Err(policy.invalid("unsupported equation solver"));
    }
    let timing = if information {
        Some(SolverTiming::start(c.runtime())?)
    } else {
        None
    };
    Ok(Nonlinear {
        model,
        solver,
        starting,
        options,
        timing,
        policy,
        c,
    })
}
fn selected_nonlinear<'a>(
    invocation: Nonlinear<'a>,
) -> impl std::future::Future<Output = BuiltinResult<Solved>> + 'a {
    let Nonlinear {
        model,
        solver,
        starting,
        options,
        timing,
        policy,
        c,
    } = invocation;
    if solver == OptimSolver::Fzero {
        Either::Left(async move {
            let outcome = scalar::solve(&model, starting, options, policy, c).await?;
            Ok(Solved::Scalar {
                model,
                outcome,
                timing,
            })
        })
    } else {
        Either::Right(async move {
            let outcome = nonlinear::solve(&model, solver, starting, options, policy, c).await?;
            Ok(Solved::Nonlinear {
                model,
                outcome,
                timing,
            })
        })
    }
}
