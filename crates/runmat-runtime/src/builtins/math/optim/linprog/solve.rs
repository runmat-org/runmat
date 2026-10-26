//! One vetted sparse numerical substrate; status is checked against original rows.
use super::*;
use clarabel::solver::{DefaultInfo, DefaultSettingsBuilder, DefaultSolver, SolverStatus};
use model::{ConicModel, LinearProgram, LinprogOutcome};

struct Progress<'a> {
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    previous: u32,
    work: u64,
}

pub(super) async fn run(
    model: &ConicModel,
    feasibility: bool,
    cooperation: &mut ScalarCooperation,
    controls: &mut options::Controls,
) -> BuiltinResult<DefaultSolver<f64>> {
    cooperation.completed(0).await?;
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .direct_solve_method("qdldl".into())
        .max_iter(controls.backend_limit()?)
        .tol_gap_abs(1e-10)
        .tol_gap_rel(1e-10)
        .tol_feas(1e-10)
        .tol_infeas_abs(1e-10)
        .tol_infeas_rel(1e-10)
        .build()
        .map_err(|source| {
            build_runtime_error("linprog: invalid backend settings")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
    let cones = model.cones();
    let original = crate::builtins::math::optim::common::conic::Model {
        p: None,
        a: &model.a,
        b: &model.b,
        f: &model.f,
        cones: &cones,
        identity: runmat_builtins::LINPROG_CATALOG_ENTRY.identity,
        objective_scale: model.objective_scale,
    };
    let mut progress = Progress {
        cooperation,
        controls,
        previous: 0,
        work: model.work_units().max(1),
    };
    crate::builtins::math::optim::common::conic::backend::run(
        &original,
        feasibility,
        settings,
        &mut progress,
        async |progress: &mut Progress<'_>, info: &DefaultInfo<f64>| {
            progress.cooperation.completed(0).await?;
            progress.controls.record_iteration(
                &mut progress.previous,
                info.iterations,
                progress.cooperation.runtime(),
            )?;
            progress.cooperation.completed(progress.work).await
        },
    )
    .await
}

pub(super) async fn solve_linprog(
    problem: LinearProgram,
    cooperation: &mut ScalarCooperation,
    controls: &mut options::Controls,
    products: super::compiled::Products,
) -> BuiltinResult<LinprogOutcome> {
    let mut validation = super::compiled::SuppliedRows;
    let execution = solve_validated(problem, cooperation, controls, products, &mut validation)?;
    execution.await
}
mod validated;

// One exact-layout continuation for a complete numerical invocation. Caller
// wrappers retain only this owner, not another inlined copy of its large future.
// The original model, controls, validation and CPU budget move/borrow unchanged.
pub(super) fn solve_validated<'a, V: super::compiled::Validation + 'a>(
    problem: LinearProgram,
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    products: super::compiled::Products,
    validation: &'a mut V,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, LinprogOutcome>> {
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        validated::solve(problem, cooperation, controls, products, validation)
    })
}

async fn incumbent<V: super::compiled::Validation>(
    model: &ConicModel,
    x: Vec<f64>,
    flag: i32,
    iterations: usize,
    message: &str,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<LinprogOutcome> {
    let mut outcome = LinprogOutcome::failure(flag, iterations, message);
    if let Some(point) = certificates::primal(model, &x, cooperation)
        .await?
        .filter(|point| point.normalized <= TOL)
    {
        if !validation.incumbent(&x, cooperation).await? {
            outcome
                .message
                .push_str(" No feasible incumbent was verified against the original model domain.");
            return Ok(outcome);
        }
        outcome.x = Some(x);
        outcome.fval = Some(point.objective);
        outcome.constrviolation = point.violation;
        outcome.primalfeasibility = point.normalized;
        outcome.message.push_str(" Returned x is a verified feasible incumbent; no optimality or multiplier claim is made.");
    } else {
        outcome
            .message
            .push_str(" No verified feasible incumbent is available.");
    }
    Ok(outcome)
}
