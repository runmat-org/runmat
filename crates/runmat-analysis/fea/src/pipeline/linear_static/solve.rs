use crate::{
    assembly::AssemblySummary,
    contracts::{FeaRunError, LinearStaticSolveOptions},
    progress::check_cancelled,
    solve::{
        backend::build_backend,
        linear::{solve_linear_system_with_execution, LinearSolveResult},
    },
};

pub(super) fn solve_static_system(
    summary: &AssemblySummary,
    options: &LinearStaticSolveOptions,
) -> Result<LinearSolveResult, FeaRunError> {
    let algebra_backend = build_backend(options.algebra_backend_kind);
    if options.initial_displacement.as_ref().is_some_and(|values| {
        values.len() != summary.dof_count || values.iter().any(|value| !value.is_finite())
    }) {
        return Err(FeaRunError::InvalidModel(format!(
            "initial displacement state must contain exactly {} finite solver DOFs",
            summary.dof_count
        )));
    }
    let result = solve_linear_system_with_execution(
        summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
        options.initial_displacement.as_deref(),
        options.local_execution,
    )
    .map_err(|error| {
        if crate::is_fea_cancelled() {
            FeaRunError::Cancelled
        } else {
            FeaRunError::Solve(error.to_string())
        }
    })?;
    check_cancelled("fea.run_linear_static")?;
    Ok(result)
}
