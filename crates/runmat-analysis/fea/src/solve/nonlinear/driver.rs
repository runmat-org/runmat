//! Public orchestration of immutable-history solid Newton solves.
use super::{
    solid::SolidNewtonError, LoadIncrementSchedule, NonlinearSolveOptions, NonlinearSolveResult,
};
use crate::{assembly::AssemblySummary, ComputeBackend, FeaRunError};
use runmat_analysis_core::AnalysisModel;

mod admission;
pub(crate) use admission::require_fixed_structural_nonlinear;
mod block;
#[cfg(test)]
pub(crate) use block::NonlinearLoadBlockResult;
mod contact;
mod continuation;
pub(crate) use contact::ContactBlockExecution;
mod original_result;
mod output;
pub(crate) use original_result::validate_original_terminal_result;
mod recovery;
pub(crate) use block::{
    run_nonlinear_load_block, NonlinearLoadBlockRequest, NonlinearLoadBlockStart,
};
pub use continuation::{NonlinearLoadContinuation, NonlinearLoadWork};
pub(crate) struct NonlinearExecution<'a> {
    pub backend: ComputeBackend,
    pub contact: Option<ContactBlockExecution<'a>>,
    pub solver_admission: Option<&'a crate::planning::resource::SolverAdmissionResourcePolicy>,
}
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn solve_nonlinear_system(
    model: &AnalysisModel,
    summary: &AssemblySummary,
    options: &NonlinearSolveOptions,
    backend: ComputeBackend,
) -> Result<NonlinearSolveResult, FeaRunError> {
    let schedule = LoadIncrementSchedule::canonical(&[], &[], 0.0, 1.0)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    solve_nonlinear_system_at_boundaries(
        model,
        summary,
        options,
        NonlinearExecution {
            backend,
            contact: None,
            solver_admission: None,
        },
        0.0,
        1.0,
        &schedule,
    )
}

/// Trusted assembled-input boundary. The public pipeline lowers model forcing and
/// constraints before calling this function; callers cannot submit foreign summaries.
pub(crate) fn solve_nonlinear_system_at_boundaries(
    model: &AnalysisModel,
    summary: &AssemblySummary,
    options: &NonlinearSolveOptions,
    execution: NonlinearExecution<'_>,
    start: f64,
    end: f64,
    schedule: &LoadIncrementSchedule,
) -> Result<NonlinearSolveResult, FeaRunError> {
    let block = run_nonlinear_load_block(NonlinearLoadBlockRequest {
        model,
        summary,
        options,
        execution,
        start,
        end,
        schedule,
        input: NonlinearLoadBlockStart::Fresh,
        accepted_block_limit: options.max_accepted_increments,
    })?;
    if !block.complete {
        return Err(FeaRunError::Solve(
            "global accepted J2 increment cap exhausted before completion".into(),
        ));
    }
    Ok(block.result)
}

fn map_error(error: SolidNewtonError) -> FeaRunError {
    match error {
        SolidNewtonError::Cancelled => FeaRunError::Cancelled,
        SolidNewtonError::Usage(error) => FeaRunError::Usage(error),
        SolidNewtonError::Invalid(reason) => FeaRunError::InvalidModel(reason),
        other => FeaRunError::Solve(other.to_string()),
    }
}

fn map_assembly_error(error: crate::assembly::nonlinear_solid::NonlinearSolidError) -> FeaRunError {
    match error {
        crate::assembly::nonlinear_solid::NonlinearSolidError::Cancelled => FeaRunError::Cancelled,
        crate::assembly::nonlinear_solid::NonlinearSolidError::Accounting(error) => {
            FeaRunError::Usage(error)
        }
        other => FeaRunError::Assembly(other.to_string()),
    }
}
