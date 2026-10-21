//! Exact runtime option alignment with portable static workloads.
use super::*;
use crate::analysis::{AnalysisRunOptions, PreconditionerMode};
pub(super) fn linear_workload_matches_prepared_execution(
    workload: &runmat_analysis_fea::execution::StructuralLinearWorkload,
    execution: &AnalysisExecutionStep,
    options: &AnalysisRunOptions,
) -> bool {
    let expected_preconditioner = match options.preconditioner_mode {
        PreconditionerMode::Auto => SpdPreconditionerKind::Auto,
        PreconditionerMode::Jacobi => SpdPreconditionerKind::Jacobi,
        PreconditionerMode::Ilu => SpdPreconditionerKind::Ilu0,
        PreconditionerMode::Amg => SpdPreconditionerKind::Amg,
    };
    let backend_matches = match execution.backend {
        runmat_analysis_fea::ComputeBackend::Cpu => {
            workload.algebra_backend == LinearAlgebraBackendKind::CpuReference
        }
        runmat_analysis_fea::ComputeBackend::Gpu => matches!(
            workload.algebra_backend,
            LinearAlgebraBackendKind::CpuReference | LinearAlgebraBackendKind::Accelerator
        ),
    };
    workload.backend == execution.backend
        && workload.preconditioner == expected_preconditioner
        && backend_matches
        && workload.observation_execution == ObservationExecutionPolicy::scalable()
        && workload.require_solver_mesh_for_solid
        && workload.state_input.is_none()
        && options.solver_mesh_artifact_path.is_none()
}
