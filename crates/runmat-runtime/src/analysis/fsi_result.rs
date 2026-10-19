//! Runtime result materialization for the existing FEA-owned partitioned FSI solve.
use runmat_analysis_fea::physics::fsi::FsiCouplingResult;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::{
    AnalysisRunIdentity, ComputeBackend, ExecutionBackend, FeaRunResult, SolverMethod,
    SolverPreconditioner,
};

pub(super) fn materialize(
    identity: AnalysisRunIdentity,
    backend: ComputeBackend,
    coupling: FsiCouplingResult,
) -> FeaRunResult {
    FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::FsiPartitionedProjection,
        preconditioner: SolverPreconditioner::InterfaceRelaxation,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        diagnostics: coupling.diagnostics,
        spatial_projection_evidence: Vec::new(),
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: coupling.fields,
    }
}
