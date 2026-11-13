//! Original sampled A/B/H and signed edge circulation; no nodal interpolation or re-solve.
use super::*;
use crate::analysis::*;
use runmat_analysis_fea::{
    contracts::{ExecutionBackend, SolverMethod, SolverPreconditioner},
    planning::resource::FeaResourcePhase,
    results::field::RecoveredFieldArtifact,
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend, FeaRunResult,
};

pub(super) fn admit(
    completed: &MagneticStudyCompletion,
    mesh: &SolverMeshArtifact,
    options: &AnalysisMagneticRunOptions,
    fields: RecoveredFieldArtifact,
    run_id: &str,
    context: &OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, String> {
    let owner =
        crate::analysis::structural_study::lifecycle::accuracy_completion::classify_accuracy;
    crate::analysis::structural_study::lifecycle::accuracy_completion::validate_publication_chain(
        completed.host_workload_digest,
        &FeaResourcePhase::Magnetostatic(completed.workload.identity().map_err(|e| e.to_string())?),
        &completed.accuracy,
        &completed.baseline_usage,
        &completed.terminal_usage,
        &completed.publication,
    )?;
    let accuracy = owner(&completed.accuracy);
    let run = FeaRunResult {
        identity: completed
            .workload
            .formulation(mesh)
            .map_err(|e| e.to_string())?,
        requested_backend: ComputeBackend::Cpu,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::SparseDirectLdlt,
        preconditioner: SolverPreconditioner::None,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        diagnostics: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: completed.observation_results.clone(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: fields
            .into_fields()
            .into_iter()
            .map(|field| field.into_storage())
            .collect(),
    };
    let provenance = RunProvenance {
        requested_backend: run.requested_backend,
        execution_backend: run.execution_backend,
        solver_backend: run.solver_backend,
        solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
        solver_host_sync_count: run.solver_host_sync_count,
        precision_mode: contracts::format_precision_mode(options.precision_mode),
        deterministic_mode: options.deterministic_mode,
        solver_method: run.solver_method,
        preconditioner: run.preconditioner,
        quality_policy: contracts::format_quality_policy(options.quality_policy),
        fallback_events: Vec::new(),
        effective_memory_ceiling_bytes: None,
        static_solve_path: None,
    };
    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: run_id.into(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(mesh)),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence: accuracy.solver_convergence,
        result_quality: accuracy.result_quality,
        run_status: accuracy.run_status,
        numerically_acceptable: accuracy.numerically_acceptable,
        quality_reasons: accuracy.quality_reasons,
        provenance,
    };
    Ok(OperationEnvelope::new(
        AnalysisOperation::Electromagnetic.name(),
        AnalysisOperation::Electromagnetic.version(),
        context,
        result,
    ))
}
