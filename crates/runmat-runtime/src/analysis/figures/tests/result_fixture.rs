//! Original result fixture shared by rendering contract tests.
use super::*;

pub(super) fn simple_run_result(
    fields: Vec<AnalysisField>,
    render_topology: AnalysisRenderTopology,
) -> AnalysisRunResult {
    let identity = test_run_identity();
    AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: "run_test".to_string(),
        run: runmat_analysis_fea::FeaRunResult {
            accelerator_evidence: Vec::new(),
            identity,
            requested_backend: runmat_analysis_fea::ComputeBackend::Cpu,
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend:
                runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            solver_host_sync_count: 0,
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_convergence: None,
            structural_static_balance: None,
            structural_constraint_result: None,
            structural_connector_result: None,
            observations: Vec::new(),
            observation_histories: Vec::new(),
            series_reductions: Vec::new(),
            fields,
        },
        render_topology: Some(render_topology),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: crate::analysis::contracts::QualityGate::Pass,
        solver_convergence: crate::analysis::contracts::QualityGate::Pass,
        result_quality: crate::analysis::contracts::QualityGate::Pass,
        run_status: crate::analysis::contracts::RunStatus::Completed,
        numerically_acceptable: true,
        quality_reasons: Vec::new(),
        provenance: crate::analysis::contracts::RunProvenance {
            requested_backend: runmat_analysis_fea::ComputeBackend::Cpu,
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend:
                runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: "double".to_string(),
            deterministic_mode: true,
            solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            quality_policy: "strict".to_string(),
            fallback_events: Vec::new(),
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
    }
}
