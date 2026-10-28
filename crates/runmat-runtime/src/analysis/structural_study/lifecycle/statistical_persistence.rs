use std::collections::HashSet;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};

use super::accuracy_completion::CompletedAccuracy;
use crate::analysis::{AnalysisRunResult, AnalysisSolverOptions, QualityGate, RunProvenance};

pub(super) struct StatisticalPersistenceRequest<'a> {
    pub identity: &'a runmat_analysis_fea::AnalysisRunIdentity,
    pub run_id: &'a str,
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub solver_method: runmat_analysis_fea::SolverMethod,
    pub solver_options: &'a AnalysisSolverOptions,
    pub solver_mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub accuracy: &'a CompletedAccuracy,
    pub fields: Vec<AnalysisField>,
}

pub(super) struct StatisticalPersistence {
    pub provenance: RunProvenance,
    pub run: AnalysisRunResult,
}

pub(super) fn admit_statistical_persistence(
    request: StatisticalPersistenceRequest<'_>,
) -> Result<StatisticalPersistence, String> {
    validate_fields(&request.fields)?;
    let (deterministic_mode, policy) = match request.solver_options {
        AnalysisSolverOptions::StructuralResponseSpectrum(options) => {
            (options.deterministic_mode, options.quality_policy)
        }
        AnalysisSolverOptions::StructuralRandomResponse(options) => {
            (options.deterministic_mode, options.quality_policy)
        }
        _ => return Err("statistical persistence received non-statistical solver options".into()),
    };
    let quality_policy = match policy {
        crate::analysis::QualityPolicy::Strict => "strict",
        crate::analysis::QualityPolicy::Balanced => "balanced",
        crate::analysis::QualityPolicy::Exploratory => "exploratory",
    };
    let provenance = RunProvenance {
        requested_backend: request.backend,
        execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
        solver_backend:
            runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_host_sync_count: 0,
        precision_mode: "fp64".into(),
        deterministic_mode,
        solver_method: request.solver_method,
        preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
        quality_policy: quality_policy.into(),
        fallback_events: Vec::new(),
        effective_memory_ceiling_bytes: None,
        static_solve_path: None,
    };
    let run = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: request.run_id.to_string(),
        run: runmat_analysis_fea::FeaRunResult {
            identity: request.identity.clone(),
            requested_backend: request.backend,
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend:
                runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_method: request.solver_method,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            solver_host_sync_count: 0,
            accelerator_evidence: Vec::new(),
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_convergence: None,
            structural_static_balance: None,
            structural_constraint_result: None,
            structural_connector_result: None,
            observations: Vec::new(),
            observation_histories: Vec::new(),
            series_reductions: Vec::new(),
            fields: request.fields,
        },
        render_topology: crate::analysis::render_topology_from_solver_mesh(Some(
            request.solver_mesh,
        )),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence: request.accuracy.solver_convergence,
        result_quality: request.accuracy.result_quality,
        run_status: request.accuracy.run_status,
        numerically_acceptable: request.accuracy.numerically_acceptable,
        quality_reasons: request.accuracy.quality_reasons.clone(),
        provenance: provenance.clone(),
    };
    Ok(StatisticalPersistence { provenance, run })
}

fn validate_fields(fields: &[AnalysisField]) -> Result<(), String> {
    if fields.is_empty() {
        return Err("statistical result has no public fields".into());
    }
    let mut ids = HashSet::with_capacity(fields.len());
    for field in fields {
        let expected = field
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| format!("statistical field `{}` shape overflows", field.field_id))?;
        let AnalysisFieldValues::HostF64(values) = &field.values else {
            return Err(format!(
                "statistical field `{}` is not canonically host-backed",
                field.field_id
            ));
        };
        if field.field_id.is_empty()
            || !ids.insert(field.field_id.as_str())
            || field.shape.is_empty()
            || field.shape.contains(&0)
            || values.len() != expected
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(format!(
                "statistical field `{}` is empty, duplicated, or invalid",
                field.field_id
            ));
        }
    }
    Ok(())
}
