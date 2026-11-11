//! Admission of an original modal result; solver geometry stays borrowed during rendering.
use super::*;

pub(super) fn analysis_admit_modal_run_with_id(
    modal_run: runmat_analysis_fea::FeaModalRunResult,
    backend: ComputeBackend,
    options: AnalysisModalRunOptions,
    context: OperationContext,
    solver_mesh: Option<&SolverMeshArtifact>,
    assigned_run_id: Option<&str>,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let basis = modal_run.basis;
    let run = modal_run.run;
    let mode_shapes = modal_run.mode_shapes;
    let mut fallback_events = Vec::new();
    if backend == ComputeBackend::Gpu && run.execution_backend != ExecutionBackend::Accelerator {
        fallback_events.push(FallbackEvent::solver_backend());
    }
    let solver_convergence = if run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_CONVERGENCE"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info
    }) {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let result_quality = if basis.modes().is_empty() {
        QualityGate::Fail
    } else if basis
        .modes()
        .iter()
        .map(|mode| {
            options
                .convergence
                .gated_residual(mode.residual_norm(), mode.backward_error())
        })
        .fold(0.0_f64, f64::max)
        > options.convergence.algebraic_tolerance()
    {
        QualityGate::Warn
    } else {
        QualityGate::Pass
    };
    let modal_orthogonality_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_ORTHOGONALITY"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let modal_separation_warn = run.diagnostics.iter().any(|item| {
        item.code == "FEA_MODAL_SEPARATION"
            && item.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    let mut quality_reasons = Vec::new();
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "modal solver convergence gate is warning".to_string(),
        });
    }
    if result_quality == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalResidualExceeded,
            detail: format!(
                "modal {} exceeds threshold {}",
                options.convergence.algebraic_measure().label(),
                options.convergence.algebraic_tolerance()
            ),
        });
    }
    if modal_orthogonality_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalOrthogonalityExceeded,
            detail: "modal M-orthogonality off-diagonal threshold exceeded".to_string(),
        });
    }
    if modal_separation_warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::ModalSeparationLow,
            detail: "modal frequency separation threshold is low".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }

    let frequency_basis = ModalFrequencyBasis::NativeEigenSolve;

    let numerically_acceptable = match options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && !quality_reasons.iter().any(|r| {
                    matches!(
                        r.code,
                        QualityReasonCode::ModalOrthogonalityExceeded
                            | QualityReasonCode::ModalSeparationLow
                    )
                })
        }
        QualityPolicy::Exploratory => {
            solver_convergence != QualityGate::Fail && result_quality != QualityGate::Fail
        }
    };
    let run_status = classify_run_status(result_quality, numerically_acceptable);
    let solver_backend = run.solver_backend;
    let execution_backend = run.execution_backend;
    let solver_device_apply_k_ratio = run.solver_device_apply_k_ratio;
    let solver_host_sync_count = run.solver_host_sync_count;
    let solver_method = run.solver_method;
    let selected_preconditioner = run.preconditioner;

    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: assigned_run_id
            .map(str::to_string)
            .unwrap_or_else(storage::next_run_id),
        run,
        render_topology: render_topology_from_solver_mesh(solver_mesh),
        modal_results: Some(ModalResultsData {
            mode_indices: (0..basis.modes().len()).collect(),
            mode_shapes,
            basis,
            mode_units: ModalFrequencyUnits::Hz,
            frequency_basis,
        }),
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        quality_reasons,
        provenance: RunProvenance {
            requested_backend: backend,
            execution_backend,
            solver_backend,
            solver_device_apply_k_ratio,
            solver_host_sync_count,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method,
            preconditioner: selected_preconditioner,
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events,
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
    };

    Ok(OperationEnvelope::new(
        AnalysisOperation::Modal.name(),
        AnalysisOperation::Modal.version(),
        &context,
        result,
    ))
}
