//! Stream original block projections; complete point history remains in the immutable root.
use super::*;
use crate::analysis::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

pub(super) fn admit(
    completed: &NonlinearSolidStudyCompletion,
    source: &NonlinearSolidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    options: &AnalysisNonlinearRunOptions,
    run_id: &str,
    context: &OperationContext,
    staging: &mut storage::AnalysisPublicationBuilder,
    retained_input: &storage::publication::PreparationReservation,
) -> Result<OperationEnvelope<AnalysisRunResult>, storage::AnalysisPublicationError> {
    structural_study::lifecycle::accuracy_completion::validate_publication_chain(
        completed.host_workload_digest,
        &runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralNonlinearPipeline,
        &completed.accuracy,
        &completed.baseline_usage,
        &completed.terminal_usage,
        &completed.publication,
    )?;
    let ceiling = staging
        .source_workspace_with_preparation(retained_input, completed)?
        .min(
            completed.workload.model.sequence.steps[0]
                .accuracy
                .resources
                .resident_byte_budget(),
        );
    let limits = CanonicalLimits::content_bounded(
        usize::try_from(ceiling).map_err(|_| storage::AnalysisPublicationError::ResourceLimit {
            message: "nonlinear host workspace exceeds address space".into(),
        })?,
        65_536,
        64,
    );
    let mut history = None;
    let mut terminal = None;
    for ordinal in 0..completed.block_count {
        // Only the final run is returned. Earlier original fields remain in history.
        drop(terminal.take());
        let retained = profile_serialized(&(&history, &terminal, mesh, options), limits)
            .map_err(storage::AnalysisPublicationError::from)?
            .materialization_resident_bytes_estimate;
        let original = source.materialize_block(completed, ordinal, retained, ceiling)?;
        let combined = profile_serialized(
            &(&history, &terminal, &original, completed, mesh, options),
            limits,
        )
        .map_err(storage::AnalysisPublicationError::from)?;
        if combined.materialization_resident_bytes_estimate > ceiling {
            return Err(storage::AnalysisPublicationError::ResourceLimit {
                message: "original nonlinear runtime fields exceed captured publication workspace"
                    .into(),
            });
        }
        let (run, next) = crate::analysis::nonlinear_result::data(original);
        terminal = Some(run);
        if let Some(history) = &mut history {
            crate::analysis::nonlinear_result::append(history, next)?;
        } else {
            history = Some(next);
        }
    }
    let run = terminal.ok_or("nonlinear completion has no original block projections")?;
    let history = history.ok_or("nonlinear original history absent")?;
    if history.load_factors.len()
        != completed
            .terminal_block
            .checkpoint
            .work()
            .accepted_increments()
    {
        return Err("nonlinear public history omits original accepted outputs".into());
    }
    let accuracy =
        structural_study::lifecycle::accuracy_completion::classify_accuracy(&completed.accuracy);
    let provenance = RunProvenance {
        requested_backend: run.requested_backend,
        execution_backend: run.execution_backend,
        solver_backend: run.solver_backend,
        solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
        solver_host_sync_count: run.solver_host_sync_count,
        precision_mode: contracts::format_precision_mode(options.precision_mode),
        deterministic_mode: false,
        solver_method: run.solver_method,
        preconditioner: run.preconditioner,
        quality_policy: contracts::format_quality_policy(QualityPolicy::Strict),
        fallback_events: Vec::new(),
        effective_memory_ceiling_bytes: None,
        static_solve_path: None,
    };
    let output = OperationEnvelope::new(
        AnalysisOperation::Nonlinear.name(),
        AnalysisOperation::Nonlinear.version(),
        context,
        AnalysisRunResult {
            run_id: run_id.into(),
            run,
            render_topology: render_topology_from_solver_mesh(Some(mesh)),
            modal_results: None,
            buckling_results: None,
            thermal_results: None,
            transient_results: None,
            nonlinear_results: Some(history),
            observation_scalars: Vec::new(),
            model_validity: QualityGate::Pass,
            solver_convergence: accuracy.solver_convergence,
            result_quality: accuracy.result_quality,
            run_status: accuracy.run_status,
            numerically_acceptable: accuracy.numerically_acceptable,
            quality_reasons: accuracy.quality_reasons,
            provenance,
        },
    );
    staging.admit_preparation(&output)?;
    Ok(output)
}
