//! Materialize retained canonical thermal field partitions for the public run, never solve again.
use crate::analysis::ThermalStudyCompletion;
use crate::analysis::*;
use runmat_analysis_fea::results::field::reconstruct_recovered_field;

pub(in crate::analysis) fn admit(
    completed: &ThermalStudyCompletion,
    source: &crate::analysis::ThermalStudyObjectSource,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisThermalRunOptions,
    run_id: &str,
    context: &crate::operations::OperationContext,
) -> Result<crate::operations::OperationEnvelope<AnalysisRunResult>, String> {
    let manifest = &completed.manifest;
    manifest.validate().map_err(|error| error.to_string())?;
    crate::analysis::structural_study::lifecycle::accuracy_completion::validate_publication_chain(
        completed.host_workload_digest,
        &runmat_analysis_fea::planning::resource::FeaResourcePhase::ThermalSolve(
            manifest.source_workload,
        ),
        &completed.accuracy,
        &completed.baseline_usage,
        &completed.terminal_usage,
        &completed.publication,
    )?;
    let accuracy =
        crate::analysis::structural_study::lifecycle::accuracy_completion::classify_accuracy(
            &completed.accuracy,
        );
    let mut histories: [Vec<runmat_analysis_core::AnalysisField>; 5] =
        std::array::from_fn(|_| Vec::new());
    for entries in &manifest.fields {
        for (kind, entry) in entries.iter().enumerate() {
            let partitions = entry
                .partitions
                .iter()
                .map(|descriptor| {
                    let partition = completed
                        .field_partitions
                        .iter()
                        .find(|partition| partition.identity == descriptor.partition)
                        .ok_or("thermal field partition is missing")?
                        .clone();
                    if partition.identity != descriptor.partition
                        || partition.source_indices.first() != Some(&descriptor.first_source_index)
                        || partition.source_indices.len() != descriptor.entity_count
                    {
                        return Err("thermal field partition differs from manifest".into());
                    }
                    Ok(partition)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let field = reconstruct_recovered_field(
                entry.field,
                entry.definition.clone(),
                entry.sample,
                partitions,
                match kind {
                    0 => manifest.ordered_node_ids.len(),
                    4 => manifest.interface_node_ids.len(),
                    _ => manifest.ordered_element_ids.len(),
                },
            )
            .map_err(|error| error.to_string())?
            .into_storage();
            histories[kind].push(field);
        }
    }
    // Canonical thermal fields live in the histories below. Duplicating the final
    // sample in run.fields violates the shared artifact store's unique field IDs.
    let mut run = manifest.run.clone();
    crate::analysis::study_result::thermal_observation_admission::project(
        completed, source, &mut run,
    )?;
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
    let [temperature_snapshots, temperature_gradient_snapshots, heat_flux_snapshots, energy_storage_rate_snapshots, interface_temperature_snapshots] =
        histories;
    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: run_id.into(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(mesh)),
        modal_results: None,
        buckling_results: None,
        thermal_results: Some(ThermalResultsData {
            thermal_payload_version: "thermal_results/v1".into(),
            time_points_s: manifest.time_points_s.clone(),
            temperature_snapshots,
            temperature_gradient_snapshots,
            heat_flux_snapshots,
            energy_storage_rate_snapshots,
            interface_node_ids: manifest.interface_node_ids.clone(),
            interface_temperature_snapshots,
            residual_norms: manifest.residual_norms.clone(),
            reference_temperature_k: runmat_types::physical::Temperature::new(
                manifest.reference_temperature_k.canonical_value(),
            ),
            initial_state: manifest.initial_state.clone(),
        }),
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
    Ok(crate::operations::OperationEnvelope::new(
        AnalysisOperation::Thermal.name(),
        AnalysisOperation::Thermal.version(),
        context,
        result,
    ))
}
