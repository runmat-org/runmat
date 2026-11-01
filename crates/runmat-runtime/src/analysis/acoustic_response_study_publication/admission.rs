//! Authenticate the complete retained closure before writing public artifacts.
use super::*;
use runmat_analysis_core::{lower_authored_observations, AnalysisProcedure, PhysicsFamily};
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, ObservationHistorySourceContext},
    results::observation::{ObservationFieldCatalogSource, ObservationPublicationEntry},
};

pub(super) fn validate(
    completed: &AcousticResponseStudyCompletion,
    source: &AcousticResponseStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisAcousticResponseRunOptions,
    definition: Digest,
) -> Result<(), String> {
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root) {
        return Err("acoustic response completion root is absent from its exact catalog".into());
    }
    completed
        .compiled
        .validate_against(mesh)
        .map_err(|e| e.to_string())?;
    evidence.validate(mesh).map_err(|e| e.to_string())?;
    if completed
        .compiled
        .definition_digest()
        .map_err(|e| e.to_string())?
        != definition
    {
        return Err(
            "acoustic response publication differs from authenticated compiled definition".into(),
        );
    }
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err(
            "acoustic response publication requires one fresh independent authored step".into(),
        );
    };
    if completed.workload.model != completed.compiled.model
        || completed.workload.step_id != step.step_id
        || step.family != PhysicsFamily::Acoustic
        || !matches!(
            step.procedure,
            AnalysisProcedure::ModalExtraction | AnalysisProcedure::TransientResponse
        )
    {
        return Err("acoustic response publication requires exact singleton workload".into());
    }
    if !completed.compiled.outputs.is_empty() {
        let lowered = lower_authored_observations(
            &completed.compiled.outputs,
            &completed.compiled.model.sequence.steps,
        )
        .map_err(|error| error.to_string())?;
        if lowered.len() != step.observations.len()
            || lowered
                .iter()
                .zip(&step.observations)
                .any(|(output, request)| {
                    output.step_id != step.step_id || output.request != *request
                })
        {
            return Err(
                "acoustic response authored outputs differ from exact step observations".into(),
            );
        }
    }
    validate_options(options, completed.workload.solver.relative_tolerance())?;
    let plan = runmat_analysis_fea::study::plan_acoustic_response_study(
        &completed.compiled,
        runmat_analysis_fea::study::AcousticResponseStudyPlanningPolicy::default(),
    )
    .map_err(|e| e.to_string())?;
    if plan != completed.workload {
        return Err(
            "acoustic response publication workload differs from exact authored planning".into(),
        );
    }
    completed
        .result
        .validate_for(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.result, mesh)
        .map_err(|e| e.to_string())?;
    match &completed.observation_catalog {
        Some(catalog) if !step.observations.is_empty() => {
            catalog
                .validate_for(step, ObservationHistorySourceContext::VolumeMesh { mesh })
                .map_err(|error| error.to_string())?;
            catalog
                .validate_source_result(
                    completed.result.producer(),
                    AccuracySourceKind::AcousticResponse,
                    completed
                        .result
                        .source_result_identity()
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
            let ObservationFieldCatalogSource::VolumeMeshRecovery {
                recovery_request, ..
            } = catalog.source().source()
            else {
                return Err(
                    "acoustic response observations require native pressure recovery".into(),
                );
            };
            if recovery_request.as_ref()
                != &completed
                    .result
                    .native_field_request_from_result()
                    .map_err(|error| error.to_string())?
            {
                return Err(
                    "acoustic response observation source differs from retained pressure".into(),
                );
            }
            let published = catalog
                .entries()
                .iter()
                .filter_map(|entry| match entry {
                    ObservationPublicationEntry::History(history) => Some(&history.history),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if published.len() != completed.observation_histories.len()
                || published
                    .iter()
                    .zip(&completed.observation_histories)
                    .any(|(expected, history)| *expected != history.manifest())
            {
                return Err(
                    "acoustic response authored history publication differs from cold history"
                        .into(),
                );
            }
        }
        None if step.observations.is_empty()
            && completed.observation_results.is_empty()
            && completed.observation_histories.is_empty()
            && completed.series_reductions.is_empty() => {}
        _ => {
            return Err(
                "acoustic response authored observations require their complete catalog".into(),
            )
        }
    }
    let request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        completed.result.producer(),
        runmat_analysis_fea::execution::AccuracySourceKind::AcousticResponse,
        completed
            .result
            .source_result_identity()
            .map_err(|e| e.to_string())?,
        completed.observation_catalog.as_ref(),
        step.accuracy.clone(),
    )
    .map_err(|e| e.to_string())?;
    let receipt = runmat_analysis_fea::execution::AccuracySourceReceipt::AcousticResponse(
        completed.numerical_receipt.clone(),
    );
    let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &request,
        receipt.evidence(&request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if expected_accuracy != completed.accuracy {
        return Err(
            "acoustic response accuracy differs from exact authored plan and accepted receipt"
                .into(),
        );
    }
    let requests = completed
        .result
        .state_publication_requests()
        .map_err(|e| e.to_string())?;
    if requests.len() != completed.states.len() {
        return Err("acoustic response publication state inventory differs".into());
    }
    for (request, state) in requests.iter().zip(&completed.states) {
        state.validate().map_err(|e| e.to_string())?;
        // Copy retained native arrays through the shared state authority; this does not
        // assemble an operator, rerun the solve, or reconstruct an absent physical field.
        let expected = runmat_analysis_fea::execution::CanonicalSemanticState::from_acoustic_response_publication(
            request, &completed.result,
        ).map_err(|e| e.to_string())?;
        if state.producer != request.producer
            || state.source_result != request.source_result
            || state.topology_digest != request.topology_digest
            || state.state_id != request.declaration.state_id
            || state.kind != request.declaration.kind
            || state.completeness != request.declaration.completeness
            || state.persistence != request.declaration.persistence
            || state.payload.fields().is_none()
            || state != &expected
        {
            return Err(
                "acoustic response publication state differs from its exact producer result".into(),
            );
        }
    }
    Ok(())
}

fn validate_options(
    options: &AnalysisAcousticResponseRunOptions,
    residual: f64,
) -> Result<(), String> {
    if options.residual_target != residual
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err(
            "acoustic response publication options differ from completed numerical controls".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_controls_cannot_substitute_precision_or_residual() {
        let mut options = AnalysisAcousticResponseRunOptions {
            residual_target: 1e-8,
            ..Default::default()
        };
        assert!(validate_options(&options, 1e-8).is_ok());
        assert!(validate_options(&options, 1e-9).is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(&options, 1e-8).is_err());
    }
}
