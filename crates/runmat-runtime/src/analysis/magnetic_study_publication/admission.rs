//! Exact retained magnetic field, observation, state and accuracy authority before public writes.
use super::*;
use runmat_analysis_fea::{
    execution::{
        AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt,
        CanonicalSemanticState, ObservationHistorySourceContext,
    },
    results::{
        field::RecoveredFieldArtifact,
        observation::{ObservationArtifact, ObservationPublicationEntry},
    },
};

pub(super) fn validate(
    completed: &MagneticStudyCompletion,
    source: &MagneticStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisMagneticRunOptions,
    definition: Digest,
) -> Result<RecoveredFieldArtifact, String> {
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root) {
        return Err("magnetic completion root is absent from its exact catalog".into());
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
        return Err("magnetic publication differs from authenticated compiled definition".into());
    }
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err("magnetic publication requires one authored standalone step".into());
    };
    if completed.workload.model != completed.compiled.model
        || completed.workload.step_id != step.step_id
    {
        return Err("magnetic publication differs from original authored workload".into());
    }
    validate_options(options, completed.workload.relative_residual_target)?;
    let expected = runmat_analysis_fea::study::plan_magnetic_study(
        &completed.compiled,
        runmat_analysis_fea::study::MagneticStudyPlanningPolicy {},
    )
    .map_err(|e| e.to_string())?;
    if expected != completed.workload {
        return Err("magnetic publication workload differs from exact authored planning".into());
    }
    completed
        .result
        .validate_for(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.result, mesh)
        .map_err(|e| e.to_string())?;
    let recovery = completed
        .result
        .native_field_request_from_result()
        .map_err(|e| e.to_string())?;
    // Copy already retained values once, preserving native 15-point and oriented-edge storage.
    let fields = completed
        .result
        .recover_native_fields_from_result(&recovery)
        .map_err(|e| e.to_string())?;
    if fields.manifest().map_err(|e| e.to_string())? != completed.fields {
        return Err(
            "magnetic native field manifest differs from original retained physical values".into(),
        );
    }
    match &completed.observation_catalog {
        Some(catalog) => {
            let runmat_analysis_fea::results::observation::ObservationFieldCatalogSource::SampledRecovery {
                recovery_request, manifest, sampling,
            } = catalog.source().source() else {
                return Err("magnetic observations require their original sampled native source".into());
            };
            if recovery_request.as_ref() != &recovery
                || manifest.as_ref() != &completed.fields
                || sampling.as_ref() != completed.result.sampling_topology()
            {
                return Err(
                    "magnetic observation catalog differs from exact retained native fields".into(),
                );
            }
            catalog
                .validate_for(
                    step,
                    ObservationHistorySourceContext::Magnetostatic {
                        workload: &completed.workload,
                        mesh,
                    },
                )
                .map_err(|e| e.to_string())?;
            catalog
                .validate_source_result(
                    completed.result.producer(),
                    AccuracySourceKind::Magnetostatic,
                    completed
                        .result
                        .source_result_identity()
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            let mut seen = std::collections::BTreeSet::new();
            for entry in catalog.entries() {
                let ObservationPublicationEntry::Sample(sample) = entry else {
                    return Err("magnetic static histories are not admitted".into());
                };
                let results = sample
                    .manifest
                    .results
                    .iter()
                    .map(|entry| {
                        let matching = completed
                            .observation_results
                            .iter()
                            .filter(|result| result.logical_identity == entry.result)
                            .collect::<Vec<_>>();
                        if matching.len() != 1 || !seen.insert(entry.result) {
                            return Err("magnetic observation result coverage or identity differs"
                                .to_string());
                        }
                        Ok(matching[0].clone())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let artifact = ObservationArtifact::from_manifest_and_results(
                    sample.manifest.clone(),
                    results,
                )
                .map_err(|e| e.to_string())?;
                sample
                    .validate_artifact(&artifact)
                    .map_err(|e| e.to_string())?;
            }
            if seen.len() != completed.observation_results.len() {
                return Err("magnetic completion contains unpublished observation results".into());
            }
        }
        None if step.observations.is_empty() && completed.observation_results.is_empty() => {}
        None => {
            return Err(
                "magnetic authored observations require their original publication catalog".into(),
            )
        }
    }
    let request = AccuracyEvaluationRequest::new(
        completed.result.producer(),
        AccuracySourceKind::Magnetostatic,
        completed
            .result
            .source_result_identity()
            .map_err(|e| e.to_string())?,
        completed.observation_catalog.as_ref(),
        step.accuracy.clone(),
    )
    .map_err(|e| e.to_string())?;
    let receipt = AccuracySourceReceipt::Magnetostatic(completed.numerical_receipt.clone());
    let expected = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &request,
        receipt.evidence(&request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if expected != completed.accuracy {
        return Err(
            "magnetic accuracy differs from exact authored plan and original receipt".into(),
        );
    }
    let requests = completed
        .result
        .state_publication_requests()
        .map_err(|e| e.to_string())?;
    if requests.len() != completed.states.len() {
        return Err("magnetic publication state inventory differs".into());
    }
    for (request, state) in requests.iter().zip(&completed.states) {
        state.validate().map_err(|e| e.to_string())?;
        let expected =
            CanonicalSemanticState::from_magnetostatic_publication(request, &completed.result)
                .map_err(|e| e.to_string())?;
        if state != &expected || state.payload.magnetostatic().is_none() {
            return Err(
                "magnetic publication state differs from exact original retained result".into(),
            );
        }
    }
    Ok(fields)
}

fn validate_options(options: &AnalysisMagneticRunOptions, residual: f64) -> Result<(), String> {
    if options.residual_target != residual
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err("magnetic publication options differ from original numerical controls".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn magnetic_controls_cannot_substitute_residual_or_precision() {
        let mut options = AnalysisMagneticRunOptions {
            residual_target: 1e-8,
            ..Default::default()
        };
        assert!(validate_options(&options, 1e-8).is_ok());
        assert!(validate_options(&options, 1e-9).is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(&options, 1e-8).is_err());
    }
}
