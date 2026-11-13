//! Authenticate the complete retained closure, including native fields and authored
//! observations, before writing public artifacts.
use super::*;
use runmat_analysis_core::{AnalysisProcedure, IndependentDomain, PhysicsFamily};
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, ObservationHistorySourceContext},
    results::{
        field::RecoveredFieldArtifact,
        observation::{
            ObservationArtifact, ObservationFieldCatalogSource, ObservationPublicationEntry,
        },
    },
};

pub(super) fn validate(
    completed: &ElectricalStudyCompletion,
    source: &ElectricalStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisElectricalRunOptions,
    definition: Digest,
) -> Result<RecoveredFieldArtifact, String> {
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root) {
        return Err("electrical completion root is absent from its exact catalog".into());
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
        return Err("electrical publication differs from authenticated compiled definition".into());
    }
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err("electrical publication requires one fresh independent authored step".into());
    };
    if completed.workload.model != completed.compiled.model
        || completed.workload.step_id != step.step_id
        || step.family != PhysicsFamily::Electromagnetic
        || step.procedure != AnalysisProcedure::Static
        || step.independent_domain != IndependentDomain::Static
    {
        return Err(
            "electrical publication differs from its original authored static workload".into(),
        );
    }
    validate_options(options, completed.workload.relative_residual_target)?;
    let plan = runmat_analysis_fea::study::plan_electrical_study(
        &completed.compiled,
        runmat_analysis_fea::study::ElectricalStudyPlanningPolicy {
            element_temperature_k: completed.workload.element_temperature_k.clone(),
            operating_time_s: completed.workload.operating_time_s,
        },
    )
    .map_err(|e| e.to_string())?;
    if plan != completed.workload {
        return Err("electrical publication workload differs from exact authored planning".into());
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
        .native_field_request(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    // Copy already retained native values once; no operator evaluation or re-solve.
    let fields = completed
        .result
        .recover_native_fields(&completed.workload, mesh, &recovery)
        .map_err(|e| e.to_string())?;
    if fields.manifest().map_err(|e| e.to_string())? != completed.fields {
        return Err(
            "electrical native field manifest differs from original retained physical values"
                .into(),
        );
    }
    validate_observations(completed, step, mesh, &recovery)?;
    let request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        completed.result.producer(),
        AccuracySourceKind::ElectricalStatic,
        completed
            .result
            .source_result_identity()
            .map_err(|e| e.to_string())?,
        completed.observation_catalog.as_ref(),
        step.accuracy.clone(),
    )
    .map_err(|e| e.to_string())?;
    let receipt = runmat_analysis_fea::execution::AccuracySourceReceipt::ElectricalStatic(
        completed.numerical_receipt.clone(),
    );
    let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &request,
        receipt.evidence(&request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if expected_accuracy != completed.accuracy {
        return Err(
            "electrical accuracy differs from exact authored plan and accepted receipt".into(),
        );
    }
    let requests = completed
        .result
        .state_publication_requests()
        .map_err(|e| e.to_string())?;
    if requests.len() != completed.states.len() {
        return Err("electrical publication state inventory differs".into());
    }
    for (request, state) in requests.iter().zip(&completed.states) {
        state.validate().map_err(|e| e.to_string())?;
        // Copy retained native arrays through the shared state authority; this does not
        // assemble an operator, rerun the solve, or reconstruct an absent physical field.
        let expected = runmat_analysis_fea::execution::CanonicalSemanticState::from_electrical_static_publication(
            request, &completed.result,
        ).map_err(|e| e.to_string())?;
        if state.producer != request.producer
            || state.source_result != request.source_result
            || state.topology_digest != request.topology_digest
            || state.state_id != request.declaration.state_id
            || state.kind != request.declaration.kind
            || state.completeness != request.declaration.completeness
            || state.persistence != request.declaration.persistence
            || state.payload.electrical_static().is_none()
            || state != &expected
        {
            return Err(
                "electrical publication state differs from its exact producer result".into(),
            );
        }
    }
    Ok(fields)
}

fn validate_observations(
    completed: &ElectricalStudyCompletion,
    step: &runmat_analysis_core::ProcedureStep,
    mesh: &SolverMeshArtifact,
    recovery: &runmat_analysis_fea::execution::FieldRecoveryRequest,
) -> Result<(), String> {
    let Some(catalog) = &completed.observation_catalog else {
        if step.observations.is_empty() && completed.observation_results.is_empty() {
            return Ok(());
        }
        return Err(
            "electrical authored observations require their original publication catalog".into(),
        );
    };
    let ObservationFieldCatalogSource::VolumeMeshRecovery {
        recovery_request,
        manifest,
        step: catalog_step,
        ..
    } = catalog.source().source()
    else {
        return Err("electrical observations require their original volume-mesh source".into());
    };
    if recovery_request.as_ref() != recovery
        || manifest.as_ref() != &completed.fields
        || catalog_step.as_ref() != step
    {
        return Err(
            "electrical observation catalog differs from exact retained native fields".into(),
        );
    }
    catalog
        .validate_for(step, ObservationHistorySourceContext::VolumeMesh { mesh })
        .map_err(|e| e.to_string())?;
    catalog
        .validate_source_result(
            completed.result.producer(),
            AccuracySourceKind::ElectricalStatic,
            completed
                .result
                .source_result_identity()
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    let mut seen = std::collections::BTreeSet::new();
    for entry in catalog.entries() {
        let ObservationPublicationEntry::Sample(sample) = entry else {
            return Err("electrical static histories are not admitted".into());
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
                    return Err(
                        "electrical observation result coverage or identity differs".to_string()
                    );
                }
                Ok(matching[0].clone())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let artifact =
            ObservationArtifact::from_manifest_and_results(sample.manifest.clone(), results)
                .map_err(|e| e.to_string())?;
        sample
            .validate_artifact(&artifact)
            .map_err(|e| e.to_string())?;
    }
    if seen.len() != completed.observation_results.len() {
        return Err("electrical completion contains unpublished observation results".into());
    }
    Ok(())
}

fn validate_options(options: &AnalysisElectricalRunOptions, residual: f64) -> Result<(), String> {
    if options.residual_target != residual
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err(
            "electrical publication options differ from completed numerical controls".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_controls_cannot_substitute_precision_or_residual() {
        let mut options = AnalysisElectricalRunOptions {
            residual_target: 1e-8,
            ..Default::default()
        };
        assert!(validate_options(&options, 1e-8).is_ok());
        assert!(validate_options(&options, 1e-9).is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(&options, 1e-8).is_err());
    }
}
