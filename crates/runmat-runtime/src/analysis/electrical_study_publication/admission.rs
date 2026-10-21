//! Authenticate the complete retained closure before writing public artifacts.
use super::*;
use runmat_analysis_core::{AnalysisProcedure, IndependentDomain, PhysicsFamily};

pub(super) fn validate(
    completed: &ElectricalStudyCompletion,
    source: &ElectricalStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisElectricalRunOptions,
    definition: Digest,
) -> Result<(), String> {
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
        || !step.observations.is_empty()
        || !completed.compiled.outputs.is_empty()
    {
        return Err("electrical publication requires exact singleton workload without unsupported observations".into());
    }
    validate_options(options, completed.workload.relative_residual_target)?;
    let plan = runmat_analysis_fea::study::plan_electrical_study(
        &completed.compiled,
        runmat_analysis_fea::study::ElectricalStudyPlanningPolicy {},
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
    let request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        completed.result.producer(),
        runmat_analysis_fea::execution::AccuracySourceKind::ElectricalStatic,
        completed
            .result
            .source_result_identity()
            .map_err(|e| e.to_string())?,
        None,
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
