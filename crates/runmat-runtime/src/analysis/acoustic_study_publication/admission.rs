//! Authenticate the complete retained closure before writing public artifacts.
use super::*;
use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily};

pub(super) fn validate(
    completed: &AcousticStudyCompletion,
    source: &AcousticStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisAcousticRunOptions,
    definition: Digest,
) -> Result<(), String> {
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root) {
        return Err("acoustic completion root is absent from its exact catalog".into());
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
        return Err("acoustic publication differs from authenticated compiled definition".into());
    }
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err("acoustic publication requires one fresh independent authored step".into());
    };
    if step.family != PhysicsFamily::Acoustic
        || step.procedure != AnalysisProcedure::HarmonicResponse
        || !step.observations.is_empty()
        || !completed.compiled.outputs.is_empty()
    {
        return Err("acoustic publication requires an exact independent authored workload without unsupported observations".into());
    }
    completed
        .result
        .validate_for(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    let authority = completed
        .result
        .publication_authority()
        .map_err(|e| e.to_string())?;
    completed
        .publication
        .validate_for(&authority)
        .map_err(|e| e.to_string())?;
    completed.publication_roots.admit(
        crate::analysis::FrequencySamplePublicationAccess {
            publication: &completed.publication,
            authority: &authority,
            object_catalog: &completed.publication_catalog,
        },
        |digest| {
            let reference = completed
                .publication_catalog
                .objects
                .iter()
                .find(|reference| reference.logical_digest == digest)
                .ok_or("acoustic publication reference is absent from its catalog")?;
            runmat_execution::value::PublishedObjectSource::read_verified(source, reference)
                .map(Some)
        },
    )?;
    let [first, ..] = completed.entries.as_slice() else {
        return Err("acoustic publication omits frequency entries".into());
    };
    validate_options(options, first.workload.solver.relative_tolerance)?;
    let plan = runmat_analysis_fea::study::plan_acoustic_study(
        &completed.compiled,
        runmat_analysis_fea::study::AcousticStudyPlanningPolicy {
            krylov_restart: first.workload.solver.restart,
            maximum_iterations: first.workload.solver.maximum_iterations,
            power_balance_relative_tolerance: first.workload.power_balance_relative_tolerance,
            uniqueness: first
                .workload
                .solver
                .uniqueness
                .ok_or("authored acoustic publication lacks uniqueness controls")?,
        },
    )
    .map_err(|e| e.to_string())?;
    if plan.frequency_plan() != completed.workload.plan()
        || plan.frequencies().len() != completed.entries.len()
        || completed.result.results().len() != completed.entries.len()
    {
        return Err("acoustic publication workload differs from exact authored planning".into());
    }
    for ((planned, completed), result) in plan
        .frequencies()
        .iter()
        .zip(&completed.entries)
        .zip(completed.result.results())
    {
        if planned.entry() != completed.entry
            || planned.workload() != &completed.workload
            || result != &completed.result
        {
            return Err(
                "acoustic publication entry differs from its exact plan and aggregate result"
                    .into(),
            );
        }
        validate_options(options, completed.workload.solver.relative_tolerance)?;
        validate_entry(completed, mesh, step)?;
    }
    Ok(())
}

fn validate_entry(
    completed: &crate::analysis::AcousticStudyEntryCompletion,
    mesh: &SolverMeshArtifact,
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<(), String> {
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
        runmat_analysis_fea::execution::AccuracySourceKind::AcousticHarmonic,
        completed
            .result
            .source_result_identity()
            .map_err(|e| e.to_string())?,
        None,
        step.accuracy.clone(),
    )
    .map_err(|e| e.to_string())?;
    let receipt = runmat_analysis_fea::execution::AccuracySourceReceipt::AcousticHarmonic(
        completed.numerical_receipt.clone(),
    );
    let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &request,
        receipt.evidence(&request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if expected_accuracy != completed.accuracy {
        return Err(
            "acoustic accuracy differs from exact authored plan and accepted receipt".into(),
        );
    }
    let requests = completed
        .result
        .state_publication_requests()
        .map_err(|e| e.to_string())?;
    if requests.len() != completed.states.len() {
        return Err("acoustic publication state inventory differs".into());
    }
    for (request, state) in requests.iter().zip(&completed.states) {
        state.validate().map_err(|e| e.to_string())?;
        if state.producer != request.producer
            || state.source_result != request.source_result
            || state.topology_digest != request.topology_digest
            || state.state_id != request.declaration.state_id
            || state.kind != request.declaration.kind
            || state.completeness != request.declaration.completeness
            || state.persistence != request.declaration.persistence
            || state.payload.acoustic().is_none()
        {
            return Err("acoustic publication state differs from its exact producer result".into());
        }
    }
    Ok(())
}

fn validate_options(options: &AnalysisAcousticRunOptions, residual: f64) -> Result<(), String> {
    if options.residual_warn_threshold != residual
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err("acoustic publication options differ from completed numerical controls".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_controls_cannot_substitute_precision_or_residual() {
        let mut options = AnalysisAcousticRunOptions {
            residual_warn_threshold: 1e-8,
            ..Default::default()
        };
        assert!(validate_options(&options, 1e-8).is_ok());
        assert!(validate_options(&options, 1e-9).is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(&options, 1e-8).is_err());
    }
}
