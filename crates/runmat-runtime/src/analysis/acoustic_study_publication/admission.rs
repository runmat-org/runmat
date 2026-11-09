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
    {
        return Err("acoustic publication requires an exact independent authored workload".into());
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
            power_balance_relative_tolerance: first.workload.power_balance_relative_tolerance,
            contraction_limit: first.workload.solver.contraction_limit,
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
    validate_histories(completed, mesh, step)
}

/// The histories over the sweep's frequencies: one catalog over every entry's field catalog,
/// with each authored history and its series reduction.
fn validate_histories(
    completed: &AcousticStudyCompletion,
    mesh: &SolverMeshArtifact,
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<(), String> {
    use runmat_analysis_fea::results::observation::{
        ObservationFieldCatalogSource, ObservationPublicationEntry, ObservationPublicationScope,
    };
    let declared = step
        .observations
        .iter()
        .any(|request| matches!(request, runmat_analysis_core::ObservationRequest::History(_)));
    let catalog = match &completed.history_catalog {
        None if !declared && completed.series_reductions.is_empty() => return Ok(()),
        Some(catalog) if declared => catalog,
        _ => return Err("acoustic history catalog differs from the authored step".into()),
    };
    if catalog.scope() != ObservationPublicationScope::EntryHistories {
        return Err("acoustic history catalog has another publication scope".into());
    }
    catalog
        .validate_for(
            step,
            runmat_analysis_fea::execution::ObservationHistorySourceContext::VolumeMesh { mesh },
        )
        .map_err(|e| e.to_string())?;
    let ObservationFieldCatalogSource::EntryHistory { samples } = catalog.source().source() else {
        return Err("acoustic history catalog is not an entry history".into());
    };
    if samples.len() != completed.entries.len()
        || samples
            .iter()
            .zip(&completed.entries)
            .any(|(sample, entry)| {
                entry
                    .observation_catalog
                    .as_ref()
                    .is_some_and(|catalog| catalog.source() != sample)
            })
    {
        return Err("acoustic history catalog differs from the entries' field catalogs".into());
    }
    // Catalog entries are in canonical identifier order; reductions are in authored order.
    let mut matched = 0;
    for entry in catalog.entries() {
        let ObservationPublicationEntry::History(history) = entry else {
            return Err("acoustic history catalog publishes an entry sample".into());
        };
        if let Some(series) = &history.series {
            let mut candidates = completed
                .series_reductions
                .iter()
                .filter(|reduction| reduction.observation_id == history.history.observation_id);
            let reduction = candidates
                .next()
                .ok_or("acoustic history omits its series reduction")?;
            if candidates.next().is_some() {
                return Err("acoustic history has more than one series reduction".into());
            }
            series
                .validate_published_result(reduction, &history.history, &history.request)
                .map_err(|e| e.to_string())?;
            matched += 1;
        }
    }
    if matched != completed.series_reductions.len() {
        return Err("acoustic publication has an undeclared series reduction".into());
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
        completed.observation_catalog.as_ref(),
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
    let sampled = runmat_analysis_fea::results::observation::authored_observations_sample(
        step,
        runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
            completed.workload.frequency.canonical_value(),
        ),
    )
    .map_err(|e| e.to_string())?;
    match &completed.observation_catalog {
        // No authored observation samples this frequency of the sweep.
        None if !sampled && completed.observation_results.is_empty() => {}
        Some(catalog) if sampled => {
            catalog
                .validate_authored_step(step)
                .map_err(|e| e.to_string())?;
            if catalog.scope()
                != runmat_analysis_fea::results::observation::ObservationPublicationScope::Entry
            {
                return Err("acoustic entry catalog has another publication scope".into());
            }
            let sample = runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
                completed.workload.frequency.canonical_value(),
            );
            // The outputs this frequency schedules, in authored order.
            let plan = catalog.plan();
            let mut scheduled = Vec::new();
            for observation in &plan.observations {
                if runmat_analysis_fea::results::observation::resolved_observation_accepts_sample(
                    plan,
                    observation,
                    sample,
                    None,
                )
                .map_err(|e| e.to_string())?
                {
                    scheduled.push(&observation.spec.observation_id);
                }
            }
            if completed.observation_results.len() != scheduled.len()
                || completed
                    .observation_results
                    .iter()
                    .zip(scheduled)
                    .any(|(result, id)| &result.observation_id != id || result.sample != sample)
            {
                return Err(
                    "acoustic outputs differ from the authored observations at this frequency"
                        .into(),
                );
            }
        }
        _ => return Err("acoustic output catalog differs from the authored step".into()),
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
