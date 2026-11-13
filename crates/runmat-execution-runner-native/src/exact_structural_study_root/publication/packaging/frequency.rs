//! Frequency numerical closure assembly; no physics is recomputed.
use super::*;

pub(super) fn frequency_closure(
    completion: &crate::CompletedStructuralFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (frequency, objects) = frequency_components(completion, access)?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralFrequency {
        frequency: Box::new(frequency),
        objects,
    })
}

pub(super) fn frequency_components(
    completion: &crate::CompletedStructuralFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<(ExactStructuralFrequencyClosure, Vec<ValueRef>)> {
    if completion.entries.len() != completion.observation_samples.len() {
        return Err(protocol(
            "structural frequency fields and observation samples have different inventories",
        ));
    }
    let (catalog, _) = completion.published_artifacts(access)?;
    let result_root = completion
        .result
        .result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = completion
        .result
        .receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let resource_usage_root = completion
        .resource_usage
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let mut entries = Vec::with_capacity(completion.entries.len());
    for (entry, observations) in completion
        .entries
        .iter()
        .zip(&completion.observation_samples)
    {
        let sample = runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
            entry.entry.result.request().entry().frequency_hz(),
        );
        if observations.sample != sample {
            return Err(protocol(
                "structural frequency observations differ from their numerical entry",
            ));
        }
        let observations_root = observations
            .observations
            .as_ref()
            .map(|observations| observations.output.root.clone());
        entries.push(ExactStructuralFrequencyEntryClosure {
            entry: entry.entry.result.request().entry(),
            source_entry: entry.entry.result.manifest().clone(),
            physical_sample: entry.entry.result.sample().identity(),
            sample,
            fields_root: entry.recovered_fields.output.root.clone(),
            observations_root,
        });
    }
    let mut histories = Vec::with_capacity(completion.histories.len());
    for history in &completion.histories {
        let checkpoint_root = history
            .checkpoint
            .objects
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        histories.push(ExactStructuralFrequencyHistoryClosure {
            observation_id: history.observation_id.clone(),
            checkpoint_root,
            finalization: history
                .finalization
                .as_ref()
                .map(|finalization| {
                    finalization
                        .published_artifacts(access)
                        .map(|(roots, _)| roots)
                })
                .transpose()?,
        });
    }
    Ok((
        ExactStructuralFrequencyClosure {
            result_root: Box::new(result_root),
            receipt_root: Box::new(receipt_root),
            publication: Box::new(completion.publication.clone()),
            entries,
            histories,
            resource_usage_root: Box::new(resource_usage_root),
            objects: catalog.objects.clone(),
        },
        catalog.objects,
    ))
}
