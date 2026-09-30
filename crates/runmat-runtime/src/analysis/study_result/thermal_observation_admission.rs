//! Admission and projection of retained observations; no evaluation or history gathering.
use crate::analysis::{ThermalStudyCompletion, ThermalStudyObjectSource};
use runmat_analysis_core::ObservationRequest;
use runmat_analysis_fea::{
    execution::ObservationHistorySourceContext,
    results::observation::{ObservationFieldCatalogSource, ObservationPublicationEntry},
    FeaRunResult,
};
use runmat_meshing_core::SolverMeshArtifact;

pub(in crate::analysis) fn validate(
    completed: &ThermalStudyCompletion,
    mesh: &SolverMeshArtifact,
) -> Result<(), String> {
    let step = completed
        .compiled
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == completed.workload.step_id)
        .ok_or("thermal observation producer is absent from compiled study")?;
    let Some(publication) = &completed.observations else {
        return if step.observations.is_empty() {
            Ok(())
        } else {
            Err("thermal completion omits authored observations".into())
        };
    };
    publication.validate(&completed.catalog)?;
    publication
        .catalog
        .validate_for(
            step,
            ObservationHistorySourceContext::Thermal {
                workload: &completed.workload,
                mesh,
            },
        )
        .map_err(|error| error.to_string())?;
    match publication.catalog.source().source() {
        ObservationFieldCatalogSource::Thermal { manifest }
            if manifest.as_ref() == &completed.manifest =>
        {
            Ok(())
        }
        _ => Err("thermal observation catalog names another solver result".into()),
    }
}

pub(in crate::analysis) fn project(
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
    run: &mut FeaRunResult,
) -> Result<(), String> {
    let Some(publication) = &completed.observations else {
        return Ok(());
    };
    let step = completed
        .compiled
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == completed.workload.step_id)
        .ok_or("thermal observation producer is absent")?;
    let catalog = &publication.catalog;
    for entry in catalog.entries() {
        match entry {
            ObservationPublicationEntry::Sample(sample) => {
                for (entry, evidence) in sample
                    .manifest
                    .results
                    .iter()
                    .zip(&sample.partition_evidence)
                {
                    // History results also occur in sample artifacts, but stay catalog-backed.
                    if step.observations.iter().any(|request| matches!(request,
                        ObservationRequest::History(spec) if spec.observation_id == entry.observation_id)) {
                        continue;
                    }
                    let result = source.read_observation(
                        catalog.identity(),
                        sample.manifest.identity,
                        &entry.observation_id,
                    )?;
                    result.validate().map_err(|error| error.to_string())?;
                    if result.logical_identity != entry.result
                        || &result.partition_evidence != evidence
                    {
                        return Err(
                            "thermal observation differs from authenticated publication".into()
                        );
                    }
                    run.observations.push(result);
                }
            }
            ObservationPublicationEntry::History(history) => {
                if let Some(series) = &history.series {
                    let result =
                        source.read_series_reduction(catalog.identity(), series.identity)?;
                    result.validate().map_err(|error| error.to_string())?;
                    if result.identity != series.identity
                        || result.source_history != history.history.identity
                        || result.observation_id != history.history.observation_id
                    {
                        return Err("thermal reduction differs from authenticated history".into());
                    }
                    run.series_reductions.push(result);
                }
            }
        }
    }
    Ok(())
}
