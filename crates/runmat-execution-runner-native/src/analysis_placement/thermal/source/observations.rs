//! Projection and selective reading of already solved observation artifacts, never evaluation.
use super::*;
use runmat_analysis_execution::{
    import_analysis_objects, import_observation_history_sample, import_observation_result,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
pub(super) use runmat_analysis_fea::results::observation::{
    ObservationArtifactIdentity as ArtifactIdentity, ObservationHistoryIdentity as HistoryIdentity,
    ObservationHistorySample as HistorySample,
    ObservationPublicationCatalogIdentity as CatalogIdentity, ObservationResult,
    SeriesReductionResult as SeriesResult, SeriesReductionResultIdentity as SeriesIdentity,
};
use runmat_execution::value::{PublishedObjectCatalog, ValueRefKind};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_runtime::analysis::{
    ThermalObservationHistoryRoot, ThermalObservationObjectRoot, ThermalObservationPublication,
    ThermalObservationSampleRoot, ThermalObservationSeriesRoot,
};

pub(super) fn publication(
    completed: &CompletedThermalStudy,
    objects: &PublishedObjectCatalog,
) -> Result<Option<ThermalObservationPublication>, String> {
    let observations = &completed.observations;
    let Some(catalog) = &observations.catalog else {
        if !observations.samples.is_empty() || !observations.histories.is_empty() {
            return Err("thermal observations have no publication catalog".into());
        }
        return Ok(None);
    };
    let samples = observations
        .samples
        .iter()
        .map(|sample| ThermalObservationSampleRoot {
            artifact: sample.manifest.identity,
            object: ThermalObservationObjectRoot {
                reference: sample.output.root.clone(),
                payload_digest: sample.output.payload_digest,
            },
        })
        .collect();
    let histories = observations
        .histories
        .iter()
        .map(|history| {
            let series = match (&history.series_reduction, &history.series_objects) {
                (Some(series), Some(objects)) => Some(ThermalObservationSeriesRoot {
                    identity: series.identity,
                    object: ThermalObservationObjectRoot {
                        reference: objects
                            .root_reference(
                                &completed.result.host.artifact_access,
                                ValueRefKind::ResultObject,
                            )
                            .map_err(|error| error.to_string())?,
                        payload_digest: objects.manifest.payload_digest,
                    },
                }),
                (None, None) => None,
                _ => return Err("thermal reduction omits its retained artifact".to_string()),
            };
            Ok(ThermalObservationHistoryRoot {
                history: history.history.manifest.identity,
                object: ThermalObservationObjectRoot {
                    reference: history.history.root.clone(),
                    payload_digest: history.history.payload_digest,
                },
                series,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let publication = ThermalObservationPublication {
        catalog: catalog.clone(),
        samples,
        histories,
    };
    publication.validate(objects)?;
    Ok(Some(publication))
}

fn require_catalog(
    completed: &CompletedThermalStudy,
    catalog: CatalogIdentity,
) -> Result<(), String> {
    if completed
        .observations
        .catalog
        .as_ref()
        .map(|value| value.identity())
        != Some(catalog)
    {
        return Err("thermal observation query names another publication catalog".into());
    }
    Ok(())
}

pub(super) fn read_observation(
    completed: &CompletedThermalStudy,
    source: &impl CacheImport,
    limits: ObjectInventoryLimits,
    catalog: CatalogIdentity,
    artifact: ArtifactIdentity,
    observation: &runmat_analysis_core::ObservationId,
) -> Result<ObservationResult, String> {
    require_catalog(completed, catalog)?;
    let sample = completed
        .observations
        .samples
        .iter()
        .find(|sample| sample.manifest.identity == artifact)
        .ok_or("thermal observation query names an unpublished sample artifact")?;
    import_observation_result(source, sample, observation, limits)
        .map_err(|error| error.to_string())
}

pub(super) fn read_history_sample(
    completed: &CompletedThermalStudy,
    source: &impl CacheImport,
    limits: ObjectInventoryLimits,
    catalog: CatalogIdentity,
    history: HistoryIdentity,
    coordinate: runmat_analysis_fea::results::field::RecoveredFieldSample,
) -> Result<HistorySample, String> {
    require_catalog(completed, catalog)?;
    let history = completed
        .observations
        .histories
        .iter()
        .find(|value| value.history.manifest.identity == history)
        .ok_or("thermal observation query names an unpublished history")?;
    if history
        .history
        .manifest
        .samples
        .iter()
        .filter(|sample| sample.coordinate == coordinate)
        .count()
        > 1
    {
        return Err(
            "thermal history coordinate is ambiguous between retained event occurrences".into(),
        );
    }
    import_observation_history_sample(source, &history.history, coordinate, limits)
        .map_err(|error| error.to_string())
}

pub(super) fn read_series_reduction(
    completed: &CompletedThermalStudy,
    source: &impl CacheImport,
    limits: ObjectInventoryLimits,
    catalog: CatalogIdentity,
    series: SeriesIdentity,
) -> Result<SeriesResult, String> {
    require_catalog(completed, catalog)?;
    let history = completed
        .observations
        .histories
        .iter()
        .find(|value| {
            value
                .series_reduction
                .as_ref()
                .is_some_and(|value| value.identity == series)
        })
        .ok_or("thermal observation query names an unpublished reduction")?;
    let expected = history
        .series_reduction
        .as_ref()
        .ok_or("thermal reduction missing")?;
    let retained = history
        .series_objects
        .as_ref()
        .ok_or("thermal reduction root missing")?;
    let access = &completed.result.host.artifact_access;
    let objects = import_analysis_objects(
        source,
        &retained
            .root_reference(access, ValueRefKind::ResultObject)
            .map_err(|error| error.to_string())?,
        AnalysisNumericalObjectKind::SeriesReductionResult,
        series.digest(),
        retained.manifest.payload_digest,
        access,
        limits,
    )
    .map_err(|error| error.to_string())?;
    let result = SeriesResult::from_canonical_bytes(&objects.payload())
        .map_err(|error| error.to_string())?;
    if &result != expected {
        return Err("thermal reduction differs from authenticated history finalization".into());
    }
    Ok(result)
}
