//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    import_acoustic_response_study_completion, prepare_acoustic_response_study_root,
    CompletedAcousticResponseStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{
    AcousticResponseStudyCompletion, AcousticResponseStudyObjectSource, AcousticResponseStudySource,
};

struct RetainedAcousticResponseSource<S> {
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    completed: CompletedAcousticResponseStudy,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedAcousticResponseSource<S> {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        crate::bounded_object_read::read_overlay_admitted(
            &self.store,
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            maximum_bytes,
            admit,
        )
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        crate::bounded_object_read::read_overlay(
            &self.store,
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            maximum_bytes,
        )
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            return Ok(Some(bytes.clone()));
        }
        self.store.read_verified(digest)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedAcousticResponseSource<S> {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.artifact_access
            .validate_reference(
                reference,
                runmat_execution::value::ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(|error| error.to_string())?;
        let bytes = CacheImport::read_verified(self, reference.logical_digest)
            .map_err(|error| error.to_string())?
            .ok_or("acoustic response source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("acoustic response source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> AcousticResponseStudySource
    for RetainedAcousticResponseSource<S>
{
    fn authenticate(&self, candidate: &AcousticResponseStudyCompletion) -> Result<(), String> {
        let replay = import_acoustic_response_study_completion(
            self,
            self.completed.preparation().map_err(|e| e.to_string())?,
            self.completed.result_closures().to_vec(),
        )
        .map_err(|error| error.to_string())?;
        let expected = completion(&replay, self)?;
        if &expected != candidate {
            return Err(
                "runtime acoustic response completion differs from authenticated replay".into(),
            );
        }
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(())
    }
}

fn completion(
    completed: &CompletedAcousticResponseStudy,
    source: &impl CacheImport,
) -> Result<AcousticResponseStudyCompletion, String> {
    let root =
        prepare_acoustic_response_study_root(completed).map_err(|error| error.to_string())?;
    Ok(AcousticResponseStudyCompletion {
        host_workload_digest: completed
            .result_closures()
            .first()
            .ok_or("acoustic response completion omits solve")?
            .workload
            .identity
            .digest(),
        compiled: completed.compiled.clone(),
        workload: completed.workload.clone(),
        result: completed.result.result.clone(),
        numerical_receipt: completed.result.numerical_receipt.clone(),
        observation_catalog: completed.observations.catalog.clone(),
        observation_results: completed
            .observations
            .samples
            .iter()
            .flat_map(|sample| {
                sample.manifest.results.iter().map(move |entry| {
                    runmat_analysis_execution::import_observation_result(
                        source,
                        sample,
                        &entry.observation_id,
                        completed
                            .preparation()
                            .map_err(|error| error.to_string())?
                            .output_policy
                            .inventory_limits(),
                    )
                    .map_err(|error| error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        observation_histories: completed
            .observations
            .histories
            .iter()
            .map(|finalized| {
                history(
                    source,
                    &finalized.history,
                    completed
                        .preparation()
                        .map_err(|error| error.to_string())?
                        .output_policy
                        .inventory_limits(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        series_reductions: completed
            .observations
            .histories
            .iter()
            .filter_map(|finalized| finalized.series_reduction.clone())
            .collect(),
        accuracy: completed.accuracy.artifact.clone(),
        states: completed
            .states
            .iter()
            .map(|state| state.state.clone())
            .collect(),
        baseline_usage: completed.baseline_usage.evidence.clone(),
        terminal_usage: completed.terminal_usage.evidence.clone(),
        publication: completed.publication.result.clone(),
        frozen_project_digest: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .frozen_project_digest,
        root: root.root,
        catalog: root.catalog,
    })
}

fn history(
    source: &impl CacheImport,
    imported: &runmat_analysis_execution::ImportedObservationHistory,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> Result<runmat_analysis_fea::results::observation::ObservationHistory, String> {
    if let runmat_analysis_execution::ImportedObservationHistoryStorage::Whole { history, .. } =
        &imported.storage
    {
        return Ok((**history).clone());
    }
    let samples = imported
        .manifest
        .samples
        .iter()
        .map(|entry| {
            runmat_analysis_execution::import_observation_history_sample(
                source,
                imported,
                entry.coordinate,
                limits,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    runmat_analysis_fea::results::observation::ObservationHistory::from_manifest_and_samples(
        imported.manifest.clone(),
        samples,
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn convert(
    completed: CompletedAcousticResponseStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<
    (
        AcousticResponseStudyCompletion,
        AcousticResponseStudyObjectSource,
    ),
    String,
> {
    let root =
        prepare_acoustic_response_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedAcousticResponseSource {
        artifact_access: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .artifact_access,
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
        completed,
        store,
    };
    let dto = completion(&source.completed, &source)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticResponseStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn acoustic_response_study_completion_from_source(
    completed: CompletedAcousticResponseStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<
    (
        AcousticResponseStudyCompletion,
        AcousticResponseStudyObjectSource,
    ),
    String,
> {
    let source = RetainedAcousticResponseSource {
        artifact_access: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .artifact_access,
        completed,
        store: source,
        objects: BTreeMap::new(),
    };
    let dto = completion(&source.completed, &source)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticResponseStudyObjectSource::new(source)))
}
