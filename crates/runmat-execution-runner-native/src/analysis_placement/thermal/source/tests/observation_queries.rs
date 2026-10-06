use super::*;
use runmat_analysis_core::{ObservationId, PartialAccessPolicy};
use runmat_analysis_fea::execution::AnalysisObjectManifest;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct RecordingObjects {
    objects: Arc<Mutex<BTreeMap<Digest, Vec<u8>>>>,
    reads: Arc<Mutex<Vec<Digest>>>,
}
impl CacheImport for RecordingObjects {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.lock().unwrap().push(digest);
        Ok(self.objects.lock().unwrap().get(&digest).cloned())
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let objects = self.objects.lock().unwrap();
        if objects
            .get(&digest)
            .is_some_and(|bytes| bytes.len() as u64 > maximum)
        {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "recorded thermal object exceeds caller read allowance".into(),
            ));
        }
        self.reads.lock().unwrap().push(digest);
        Ok(objects.get(&digest).cloned())
    }
}

pub(super) fn check_persistence(
    run_id: &str,
    run: &runmat_runtime::analysis::AnalysisRunResult,
    dto: &ThermalStudyCompletion,
    reader: &ThermalStudyObjectSource,
    source: &RetainedThermalSource<FilesystemObjectStore>,
) {
    use runmat_runtime::analysis::storage::{
        self, AnalysisArtifactStore, FilesystemAnalysisArtifactStore, ObservationHistoryLookup,
    };
    assert_eq!(
        storage::load_observation_publication(run_id).unwrap(),
        dto.observations
    );
    let wrong_directory = tempfile::tempdir().unwrap();
    let wrong_store = FilesystemAnalysisArtifactStore::new(wrong_directory.path().to_path_buf());
    let mut wrong_run = run.clone();
    wrong_run.run.observations.clear();
    wrong_store.persist_run(&wrong_run).unwrap();
    assert!(wrong_store
        .persist_thermal_observations(run_id, dto, reader)
        .is_err());
    let directory = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(directory.path().to_path_buf());
    store.persist_run(run).unwrap();
    store
        .persist_thermal_observations(run_id, dto, reader)
        .unwrap();
    drop(store);
    let reopened = FilesystemAnalysisArtifactStore::new(directory.path().to_path_buf());
    assert_eq!(
        reopened.load_observation_publication(run_id).unwrap(),
        dto.observations
    );
    let metadata = reopened.load_run_metadata(run_id).unwrap().unwrap();
    assert!(metadata.run.observations.is_empty());
    assert!(metadata.run.series_reductions.is_empty());
    assert!(metadata.run.observation_histories.is_empty());
    for retained in &source.completed.observations.histories {
        let manifest = &retained.history.manifest;
        let entry = &manifest.samples[0];
        let operation = runmat_runtime::analysis::analysis_observation_history_sample_by_run_id_op(
            run_id,
            manifest.observation_id.clone(),
            entry.coordinate,
            runmat_runtime::operations::OperationContext::new(None, None),
        )
        .unwrap();
        assert_eq!(operation.data.identity, entry.sample);
        assert!(
            runmat_runtime::analysis::analysis_observation_history_sample_by_run_id_op(
                run_id,
                ObservationId("absent".into()),
                entry.coordinate,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .is_err()
        );
        for actual in [
            storage::load_observation_history_sample(
                run_id,
                &manifest.observation_id,
                entry.coordinate,
            )
            .unwrap(),
            reopened
                .load_observation_history_sample(run_id, &manifest.observation_id, entry.coordinate)
                .unwrap(),
        ] {
            let ObservationHistoryLookup::Sample(sample) = actual else {
                panic!("expected retained observation sample after reopening");
            };
            assert_eq!(sample.identity, entry.sample);
            assert_eq!(sample.result.logical_identity, entry.result);
        }
        if manifest.partial_access == PartialAccessPolicy::ManifestAndChunks {
            let publication = dto.observations.as_ref().unwrap();
            let root = publication
                .histories
                .iter()
                .find(|root| root.history == manifest.identity)
                .unwrap();
            let object = AnalysisObjectManifest::from_canonical_bytes(
                &PublishedObjectSource::read_verified(reader, &root.object.reference).unwrap(),
            )
            .unwrap();
            let chunk_for = |identity: Digest| {
                let partition = object
                    .partitions
                    .iter()
                    .find(|partition| partition.logical_identity == identity)
                    .unwrap();
                object.chunks[partition.first_chunk_ordinal as usize].digest
            };
            let first = chunk_for(entry.sample.digest());
            let later = manifest.samples.last().unwrap();
            let last = chunk_for(later.sample.digest());
            assert_ne!(
                first, last,
                "sample partitions must have independent content"
            );
            for (digest, coordinate, unrelated) in [
                (last, later.coordinate, true),
                (first, entry.coordinate, false),
            ] {
                let path = directory.path().join("objects").join(
                    digest
                        .bytes()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                );
                let bytes = runmat_filesystem::read(&path).unwrap();
                runmat_filesystem::remove_file(&path).unwrap();
                if unrelated {
                    assert!(matches!(
                        reopened
                            .load_observation_history_sample(
                                run_id,
                                &manifest.observation_id,
                                entry.coordinate
                            )
                            .unwrap(),
                        ObservationHistoryLookup::Sample(_)
                    ));
                }
                assert!(reopened
                    .load_observation_history_sample(run_id, &manifest.observation_id, coordinate)
                    .is_err());
                runmat_filesystem::write(&path, b"substituted retained sample").unwrap();
                assert!(reopened
                    .load_observation_history_sample(run_id, &manifest.observation_id, coordinate)
                    .is_err());
                runmat_filesystem::write(&path, bytes).unwrap();
            }
        }
    }
    assert_eq!(
        reopened
            .load_observation_history_sample(
                run_id,
                &ObservationId("absent".into()),
                source.completed.observations.histories[0]
                    .history
                    .manifest
                    .samples[0]
                    .coordinate
            )
            .unwrap(),
        ObservationHistoryLookup::ObservationNotFound
    );
}

pub(super) fn check(
    source: &RetainedThermalSource<FilesystemObjectStore>,
    dto: &ThermalStudyCompletion,
) {
    let publication = dto.observations.as_ref().unwrap();
    publication.validate(&dto.catalog).unwrap();
    let mut forged = dto.clone();
    forged.observations.as_mut().unwrap().histories.pop();
    assert!(source.authenticate(&forged).is_err());
    assert!(forged
        .observations
        .as_ref()
        .unwrap()
        .validate(&dto.catalog)
        .is_err());
    let mut forged = dto.clone();
    forged.observations.as_mut().unwrap().histories[0]
        .object
        .payload_digest = Digest::sha256(b"foreign history");
    assert!(source.authenticate(&forged).is_err());
    let objects = dto
        .catalog
        .objects
        .iter()
        .map(|reference| {
            (
                reference.logical_digest,
                PublishedObjectSource::read_verified(source, reference).unwrap(),
            )
        })
        .collect();
    let store = RecordingObjects {
        objects: Arc::new(Mutex::new(objects)),
        reads: Default::default(),
    };
    let (_, reader) =
        thermal_study_completion_from_source(source.completed.clone(), store.clone()).unwrap();
    let catalog = publication.catalog.identity();
    let sample = publication
        .samples
        .iter()
        .find(|sample| {
            source
                .completed
                .observations
                .samples
                .iter()
                .any(|retained| {
                    retained.manifest.identity == sample.artifact
                        && retained
                            .manifest
                            .results
                            .iter()
                            .any(|entry| entry.observation_id.0 == "temperature-final")
                })
        })
        .unwrap();
    let result = reader
        .read_observation(
            catalog,
            sample.artifact,
            &ObservationId("temperature-final".into()),
        )
        .unwrap();
    assert_eq!(result.observation_id.0, "temperature-final");
    assert!(reader
        .read_observation(catalog, sample.artifact, &ObservationId("absent".into()))
        .is_err());
    let foreign_catalog =
        serde_json::from_value(serde_json::to_value(Digest::sha256(b"foreign catalog")).unwrap())
            .unwrap();
    assert!(reader
        .read_observation(foreign_catalog, sample.artifact, &result.observation_id)
        .is_err());
    for root in &publication.histories {
        let history = source
            .completed
            .observations
            .histories
            .iter()
            .find(|history| history.history.manifest.identity == root.history)
            .unwrap();
        let entry = &history.history.manifest.samples[0];
        store.reads.lock().unwrap().clear();
        let value = reader
            .read_history_sample(catalog, root.history, entry.coordinate)
            .unwrap();
        assert_eq!(value.identity, entry.sample);
        assert_eq!(value.result.logical_identity, entry.result);
        if history.history.manifest.partial_access == PartialAccessPolicy::ManifestAndChunks {
            let object = AnalysisObjectManifest::from_canonical_bytes(
                &store.objects.lock().unwrap()[&root.object.reference.logical_digest],
            )
            .unwrap();
            let partition = object
                .partitions
                .iter()
                .find(|partition| partition.logical_identity == entry.sample.digest())
                .unwrap();
            let chunks = &object.chunks[partition.first_chunk_ordinal as usize
                ..(partition.first_chunk_ordinal + partition.chunk_count) as usize];
            let allowed = std::iter::once(root.object.reference.logical_digest)
                .chain(chunks.iter().map(|chunk| chunk.digest))
                .collect::<std::collections::BTreeSet<_>>();
            assert!(
                store
                    .reads
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|digest| allowed.contains(digest)),
                "selective history read touched unrelated sample/artifact"
            );
            let chunk = chunks[0].digest;
            let bytes = store.objects.lock().unwrap().remove(&chunk).unwrap();
            assert!(reader
                .read_history_sample(catalog, root.history, entry.coordinate)
                .is_err());
            store
                .objects
                .lock()
                .unwrap()
                .insert(chunk, b"substituted sample bytes".to_vec());
            assert!(reader
                .read_history_sample(catalog, root.history, entry.coordinate)
                .is_err());
            store.objects.lock().unwrap().insert(chunk, bytes);
        }
        if let Some(series) = &root.series {
            let result = reader
                .read_series_reduction(catalog, series.identity)
                .unwrap();
            assert_eq!(&result, history.series_reduction.as_ref().unwrap());
            assert_eq!(result.source_history, root.history);
            let bytes = store
                .objects
                .lock()
                .unwrap()
                .remove(&series.object.reference.logical_digest)
                .unwrap();
            assert!(reader
                .read_series_reduction(catalog, series.identity)
                .is_err());
            store.objects.lock().unwrap().insert(
                series.object.reference.logical_digest,
                b"substituted reduction bytes".to_vec(),
            );
            assert!(reader
                .read_series_reduction(catalog, series.identity)
                .is_err());
            store
                .objects
                .lock()
                .unwrap()
                .insert(series.object.reference.logical_digest, bytes);
        }
    }
}
