use runmat_analysis_core::PartialAccessPolicy;
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisPayloadPartition};
use runmat_analysis_fea::results::field::RecoveredFieldSample;
use runmat_analysis_fea::results::observation::{
    ObservationHistory, ObservationHistoryIdentity, ObservationHistoryManifest,
    ObservationHistorySample,
};
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};

use crate::{
    import_analysis_objects, import_analysis_partition, prepare_analysis_objects,
    prepare_partitioned_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisPartitionImport, ImportedAnalysisPartition, PreparedAnalysisObjects,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ImportedObservationHistoryStorage {
    Partitioned(ImportedAnalysisPartition),
    Whole {
        objects: PreparedAnalysisObjects,
        history: Box<ObservationHistory>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservationHistory {
    pub manifest: ObservationHistoryManifest,
    pub storage: ImportedObservationHistoryStorage,
    pub root: ValueRef,
    pub payload_digest: Digest,
    pub access: ArtifactAccess,
}

pub fn prepare_observation_history_objects(
    history: &ObservationHistory,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    history
        .validate()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let manifest = history.manifest();
    match manifest.partial_access {
        PartialAccessPolicy::ManifestAndChunks => {
            let manifest_bytes = manifest
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            let sample_bytes = history
                .samples()
                .iter()
                .map(|sample| {
                    sample
                        .canonical_bytes()
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut partitions = Vec::with_capacity(sample_bytes.len() + 1);
            partitions.push(AnalysisPayloadPartition {
                logical_identity: manifest.identity.digest(),
                payload: &manifest_bytes,
            });
            partitions.extend(history.samples().iter().zip(&sample_bytes).map(
                |(sample, bytes)| AnalysisPayloadPartition {
                    logical_identity: sample.identity.digest(),
                    payload: bytes,
                },
            ));
            prepare_partitioned_analysis_objects(
                AnalysisNumericalObjectKind::History,
                manifest.identity.digest(),
                &partitions,
                chunk_bytes,
                namespace,
                limits,
            )
        }
        PartialAccessPolicy::WholeArtifact => {
            let bytes = history
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::History,
                manifest.identity.digest(),
                &bytes,
                chunk_bytes,
                namespace,
                limits,
            )
        }
    }
}

pub struct ObservationHistoryImport<'a> {
    pub root: &'a ValueRef,
    pub expected_identity: ObservationHistoryIdentity,
    pub expected_payload_digest: Digest,
    pub partial_access: PartialAccessPolicy,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub fn import_observation_history(
    source: &impl CacheImport,
    import: ObservationHistoryImport<'_>,
) -> AnalysisBridgeResult<ImportedObservationHistory> {
    let expected_identity = import.expected_identity.digest();
    let (manifest, storage) = match import.partial_access {
        PartialAccessPolicy::ManifestAndChunks => {
            let catalog = import_analysis_partition(
                source,
                AnalysisPartitionImport {
                    root: import.root,
                    expected_kind: AnalysisNumericalObjectKind::History,
                    expected_logical_identity: expected_identity,
                    expected_payload_digest: import.expected_payload_digest,
                    partition_identity: expected_identity,
                    access: import.access,
                    limits: import.limits,
                },
            )?;
            let manifest = ObservationHistoryManifest::from_canonical_bytes(&catalog.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            (
                manifest,
                ImportedObservationHistoryStorage::Partitioned(catalog),
            )
        }
        PartialAccessPolicy::WholeArtifact => {
            let objects = import_analysis_objects(
                source,
                import.root,
                AnalysisNumericalObjectKind::History,
                expected_identity,
                import.expected_payload_digest,
                import.access,
                import.limits,
            )?;
            let history = ObservationHistory::from_canonical_bytes(&objects.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            let manifest = history.manifest().clone();
            (
                manifest,
                ImportedObservationHistoryStorage::Whole {
                    objects,
                    history: Box::new(history),
                },
            )
        }
    };
    if manifest.identity != import.expected_identity
        || manifest.partial_access != import.partial_access
    {
        return Err(AnalysisBridgeError::Identity(
            "imported history differs from its expected identity or access policy",
        ));
    }
    Ok(ImportedObservationHistory {
        manifest,
        storage,
        root: import.root.clone(),
        payload_digest: import.expected_payload_digest,
        access: import.access.clone(),
    })
}

pub fn import_observation_history_sample(
    source: &impl CacheImport,
    history: &ImportedObservationHistory,
    coordinate: RecoveredFieldSample,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ObservationHistorySample> {
    let entry = history
        .manifest
        .samples
        .iter()
        .find(|entry| entry.coordinate == coordinate)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("requested history sample is not retained".into())
        })?;
    let sample = match &history.storage {
        ImportedObservationHistoryStorage::Partitioned(_) => {
            let imported = import_analysis_partition(
                source,
                AnalysisPartitionImport {
                    root: &history.root,
                    expected_kind: AnalysisNumericalObjectKind::History,
                    expected_logical_identity: history.manifest.identity.digest(),
                    expected_payload_digest: history.payload_digest,
                    partition_identity: entry.sample.digest(),
                    access: &history.access,
                    limits,
                },
            )?;
            ObservationHistorySample::from_canonical_bytes(&imported.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        }
        ImportedObservationHistoryStorage::Whole { history, .. } => history
            .samples()
            .iter()
            .find(|sample| sample.identity == entry.sample)
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "whole history omits a manifest sample",
            ))?,
    };
    if sample.identity != entry.sample
        || sample.coordinate != entry.coordinate
        || sample.result.logical_identity != entry.result
    {
        return Err(AnalysisBridgeError::Identity(
            "history sample differs from its semantic manifest entry",
        ));
    }
    Ok(sample)
}
