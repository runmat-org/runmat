//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    import_thermal_study_completion, prepare_thermal_study_root, CompletedThermalStudy,
};
use runmat_analysis_fea::results::field::RecoveredFieldPartition;
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{
    ThermalStudyCompletion, ThermalStudyObjectSource, ThermalStudySource,
};

mod authentication;
mod observations;
#[cfg(test)]
use runmat_execution_artifact::cache::FilesystemObjectStore;
#[cfg(test)]
mod tests;

struct RetainedThermalSource<S> {
    inventory_limits: runmat_execution_artifact::object::ObjectInventoryLimits,
    completed: CompletedThermalStudy,
    catalog: runmat_execution::value::PublishedObjectCatalog,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedThermalSource<S> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, self.inventory_limits.max_object_bytes)
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let reference = self
            .catalog
            .objects
            .iter()
            .find(|r| r.logical_digest == digest)
            .ok_or_else(|| {
                runmat_execution_artifact::ArtifactError::Identity(
                    "thermal source read outside original catalog".into(),
                )
            })?;
        let maximum = maximum_bytes.min(self.inventory_limits.max_object_bytes);
        if reference.encoded_length > maximum {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "thermal source object exceeds caller read allowance".into(),
            ));
        }
        crate::bounded_object_read::read_overlay_admitted(
            &self.store,
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            reference.encoded_length,
            admit,
        )
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let reference = self
            .catalog
            .objects
            .iter()
            .find(|r| r.logical_digest == digest)
            .ok_or_else(|| {
                runmat_execution_artifact::ArtifactError::Identity(
                    "thermal source read outside original catalog".into(),
                )
            })?;
        let maximum = maximum_bytes.min(self.inventory_limits.max_object_bytes);
        if reference.encoded_length > maximum {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "thermal source object exceeds caller read allowance".into(),
            ));
        }
        let maximum = reference.encoded_length;
        if let Some(bytes) = self.objects.get(&digest) {
            if bytes.len() as u64 > maximum {
                return Err(runmat_execution_artifact::ArtifactError::Limit(
                    "thermal source object exceeds caller read allowance".into(),
                ));
            }
            return Ok(Some(bytes.clone()));
        }
        self.store.read_verified_bounded(digest, maximum)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedThermalSource<S> {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.read_catalog_object(reference)
            .map_err(|error| error.to_string())
    }
}

impl<S: CacheImport + Send + Sync> ThermalStudySource for RetainedThermalSource<S> {
    fn publication_workspace(
        &self,
        candidate: &ThermalStudyCompletion,
        maximum: u64,
    ) -> Result<u64, runmat_runtime::analysis::storage::AnalysisPublicationError> {
        use super::super::source_failure::{bridge, limit};
        if candidate
            .catalog
            .objects
            .iter()
            .any(|r| r.encoded_length > maximum)
        {
            return Err(limit(
                "declared thermal source transport exceeds captured allowance",
            ));
        }
        if candidate.catalog != self.catalog
            || self.catalog.roots.len() != 1
            || self.catalog.roots.first() != Some(&candidate.root)
        {
            return Err("thermal source workspace catalog differs from original closure".into());
        }
        let codec = runmat_canonical_codec::CanonicalLimits::content_bounded(
            usize::try_from(maximum)
                .map_err(|_| limit("thermal source workspace address space"))?,
            65_536,
            64,
        );
        let mut owned = runmat_canonical_codec::profile_serialized(&self.catalog, codec)?
            .materialization_resident_bytes_estimate;
        for bytes in self.objects.values() {
            owned = owned
                .checked_add(bytes.capacity() as u64)
                .ok_or_else(|| limit("thermal source root storage overflow"))?;
        }
        let available = maximum
            .checked_sub(owned)
            .ok_or_else(|| limit("thermal source retained roots exceed allowance"))?;
        let required = self
            .completed
            .publication_workspace_bytes(self, &self.catalog, available)
            .map_err(bridge)?;
        owned
            .checked_add(required)
            .ok_or_else(|| limit("thermal source workspace overflow"))
    }
    fn authenticate_admitted(
        &self,
        candidate: &ThermalStudyCompletion,
        maximum: u64,
    ) -> Result<(), runmat_runtime::analysis::storage::AnalysisPublicationError> {
        self.publication_workspace(candidate, maximum)?;
        self.authenticate_original(candidate)
    }
    fn read_observation(
        &self,
        catalog: observations::CatalogIdentity,
        artifact: observations::ArtifactIdentity,
        observation: &runmat_analysis_core::ObservationId,
    ) -> Result<observations::ObservationResult, String> {
        observations::read_observation(
            &self.completed,
            self,
            self.inventory_limits,
            catalog,
            artifact,
            observation,
        )
    }
    fn read_history_sample(
        &self,
        catalog: observations::CatalogIdentity,
        history: observations::HistoryIdentity,
        coordinate: runmat_analysis_fea::results::field::RecoveredFieldSample,
    ) -> Result<observations::HistorySample, String> {
        observations::read_history_sample(
            &self.completed,
            self,
            self.inventory_limits,
            catalog,
            history,
            coordinate,
        )
    }
    fn read_series_reduction(
        &self,
        catalog: observations::CatalogIdentity,
        series: observations::SeriesIdentity,
    ) -> Result<observations::SeriesResult, String> {
        observations::read_series_reduction(
            &self.completed,
            self,
            self.inventory_limits,
            catalog,
            series,
        )
    }
    fn authenticate(&self, candidate: &ThermalStudyCompletion) -> Result<(), String> {
        self.authenticate_original(candidate)
            .map_err(|error| error.to_string())
    }
}

fn completion(completed: &CompletedThermalStudy) -> Result<ThermalStudyCompletion, String> {
    let root = prepare_thermal_study_root(completed).map_err(|error| error.to_string())?;
    let mut partitions = BTreeMap::new();
    for field in completed.result.manifest.fields.iter().flatten() {
        for descriptor in &field.partitions {
            let identity = descriptor.partition.digest();
            let bytes = completed
                .result
                .objects
                .partition_payload(identity)
                .map_err(|error| error.to_string())?;
            let partition = RecoveredFieldPartition::from_canonical_bytes(&bytes)
                .map_err(|error| error.to_string())?;
            partitions.insert(identity, partition);
        }
    }
    Ok(ThermalStudyCompletion {
        observations: observations::publication(completed, &root.catalog)?,
        host_workload_digest: completed.result.host.workload.identity.digest(),
        compiled: completed.compiled.clone(),
        workload: completed.workload.clone(),
        manifest: completed.result.manifest.clone(),
        numerical_receipt: completed.result.numerical_receipt.clone(),
        field_partitions: partitions.into_values().collect(),
        accuracy: completed.accuracy.artifact.clone(),
        states: completed
            .states
            .iter()
            .map(|state| state.state.clone())
            .collect(),
        baseline_usage: completed.baseline_usage.evidence.clone(),
        terminal_usage: completed.terminal_usage.evidence.clone(),
        publication: completed.publication.result.clone(),
        frozen_project_digest: completed.frozen_project_digest(),
        root: root.root,
        catalog: root.catalog,
    })
}

pub(crate) fn convert(
    completed: CompletedThermalStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<(ThermalStudyCompletion, ThermalStudyObjectSource), String> {
    let root = prepare_thermal_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedThermalSource {
        inventory_limits: completed.output_policy().inventory_limits(),
        catalog: root.catalog,
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
        completed,
        store,
    };
    let dto = completion(&source.completed)?;
    source.authenticate(&dto)?;
    Ok((dto, ThermalStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn thermal_study_completion_from_source(
    completed: CompletedThermalStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<(ThermalStudyCompletion, ThermalStudyObjectSource), String> {
    let root = prepare_thermal_study_root(&completed).map_err(|e| e.to_string())?;
    let source = RetainedThermalSource {
        inventory_limits: completed.output_policy().inventory_limits(),
        catalog: root.catalog,
        completed,
        store: source,
        objects: BTreeMap::new(),
    };
    let dto = completion(&source.completed)?;
    source.authenticate(&dto)?;
    Ok((dto, ThermalStudyObjectSource::new(source)))
}
