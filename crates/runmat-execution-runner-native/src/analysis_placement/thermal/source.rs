//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    prepare_thermal_study_root, CompletedThermalStudy, ThermalPrestressSource,
};
use runmat_analysis_fea::results::field::RecoveredFieldPartition;
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{
    cache::{CacheImport, FilesystemObjectStore},
    ArtifactResult,
};
use runmat_runtime::analysis::{
    ThermalStudyCompletion, ThermalStudyObjectSource, ThermalStudySource,
};

#[cfg(test)]
mod tests;

struct RetainedThermalSource<S> {
    completed: CompletedThermalStudy,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedThermalSource<S> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            return Ok(Some(bytes.clone()));
        }
        self.store.read_verified(digest)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedThermalSource<S> {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.completed
            .result
            .host
            .artifact_access
            .validate_reference(
                reference,
                runmat_execution::value::ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(|error| error.to_string())?;
        let bytes = CacheImport::read_verified(self, reference.logical_digest)
            .map_err(|error| error.to_string())?
            .ok_or("thermal source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("thermal source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> ThermalStudySource for RetainedThermalSource<S> {
    fn authenticate(&self, candidate: &ThermalStudyCompletion) -> Result<(), String> {
        let replay = ThermalPrestressSource::import(
            self,
            self.completed.preparation(),
            self.completed.result_closures().to_vec(),
        )
        .map_err(|error| error.to_string())?;
        if replay.completed() != &self.completed {
            return Err("thermal completion differs from authenticated replay".into());
        }
        let expected = completion(replay.completed())?;
        if &expected != candidate {
            return Err("runtime thermal completion differs from authenticated replay".into());
        }
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(())
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
        frozen_project_digest: completed.preparation().frozen_project_digest,
        root: root.root,
        catalog: root.catalog,
    })
}

pub(super) fn convert(
    completed: CompletedThermalStudy,
    store: FilesystemObjectStore,
) -> Result<(ThermalStudyCompletion, ThermalStudyObjectSource), String> {
    let root = prepare_thermal_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedThermalSource {
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
    let source = RetainedThermalSource {
        completed,
        store: source,
        objects: BTreeMap::new(),
    };
    let dto = completion(&source.completed)?;
    source.authenticate(&dto)?;
    Ok((dto, ThermalStudyObjectSource::new(source)))
}
