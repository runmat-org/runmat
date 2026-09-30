//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    import_acoustic_study_completion, prepare_acoustic_study_root, CompletedAcousticStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{
    AcousticStudyCompletion, AcousticStudyObjectSource, AcousticStudySource,
};

struct RetainedAcousticSource<S> {
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    completed: CompletedAcousticStudy,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedAcousticSource<S> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            return Ok(Some(bytes.clone()));
        }
        self.store.read_verified(digest)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedAcousticSource<S> {
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
            .ok_or("acoustic source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("acoustic source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> AcousticStudySource for RetainedAcousticSource<S> {
    fn authenticate(&self, candidate: &AcousticStudyCompletion) -> Result<(), String> {
        let replay = import_acoustic_study_completion(
            self,
            self.completed.preparation().map_err(|e| e.to_string())?,
            self.completed.result_closures().to_vec(),
        )
        .map_err(|error| error.to_string())?;
        let expected = completion(&replay)?;
        if &expected != candidate {
            return Err("runtime acoustic completion differs from authenticated replay".into());
        }
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(())
    }
}

fn completion(completed: &CompletedAcousticStudy) -> Result<AcousticStudyCompletion, String> {
    let root = prepare_acoustic_study_root(completed).map_err(|error| error.to_string())?;
    Ok(AcousticStudyCompletion {
        host_workload_digest: completed
            .result_closures()
            .first()
            .ok_or("acoustic completion omits solve")?
            .workload
            .identity
            .digest(),
        compiled: completed.compiled.clone(),
        workload: completed.workload.clone(),
        result: completed.result.result.clone(),
        numerical_receipt: completed.result.numerical_receipt.clone(),
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

pub(crate) fn convert(
    completed: CompletedAcousticStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<(AcousticStudyCompletion, AcousticStudyObjectSource), String> {
    let root = prepare_acoustic_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedAcousticSource {
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
    let dto = completion(&source.completed)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn acoustic_study_completion_from_source(
    completed: CompletedAcousticStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<(AcousticStudyCompletion, AcousticStudyObjectSource), String> {
    let source = RetainedAcousticSource {
        artifact_access: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .artifact_access,
        completed,
        store: source,
        objects: BTreeMap::new(),
    };
    let dto = completion(&source.completed)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticStudyObjectSource::new(source)))
}
