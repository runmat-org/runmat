//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    import_magnetic_study_completion, prepare_magnetic_study_root, CompletedMagneticStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{
    MagneticStudyCompletion, MagneticStudyObjectSource, MagneticStudySource,
};

struct RetainedMagneticSource<S> {
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    completed: CompletedMagneticStudy,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedMagneticSource<S> {
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

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedMagneticSource<S> {
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
            .ok_or("magnetic source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("magnetic source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> MagneticStudySource for RetainedMagneticSource<S> {
    fn authenticate(&self, candidate: &MagneticStudyCompletion) -> Result<(), String> {
        let replay = import_magnetic_study_completion(
            self,
            self.completed.preparation().map_err(|e| e.to_string())?,
            self.completed.result_closures().to_vec(),
        )
        .map_err(|error| error.to_string())?;
        let expected = completion(&replay, self)?;
        if &expected != candidate {
            return Err("runtime magnetic completion differs from authenticated replay".into());
        }
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(())
    }
}

fn completion(
    completed: &CompletedMagneticStudy,
    source: &impl CacheImport,
) -> Result<MagneticStudyCompletion, String> {
    let root = prepare_magnetic_study_root(completed).map_err(|error| error.to_string())?;
    let preparation = completed.preparation().map_err(|error| error.to_string())?;
    Ok(MagneticStudyCompletion {
        host_workload_digest: completed
            .result_closures()
            .first()
            .ok_or("magnetic completion omits solve")?
            .workload
            .identity
            .digest(),
        compiled: completed.compiled.clone(),
        workload: completed.workload.clone(),
        result: completed.result.result.clone(),
        numerical_receipt: completed.result.numerical_receipt.clone(),
        fields: completed.fields.manifest.clone(),
        observation_catalog: completed.observation_catalog.clone(),
        observation_results: match &completed.observations {
            Some(observations) => completed
                .workload
                .model
                .sequence
                .steps
                .iter()
                .find(|step| step.step_id == completed.workload.step_id)
                .ok_or("magnetic completion omits its selected authored step")?
                .observations
                .iter()
                .map(|request| {
                    let spec = request.spec().ok_or(
                        "magnetic publication has a non-field observation without a result spec",
                    )?;
                    runmat_analysis_execution::import_observation_result(
                        source,
                        observations,
                        &spec.observation_id,
                        preparation.output_policy.inventory_limits(),
                    )
                    .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        },
        accuracy: completed.accuracy.artifact.clone(),
        states: completed
            .states
            .iter()
            .map(|state| state.state.clone())
            .collect(),
        baseline_usage: completed.baseline_usage.evidence.clone(),
        terminal_usage: completed.terminal_usage.evidence.clone(),
        publication: completed.publication.result.clone(),
        frozen_project_digest: preparation.frozen_project_digest,
        root: root.root,
        catalog: root.catalog,
    })
}

pub(crate) fn convert(
    completed: CompletedMagneticStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<(MagneticStudyCompletion, MagneticStudyObjectSource), String> {
    let root = prepare_magnetic_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedMagneticSource {
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
    Ok((dto, MagneticStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn magnetic_study_completion_from_source(
    completed: CompletedMagneticStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<(MagneticStudyCompletion, MagneticStudyObjectSource), String> {
    let source = RetainedMagneticSource {
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
    Ok((dto, MagneticStudyObjectSource::new(source)))
}
