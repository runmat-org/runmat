//! Immutable observation attachments use the shared run and content-object stores.
use super::*;
use crate::analysis::{
    ThermalObservationPublication, ThermalStudyCompletion, ThermalStudyObjectSource,
};
use runmat_analysis_core::ObservationId;
use runmat_analysis_fea::results::{
    field::RecoveredFieldSample, observation::ObservationHistorySample,
};
use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource};

mod backends;
mod binding;
mod history;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq)]
pub enum ObservationHistoryLookup {
    RunNotFound,
    ObservationNotFound,
    SampleNotFound,
    Sample(Box<ObservationHistorySample>),
}

/// Schema of a persisted observation attachment. Version 2: sampled-history publications name
/// each sample's scope. A stored attachment of another version is legacy: readers treat it as
/// absent (a cache miss) and object reclamation defers rather than infer its closure.
pub(super) const OBSERVATION_ATTACHMENT_SCHEMA_VERSION: u16 = 2;

/// The version field alone, read when a stored attachment does not decode at this schema.
#[derive(Deserialize)]
pub(super) struct ObservationAttachmentVersion {
    pub schema_version: u16,
}

/// A stored attachment as read from the filesystem store.
pub(super) enum StoredObservationAttachment {
    Current(ObservationAttachment),
    /// Written at another schema version; its closure is unknown to this reader.
    Legacy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationAttachment {
    schema_version: u16,
    run_id: String,
    run_digest: Digest,
    publication: ThermalObservationPublication,
    pub(in crate::analysis::storage) objects: PublishedObjectCatalog,
}

impl ObservationAttachment {
    fn prepare(
        run: &AnalysisRunResult,
        completed: &ThermalStudyCompletion,
        source: &ThermalStudyObjectSource,
        run_digest: Digest,
    ) -> Result<Self, String> {
        source.authenticate(completed)?;
        if run.run.identity != completed.manifest.run.identity || run.thermal_results.is_none() {
            return Err("observation attachment names a different thermal run".into());
        }
        let publication = completed
            .observations
            .clone()
            .ok_or("thermal completion has no observation catalog")?;
        publication.validate(&completed.catalog)?;
        if publication.catalog.producer() != completed.manifest.producer {
            return Err("observation catalog has a foreign producer".into());
        }
        Ok(Self {
            schema_version: OBSERVATION_ATTACHMENT_SCHEMA_VERSION,
            run_id: run.run_id.clone(),
            run_digest,
            publication,
            objects: completed.catalog.clone(),
        })
    }

    pub(in crate::analysis::storage) fn validate(
        &self,
        run: &AnalysisRunResult,
        run_digest: Digest,
    ) -> Result<(), String> {
        if self.schema_version != OBSERVATION_ATTACHMENT_SCHEMA_VERSION {
            return Err("observation attachment schema is unsupported".into());
        }
        if self.run_id != run.run_id {
            return Err(format!(
                "observation attachment run ID `{}` differs from original run `{}`",
                self.run_id, run.run_id,
            ));
        }
        if self.run_digest != run_digest {
            return Err(format!(
                "observation attachment digest {} differs from immutable run digest {}",
                self.run_digest, run_digest,
            ));
        }
        self.publication.validate(&self.objects)
    }

    fn ingest(
        &self,
        source: &dyn PublishedObjectSource,
        mut store: impl FnMut(Digest, Vec<u8>) -> Result<(), String>,
    ) -> Result<(), String> {
        for reference in &self.objects.objects {
            let bytes = source.read_verified(reference)?;
            if bytes.len() as u64 != reference.encoded_length
                || Digest::sha256(&bytes) != reference.logical_digest
            {
                return Err("observation source object differs from catalog".into());
            }
            store(reference.logical_digest, bytes)?;
        }
        Ok(())
    }
}

pub fn persist_thermal_observations(
    run_id: &str,
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
) -> Result<(), String> {
    crate::analysis::fea_run_admission::FeaRunScope::check_publication_checked()
        .map_err(|denied| denied.to_string())?;
    capture_artifact_store()?.persist_thermal_observations(run_id, completed, source)
}

pub fn load_observation_publication(
    run_id: &str,
) -> Result<Option<ThermalObservationPublication>, String> {
    capture_artifact_store()?.load_observation_publication(run_id)
}

pub fn load_observation_history_sample(
    run_id: &str,
    observation_id: &ObservationId,
    sample: RecoveredFieldSample,
) -> Result<ObservationHistoryLookup, String> {
    capture_artifact_store()?.load_observation_history_sample(run_id, observation_id, sample)
}

#[cfg(test)]
pub(super) fn prune_attachment(root: &Path, run_id: &str) -> Result<(), String> {
    let store = FilesystemAnalysisArtifactStore::new(root.to_path_buf());
    prune_attachment_scoped(&store.access(true)?, run_id)
}
#[cfg(test)]
pub(super) fn prune_attachment_scoped(
    store: &access::FilesystemAccess<'_>,
    run_id: &str,
) -> Result<(), String> {
    let root = &store.root;
    validate_run_id_token(run_id)?;
    let directory = root.join("observations");
    if !store
        .owned_directory_exists(&directory, "observation attachments")
        .map_err(|e| e.to_string())?
    {
        return Ok(());
    }
    let path = directory.join(format!("{run_id}.json"));
    match store.fs_symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type() == FsFileType::File => store
            .fs_remove_file(path)
            .map_err(|error| error.to_string()),
        Ok(_) => Err("observation attachment must be a regular file".into()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

/// Authenticate/ingest outside the write fence; sources may read this same provider.
pub(super) fn prepare_filesystem_observations(
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
    completed: &ThermalStudyCompletion,
    source: &ThermalStudyObjectSource,
) -> Result<(ObservationAttachment, BTreeMap<Digest, Vec<u8>>), String> {
    let access = store.access(false)?;
    let run = access
        .load_run(run_id)
        .map_err(|e| e.to_string())?
        .ok_or("observation run is absent")?;
    let digest = access
        .read_standard(run_id)
        .map_err(|e| e.to_string())?
        .ok_or("observation run is absent")?
        .run_digest;
    drop(access);
    let attachment = ObservationAttachment::prepare(&run, completed, source, digest)?;
    binding::validate_run(&run, completed, source)?;
    let mut objects = BTreeMap::new();
    attachment.ingest(source, |digest, bytes| {
        if objects.get(&digest).is_some_and(|old| old != &bytes) {
            return Err("observation object collision".into());
        }
        objects.insert(digest, bytes);
        Ok(())
    })?;
    Ok((attachment, objects))
}
