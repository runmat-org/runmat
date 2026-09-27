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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationAttachment {
    schema_version: u16,
    run_id: String,
    run_digest: Digest,
    publication: ThermalObservationPublication,
    objects: PublishedObjectCatalog,
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
            schema_version: 1,
            run_id: run.run_id.clone(),
            run_digest,
            publication,
            objects: completed.catalog.clone(),
        })
    }

    fn validate(&self, run: &AnalysisRunResult, run_digest: Digest) -> Result<(), String> {
        if self.schema_version != 1 || self.run_id != run.run_id || self.run_digest != run_digest {
            return Err("observation attachment differs from immutable run metadata".into());
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
    global_store()
        .read()
        .map_err(|_| "artifact store lock poisoned")?
        .persist_thermal_observations(run_id, completed, source)
}

pub fn load_observation_publication(
    run_id: &str,
) -> Result<Option<ThermalObservationPublication>, String> {
    global_store()
        .read()
        .map_err(|_| "artifact store lock poisoned")?
        .load_observation_publication(run_id)
}

pub fn load_observation_history_sample(
    run_id: &str,
    observation_id: &ObservationId,
    sample: RecoveredFieldSample,
) -> Result<ObservationHistoryLookup, String> {
    global_store()
        .read()
        .map_err(|_| "artifact store lock poisoned")?
        .load_observation_history_sample(run_id, observation_id, sample)
}

pub(super) fn prune_attachment(root: &Path, run_id: &str) -> Result<(), String> {
    validate_run_id_token(run_id)?;
    let directory = root.join("observations");
    if !owned_directory_exists(&directory, "observation attachments")? {
        return Ok(());
    }
    let path = directory.join(format!("{run_id}.json"));
    match fs_symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type() == FsFileType::File => {
            fs_remove_file(path).map_err(|error| error.to_string())
        }
        Ok(_) => Err("observation attachment must be a regular file".into()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}
