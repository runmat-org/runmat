use super::*;

impl InMemoryAnalysisArtifactStore {
    fn observation_run_digest(&self, run_id: &str) -> Result<Digest, String> {
        let runs = self.runs.read().map_err(|_| "run store lock poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
            return Err("observation run is absent".into());
        };
        validate_persisted_run(run, run_id)?;
        Ok(run.run_digest)
    }
    pub(in crate::analysis::storage) fn observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<ThermalObservationPublication>, String> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(None);
        };
        let attachments = self
            .observations
            .read()
            .map_err(|_| "observation store lock poisoned")?;
        attachments
            .get(run_id)
            .map(|attachment| {
                attachment.validate(&run, self.observation_run_digest(run_id)?)?;
                Ok(attachment.publication.clone())
            })
            .transpose()
    }
    pub(in crate::analysis::storage) fn persist_observations(
        &self,
        run_id: &str,
        completed: &ThermalStudyCompletion,
        source: &ThermalStudyObjectSource,
    ) -> Result<(), String> {
        let run = self
            .load_run_metadata(run_id)?
            .ok_or("observation run is absent")?;
        let attachment = ObservationAttachment::prepare(
            &run,
            completed,
            source,
            self.observation_run_digest(run_id)?,
        )?;
        binding::validate_run(
            &self.load_run(run_id)?.ok_or("observation run is absent")?,
            completed,
            source,
        )?;
        let mut attachments = self
            .observations
            .write()
            .map_err(|_| "observation store lock poisoned")?;
        if let Some(existing) = attachments.get(run_id) {
            return if existing == &attachment {
                Ok(())
            } else {
                Err("run already has a different observation catalog".into())
            };
        }
        let mut objects = self
            .chunks
            .write()
            .map_err(|_| "object store lock poisoned")?;
        attachment.ingest(source, |digest, bytes| {
            if let Some(existing) = objects.get(&digest) {
                if existing.as_ref() != bytes.as_slice() {
                    return Err("observation object collision".into());
                }
            } else {
                objects.insert(digest, Arc::from(bytes));
            }
            Ok(())
        })?;
        attachments.insert(run_id.into(), attachment);
        Ok(())
    }

    pub(in crate::analysis::storage) fn observation_sample(
        &self,
        run_id: &str,
        observation_id: &ObservationId,
        sample: RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, String> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(ObservationHistoryLookup::RunNotFound);
        };
        let attachments = self
            .observations
            .read()
            .map_err(|_| "observation store lock poisoned")?;
        let Some(attachment) = attachments.get(run_id) else {
            return Ok(ObservationHistoryLookup::ObservationNotFound);
        };
        attachment.validate(&run, self.observation_run_digest(run_id)?)?;
        let objects = self
            .chunks
            .read()
            .map_err(|_| "object store lock poisoned")?;
        history::load(attachment, observation_id, sample, |digest| {
            Ok(objects.get(&digest).map(|bytes| bytes.to_vec()))
        })
    }
}

impl FilesystemAnalysisArtifactStore {
    fn observation_run_digest(&self, run_id: &str) -> Result<Digest, String> {
        Ok(self
            .read_standard(run_id)?
            .ok_or("observation run is absent")?
            .run_digest)
    }
    pub(in crate::analysis::storage) fn observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<ThermalObservationPublication>, String> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(None);
        };
        Ok(self
            .read_observation_attachment(&run)?
            .map(|attachment| attachment.publication))
    }
    fn read_observation_attachment(
        &self,
        run: &AnalysisRunResult,
    ) -> Result<Option<ObservationAttachment>, String> {
        if !owned_directory_exists(&self.root.join("observations"), "observation attachments")? {
            return Ok(None);
        }
        let path = self.observation_path(&run.run_id)?;
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(path).map_err(|error| error.to_string())?
            }
            Ok(_) => return Err("observation attachment must be a regular file".into()),
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        let attachment: ObservationAttachment =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        attachment.validate(run, self.observation_run_digest(&run.run_id)?)?;
        Ok(Some(attachment))
    }
    fn observation_path(&self, run_id: &str) -> Result<PathBuf, String> {
        validate_run_id_token(run_id)?;
        Ok(self
            .root
            .join("observations")
            .join(format!("{run_id}.json")))
    }

    pub(in crate::analysis::storage) fn persist_observations(
        &self,
        run_id: &str,
        completed: &ThermalStudyCompletion,
        source: &ThermalStudyObjectSource,
    ) -> Result<(), String> {
        let run = self
            .load_run_metadata(run_id)?
            .ok_or("observation run is absent")?;
        let attachment = ObservationAttachment::prepare(
            &run,
            completed,
            source,
            self.observation_run_digest(run_id)?,
        )?;
        binding::validate_run(
            &self.load_run(run_id)?.ok_or("observation run is absent")?,
            completed,
            source,
        )?;
        self.ensure_objects_directory()?;
        attachment.ingest(
            source,
            |digest, bytes| match runmat_filesystem::atomic_create_new(
                self.object_path(digest),
                &bytes,
            )
            .map_err(|error| error.to_string())?
            {
                AtomicCreateResult::Published { .. } => Ok(()),
                AtomicCreateResult::Existing(existing) if existing == bytes => Ok(()),
                AtomicCreateResult::Existing(_) => Err("observation object collision".into()),
            },
        )?;
        ensure_owned_directory(&self.root.join("observations"), "observation attachments")?;
        let bytes = serde_json::to_vec(&attachment).map_err(|error| error.to_string())?;
        match runmat_filesystem::atomic_create_new(self.observation_path(run_id)?, &bytes)
            .map_err(|error| error.to_string())?
        {
            AtomicCreateResult::Published { .. } => Ok(()),
            AtomicCreateResult::Existing(existing) if existing == bytes => Ok(()),
            AtomicCreateResult::Existing(_) => {
                Err("run already has a different observation catalog".into())
            }
        }
    }

    pub(in crate::analysis::storage) fn observation_sample(
        &self,
        run_id: &str,
        observation_id: &ObservationId,
        sample: RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, String> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(ObservationHistoryLookup::RunNotFound);
        };
        let Some(attachment) = self.read_observation_attachment(&run)? else {
            return Ok(ObservationHistoryLookup::ObservationNotFound);
        };
        history::load(&attachment, observation_id, sample, |digest| {
            self.read_chunk(digest)
        })
    }
}
