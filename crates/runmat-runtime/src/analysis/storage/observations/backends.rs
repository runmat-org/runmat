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
        let run = self.load_run(run_id)?.ok_or("observation run is absent")?;
        let attachment = self.prepare_observations(run_id, &run, completed, source)?;
        self.install_observations(run_id, attachment, source)
    }

    /// Binds observations to `materialized`, the run `run_id` read back with its field values
    /// from wherever its writer put the chunks (this store, or a publication's spill area).
    pub(in crate::analysis::storage) fn prepare_observations(
        &self,
        run_id: &str,
        materialized: &AnalysisRunResult,
        completed: &ThermalStudyCompletion,
        source: &ThermalStudyObjectSource,
    ) -> Result<ObservationAttachment, String> {
        let run = self
            .load_run_metadata(run_id)?
            .ok_or("observation run is absent")?;
        let attachment = ObservationAttachment::prepare(
            &run,
            completed,
            source,
            self.observation_run_digest(run_id)?,
        )?;
        binding::validate_run(materialized, completed, source)?;
        Ok(attachment)
    }

    pub(in crate::analysis::storage) fn install_observations(
        &self,
        run_id: &str,
        attachment: ObservationAttachment,
        source: &ThermalStudyObjectSource,
    ) -> Result<(), String> {
        let run = self
            .load_run_metadata(run_id)?
            .ok_or("observation run is absent")?;
        attachment.validate(&run, self.observation_run_digest(run_id)?)?;
        let mut attachments = self
            .observations
            .write()
            .map_err(|_| "observation store lock poisoned")?;
        if self
            .publication_runs
            .read()
            .map_err(|_| "publication ownership lock poisoned")?
            .contains_key(run_id)
        {
            return Err("committed publication cannot accept additional observations".into());
        }
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

impl access::FilesystemAccess<'_> {
    fn observation_run_digest(&self, run_id: &str) -> Result<Digest, AnalysisPublicationError> {
        Ok(self
            .read_standard(run_id)?
            .ok_or("observation run is absent")?
            .run_digest)
    }
    pub(in crate::analysis::storage) fn observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<ThermalObservationPublication>, AnalysisPublicationError> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(None);
        };
        Ok(self
            .read_observation_attachment(&run)?
            .map(|attachment| attachment.publication))
    }
    /// The run's attachment at the current schema; a legacy attachment is a miss.
    pub(in crate::analysis::storage) fn read_observation_attachment(
        &self,
        run: &AnalysisRunResult,
    ) -> Result<Option<ObservationAttachment>, AnalysisPublicationError> {
        let mut attachment = None;
        self.visit_stored_observation_attachment(run, |current| {
            attachment = Some(current);
            Ok(())
        })?;
        Ok(attachment)
    }

    /// The run's stored attachment. One that does not decode at the current schema and names
    /// another schema version is [`ObservationAttachmentReadOutcome::Legacy`]; one that names
    /// the current version and still does not decode is an error. A current attachment moves
    /// directly into the visitor without a separate heap allocation for its envelope.
    pub(in crate::analysis::storage) fn visit_stored_observation_attachment(
        &self,
        run: &AnalysisRunResult,
        current: impl FnOnce(ObservationAttachment) -> Result<(), AnalysisPublicationError>,
    ) -> Result<ObservationAttachmentReadOutcome, AnalysisPublicationError> {
        let path = self.observation_path(&run.run_id)?;
        let parent = path.parent().ok_or("observation parent absent")?;
        if !self.owned_directory_exists(parent, "observation attachments")? {
            return Ok(ObservationAttachmentReadOutcome::Missing);
        }
        let bytes = match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                self.fs_read(path).map_err(|error| error.to_string())?
            }
            Ok(_) => return Err("observation attachment must be a regular file".into()),
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(ObservationAttachmentReadOutcome::Missing)
            }
            Err(error) => return Err(error.to_string().into()),
        };
        let attachment: ObservationAttachment = match self.decode_metadata(&bytes) {
            Ok(attachment) => attachment,
            Err(error) => {
                return match self.decode_metadata::<ObservationAttachmentVersion>(&bytes) {
                    Ok(version)
                        if version.schema_version != OBSERVATION_ATTACHMENT_SCHEMA_VERSION =>
                    {
                        Ok(ObservationAttachmentReadOutcome::Legacy)
                    }
                    _ => Err(error),
                }
            }
        };
        if attachment.schema_version != OBSERVATION_ATTACHMENT_SCHEMA_VERSION {
            return Ok(ObservationAttachmentReadOutcome::Legacy);
        }
        self.validate_bound_observation(&run.run_id, &attachment)?;
        attachment.validate(run, self.observation_run_digest(&run.run_id)?)?;
        drop(bytes);
        current(attachment)?;
        Ok(ObservationAttachmentReadOutcome::Current)
    }
    fn observation_path(&self, run_id: &str) -> Result<PathBuf, AnalysisPublicationError> {
        validate_run_id_token(run_id)?;
        self.publication_observation_path(run_id)?
            .ok_or_else(|| "observation publication is not committed".into())
    }

    pub(in crate::analysis::storage) fn install_observations(
        &self,
        run_id: &str,
        attachment: ObservationAttachment,
        objects: BTreeMap<Digest, Vec<u8>>,
    ) -> Result<(), AnalysisPublicationError> {
        self.reserve_standalone(run_id)?;
        let run = self
            .load_run_metadata(run_id)?
            .ok_or("observation run is absent")?;
        attachment.validate(&run, self.observation_run_digest(run_id)?)?;
        self.ensure_objects_directory()?;
        for (digest, bytes) in objects {
            match self
                .atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| error.to_string())?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("observation object collision".into())
                }
            }
        }
        self.ensure_owned_directory(&self.root.join("observations"), "observation attachments")?;
        let bytes = serde_json::to_vec(&attachment).map_err(|error| error.to_string())?;
        match self
            .atomic_create_new(self.observation_path(run_id)?, &bytes)
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
    ) -> Result<ObservationHistoryLookup, AnalysisPublicationError> {
        let Some(run) = self.load_run_metadata(run_id)? else {
            return Ok(ObservationHistoryLookup::RunNotFound);
        };
        let Some(attachment) = self.read_observation_attachment(&run)? else {
            return Ok(ObservationHistoryLookup::ObservationNotFound);
        };
        history::load(&attachment, observation_id, sample, |digest| {
            self.read_chunk(digest).map_err(|e| e.to_string())
        })
        .map_err(Into::into)
    }
}
