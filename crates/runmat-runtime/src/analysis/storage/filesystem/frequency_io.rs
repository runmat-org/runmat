//! Scoped filesystem transport for existing exact frequency publication owners.
use super::*;

impl access::FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn persist_structural_frequency_record(
        &self,
        prepared: PersistedStructuralFrequencyArtifact,
        objects: BTreeMap<Digest, Vec<u8>>,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        self.reserve_standalone(&prepared.step.run_id)?;
        let path = self.run_path(&prepared.step.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        publication::filesystem::reclamation::before_standalone(
            self,
            &prepared.step.run_id,
            &prepared.step.identity,
            &prepared.created_at,
            objects.keys().copied(),
        )?;
        for (digest, bytes) in objects {
            match self
                .atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| {
                    AnalysisPublicationError::from_io(error)
                        .context("failed to publish immutable analysis object")
                })? {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis object digest collision".into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared).map_err(|error| {
            AnalysisPublicationError::from(error)
                .context("failed to encode structural frequency artifact")
        })?;
        match self.atomic_create_new(&path, &bytes).map_err(|error| {
            AnalysisPublicationError::from_io(error)
                .context("failed to publish immutable run artifact")
        })? {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::StructuralFrequency(existing) =
                    self.decode_record(&existing_bytes, &prepared.step.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !frequency::same_content(&existing, &prepared) {
                    return Err("analysis run identity already contains different content".into());
                }
                return Ok(frequency::artifact_record(&existing));
            }
        }
        prune_filesystem_runs(self, &prepared.step.run_id)?;
        Ok(frequency::artifact_record(&prepared))
    }
    pub(in crate::analysis::storage) fn load_structural_frequency_record(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, AnalysisPublicationError> {
        Ok(self
            .load_structural_frequency_artifact(run_id)?
            .map(|artifact| artifact.step))
    }
    pub(in crate::analysis::storage) fn load_structural_frequency_artifact(
        &self,
        run_id: &str,
    ) -> Result<Option<PersistedStructuralFrequencyArtifact>, AnalysisPublicationError> {
        let Some((path, _)) = self.publication_record_path(run_id)? else {
            return Ok(None);
        };
        if !self.owned_directory_exists(&self.root, "analysis artifact root")?
            || !self.owned_directory_exists(
                &self.root.join("runs"),
                "analysis artifact runs directory",
            )?
        {
            return Ok(None);
        }
        let bytes = match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                self.fs_read(&path).map_err(|error| {
                    format!("failed to read structural frequency artifact: {error}")
                })?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                )
                .into());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}").into()),
        };
        let PersistedAnalysisArtifact::StructuralFrequency(artifact) =
            self.decode_bound_record(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(*artifact))
    }
    pub(in crate::analysis::storage) fn persist_acoustic_frequency_record(
        &self,
        prepared: PersistedAcousticFrequencyArtifact,
        objects: BTreeMap<Digest, Vec<u8>>,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        self.reserve_standalone(&prepared.step.run_id)?;
        let path = self.run_path(&prepared.step.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        publication::filesystem::reclamation::before_standalone(
            self,
            &prepared.step.run_id,
            &prepared.step.identity,
            &prepared.created_at,
            objects.keys().copied(),
        )?;
        for (digest, bytes) in objects {
            match self
                .atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| {
                    AnalysisPublicationError::from_io(error)
                        .context("failed to publish immutable analysis object")
                })? {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis object digest collision".into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared).map_err(|error| {
            AnalysisPublicationError::from(error)
                .context("failed to encode acoustic frequency artifact")
        })?;
        match self.atomic_create_new(&path, &bytes).map_err(|error| {
            AnalysisPublicationError::from_io(error)
                .context("failed to publish immutable run artifact")
        })? {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::AcousticFrequency(existing) =
                    self.decode_record(&existing_bytes, &prepared.step.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !frequency_sample::same_content(&existing, &prepared) {
                    return Err("analysis run identity already contains different content".into());
                }
                return Ok(frequency_sample::artifact_record(&existing));
            }
        }
        prune_filesystem_runs(self, &prepared.step.run_id)?;
        Ok(frequency_sample::artifact_record(&prepared))
    }
    pub(in crate::analysis::storage) fn load_acoustic_frequency_record(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, AnalysisPublicationError> {
        let Some((path, _)) = self.publication_record_path(run_id)? else {
            return Ok(None);
        };
        if !self.owned_directory_exists(&self.root, "analysis artifact root")?
            || !self.owned_directory_exists(
                &self.root.join("runs"),
                "analysis artifact runs directory",
            )?
        {
            return Ok(None);
        }
        let bytes = match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                self.fs_read(&path).map_err(|error| {
                    AnalysisPublicationError::from_io(error)
                        .context("failed to read acoustic frequency artifact")
                })?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                )
                .into());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}").into()),
        };
        let PersistedAnalysisArtifact::AcousticFrequency(artifact) =
            self.decode_bound_record(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(artifact.step))
    }
}
