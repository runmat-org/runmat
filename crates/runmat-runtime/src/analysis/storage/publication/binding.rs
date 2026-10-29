//! Small typed reservations prevent standalone and study writers stealing run identities.
use super::*;

const LOOKUP_BYTES: u64 = 64 * 1024 * 1024;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum RunBinding {
    Standalone,
    Study { publication: Digest },
}

impl access::FilesystemAccess<'_> {
    pub(super) fn visit_bindings_for(
        &self,
        publication: Digest,
        mut visit: impl FnMut(&Path) -> Result<(), AnalysisPublicationError>,
    ) -> Result<(), AnalysisPublicationError> {
        let root = self.root.join("publication-bindings");
        if !self.owned_directory_exists(&root, "publication bindings")? {
            return Ok(());
        }
        let entries = self
            .fs_read_dir(root)
            .map_err(AnalysisPublicationError::from_io)?;
        let _directory_reservation = self.retain_directory(&entries)?;
        for entry in entries {
            if entry.file_type() != FsFileType::File
                || entry.path().extension().and_then(|s| s.to_str()) != Some("json")
            {
                return Err("publication binding must be a regular JSON file".into());
            }
            let id = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("publication binding filename invalid")?;
            if self.read_binding(id)? == Some(RunBinding::Study { publication }) {
                visit(entry.path())?;
            }
        }
        Ok(())
    }
    pub(super) fn publication_owner(
        &self,
        run_id: &str,
    ) -> Result<Option<Digest>, AnalysisPublicationError> {
        Ok(match self.read_binding(run_id)? {
            Some(RunBinding::Study { publication }) => Some(publication),
            _ => None,
        })
    }
    fn binding_path(&self, run_id: &str) -> Result<PathBuf, AnalysisPublicationError> {
        validate_run_id_token(run_id)?;
        Ok(self
            .root
            .join("publication-bindings")
            .join(format!("{run_id}.json")))
    }
    fn read_binding(&self, run_id: &str) -> Result<Option<RunBinding>, AnalysisPublicationError> {
        if !self.owned_directory_exists(&self.root, "analysis artifact root")?
            || !self.owned_directory_exists(
                &self.root.join("publication-bindings"),
                "publication bindings",
            )?
        {
            return Ok(None);
        }
        let bytes = self
            .read_bounded_artifact(&self.binding_path(run_id)?, 4096)
            .map_err(AnalysisPublicationError::from)?;
        bytes
            .map(|bytes| serde_json::from_slice(&bytes).map_err(AnalysisPublicationError::from))
            .transpose()
    }
    fn reserve(&self, run_id: &str, binding: RunBinding) -> Result<(), AnalysisPublicationError> {
        self.ensure_runs_directory()?;
        self.ensure_owned_directory(
            &self.root.join("publication-bindings"),
            "publication bindings",
        )?;
        let bytes = serde_json::to_vec(&binding).map_err(|e| e.to_string())?;
        match self
            .atomic_create_new(self.binding_path(run_id)?, &bytes)
            .map_err(AnalysisPublicationError::from_io)?
        {
            AtomicCreateResult::Published { .. } => Ok(()),
            AtomicCreateResult::Existing(existing) => {
                let existing: RunBinding =
                    serde_json::from_slice(&existing).map_err(|e| e.to_string())?;
                if existing == binding {
                    Ok(())
                } else {
                    Err("run identity is reserved by another publication".into())
                }
            }
        }
    }
    pub(in crate::analysis::storage) fn reserve_standalone(
        &self,
        run_id: &str,
    ) -> Result<(), AnalysisPublicationError> {
        self.reserve(run_id, RunBinding::Standalone)
    }
    pub(super) fn reserve_study(
        &self,
        run_id: &str,
        publication: Digest,
    ) -> Result<(), AnalysisPublicationError> {
        if self.fs_symlink_metadata(self.run_path(run_id)?).is_ok() {
            return Err("publication cannot adopt a visible standalone run".into());
        }
        self.reserve(run_id, RunBinding::Study { publication })
    }
    /// Each committed attempt leaves an immutable marker under its definition run id, so earlier
    /// attempts stay listed and addressable after a rerun.
    pub(super) fn record_run_attempt(
        &self,
        run_id: &str,
        publication: Digest,
    ) -> Result<(), AnalysisPublicationError> {
        let Some((definition, _)) = crate::analysis::split_attempt_run_id(run_id) else {
            return Ok(());
        };
        let directory = self.root.join("run-attempts");
        self.ensure_owned_directory(&directory, "run attempts")?;
        let directory = directory.join(definition);
        self.ensure_owned_directory(&directory, "definition run attempts")?;
        let bytes =
            serde_json::to_vec(&RunBinding::Study { publication }).map_err(|e| e.to_string())?;
        match self
            .atomic_create_new(directory.join(format!("{run_id}.json")), &bytes)
            .map_err(AnalysisPublicationError::from_io)?
        {
            AtomicCreateResult::Published { .. } => Ok(()),
            AtomicCreateResult::Existing(existing) if existing == bytes => Ok(()),
            AtomicCreateResult::Existing(_) => Err("run attempt marker collision".into()),
        }
    }
    /// Committed attempts of a definition run id, oldest first. Attempts whose publication was
    /// reclaimed are omitted.
    pub(in crate::analysis::storage) fn run_attempts(
        &self,
        definition_run_id: &str,
    ) -> Result<Vec<String>, AnalysisPublicationError> {
        validate_run_id_token(definition_run_id)?;
        let directory = self.root.join("run-attempts").join(definition_run_id);
        if !self.owned_directory_exists(&self.root, "analysis artifact root")?
            || !self.owned_directory_exists(&self.root.join("run-attempts"), "run attempts")?
            || !self.owned_directory_exists(&directory, "definition run attempts")?
        {
            return Ok(Vec::new());
        }
        let entries = self
            .fs_read_dir(&directory)
            .map_err(AnalysisPublicationError::from_io)?;
        let _directory_reservation = self.retain_directory(&entries)?;
        let mut attempts = Vec::new();
        for entry in entries {
            let id = entry
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("run attempt marker filename invalid")?
                .to_owned();
            if entry.file_type() != FsFileType::File
                || crate::analysis::split_attempt_run_id(&id).map(|(base, _)| base)
                    != Some(definition_run_id)
            {
                return Err("run attempt marker is not an attempt of its definition".into());
            }
            if matches!(self.read_binding(&id)?, Some(RunBinding::Study { .. })) {
                attempts.push(id);
            }
        }
        attempts.sort();
        Ok(attempts)
    }
    /// A definition run id resolves to its latest committed attempt; any other id is itself.
    pub(in crate::analysis::storage) fn resolve_run_attempt(
        &self,
        run_id: &str,
    ) -> Result<String, AnalysisPublicationError> {
        if self.read_binding(run_id)?.is_some() {
            return Ok(run_id.to_owned());
        }
        Ok(self
            .run_attempts(run_id)?
            .pop()
            .unwrap_or_else(|| run_id.to_owned()))
    }
    pub(in crate::analysis::storage) fn publication_record_path(
        &self,
        run_id: &str,
    ) -> Result<Option<(PathBuf, Option<Digest>)>, AnalysisPublicationError> {
        match self.read_binding(run_id)? {
            None | Some(RunBinding::Standalone) => Ok(Some((self.run_path(run_id)?, None))),
            Some(RunBinding::Study { publication }) => {
                let Some(descriptor) = filesystem::descriptor(self, publication, LOOKUP_BYTES)?
                else {
                    return Ok(None);
                };
                let expected = descriptor
                    .records
                    .get(run_id)
                    .copied()
                    .ok_or("publication binding is outside its closed inventory")?;
                Ok(Some((
                    self.committed_record_path(publication, run_id)?,
                    Some(expected),
                )))
            }
        }
    }
    pub(in crate::analysis::storage) fn assessment_record_path(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<(PathBuf, Option<Digest>, u64)>, AssessmentStoreError> {
        let path = self
            .binding_path(run_id)
            .map_err(AssessmentStoreError::from)?;
        if !self
            .owned_directory_exists(&self.root, "analysis artifact root")
            .map_err(AssessmentStoreError::from)?
        {
            return Ok(None);
        }
        let mut remaining = maximum_bytes;
        let binding = if self
            .owned_directory_exists(
                &self.root.join("publication-bindings"),
                "publication bindings",
            )
            .map_err(AssessmentStoreError::from)?
        {
            match self.read_bounded_artifact(&path, remaining.min(4096))? {
                Some(bytes) => {
                    remaining -= bytes.len() as u64;
                    Some(
                        self.decode_metadata::<RunBinding>(&bytes)
                            .map_err(|e| AssessmentStoreError::from(e))?,
                    )
                }
                None => None,
            }
        } else {
            None
        };
        match binding {
            None | Some(RunBinding::Standalone) => Ok(Some((
                self.run_path(run_id)
                    .map_err(AssessmentStoreError::Invalid)?,
                None,
                remaining,
            ))),
            Some(RunBinding::Study { publication }) => {
                let Some((descriptor, bytes)) =
                    filesystem::descriptor_with_length(self, publication, remaining)?
                else {
                    return Ok(None);
                };
                remaining -= bytes;
                let expected = descriptor.records.get(run_id).copied().ok_or_else(|| {
                    AssessmentStoreError::Invalid(
                        "publication binding is outside its closed inventory".into(),
                    )
                })?;
                Ok(Some((
                    self.committed_record_path(publication, run_id)
                        .map_err(AssessmentStoreError::from)?,
                    Some(expected),
                    remaining,
                )))
            }
        }
    }
    fn committed_record_path(
        &self,
        publication: Digest,
        run_id: &str,
    ) -> Result<PathBuf, AnalysisPublicationError> {
        validate_run_id_token(run_id)?;
        let records = filesystem::directory(self, publication).join("runs");
        if !self.owned_directory_exists(&records, "publication records")? {
            return Err("committed publication records absent".into());
        }
        let path = records.join(format!("{run_id}.json"));
        let metadata = self
            .fs_symlink_metadata(&path)
            .map_err(|e| format!("committed publication record absent: {e}"))?;
        if metadata.file_type() != FsFileType::File {
            return Err("committed publication record must be a regular file".into());
        }
        Ok(path)
    }
    pub(in crate::analysis::storage) fn decode_bound_record(
        &self,
        bytes: &[u8],
        run_id: &str,
    ) -> Result<PersistedAnalysisArtifact, AnalysisPublicationError> {
        let (_, expected) = self
            .publication_record_path(run_id)?
            .ok_or("publication is not committed")?;
        let record = self.decode_record(bytes, run_id)?;
        if expected.is_some_and(|expected| record_digest(&record).ok() != Some(expected)) {
            return Err("run differs from its committed publication".into());
        }
        Ok(record)
    }
    pub(in crate::analysis::storage) fn publication_observation_path(
        &self,
        run_id: &str,
    ) -> Result<Option<PathBuf>, AnalysisPublicationError> {
        match self.read_binding(run_id)? {
            Some(RunBinding::Study { publication }) => {
                if self.publication_record_path(run_id)?.is_none() {
                    return Ok(None);
                }
                Ok(Some(
                    filesystem::directory(self, publication)
                        .join("observations")
                        .join(format!("{run_id}.json")),
                ))
            }
            _ => Ok(Some(
                self.root
                    .join("observations")
                    .join(format!("{run_id}.json")),
            )),
        }
    }
    pub(in crate::analysis::storage) fn validate_bound_observation(
        &self,
        run_id: &str,
        attachment: &observations::ObservationAttachment,
    ) -> Result<(), AnalysisPublicationError> {
        if let Some(RunBinding::Study { publication }) = self.read_binding(run_id)? {
            let descriptor = filesystem::descriptor(self, publication, LOOKUP_BYTES)?
                .ok_or("publication is not committed")?;
            if descriptor.observations.get(run_id).copied()
                != Some(canonical_digest(
                    "analysis-publication-observation",
                    attachment,
                )?)
            {
                return Err("observation differs from its committed publication".into());
            }
        }
        Ok(())
    }
    pub(in crate::analysis::storage) fn visit_publication_records(
        &self,
        mut visit: impl FnMut(PersistedAnalysisArtifact) -> Result<(), AnalysisPublicationError>,
    ) -> Result<(), AnalysisPublicationError> {
        let root = self.root.join("publication-bindings");
        if !self.owned_directory_exists(&root, "publication bindings")? {
            return Ok(());
        }
        let mut groups = BTreeSet::new();
        let mut group_reservation = self.reserve_resident(0)?;
        let entries = self
            .fs_read_dir(root)
            .map_err(AnalysisPublicationError::from_io)?;
        let _directory_reservation = self.retain_directory(&entries)?;
        for entry in entries {
            if entry.file_type() != FsFileType::File {
                return Err("publication binding must be a regular file".into());
            }
            let path = entry.path();
            let id = path
                .file_stem()
                .and_then(|id| id.to_str())
                .ok_or("publication binding filename invalid")?;
            if let Some(RunBinding::Study { publication }) = self.read_binding(id)? {
                if !groups.contains(&publication) {
                    group_reservation.grow(self.retained_estimate(&publication)?)?;
                    groups.insert(publication);
                }
            }
        }
        // Resolve each visibility marker once; a commit during discovery cannot expose
        // only the suffix of a group's run inventory.
        for publication in groups {
            let Some(descriptor) = filesystem::descriptor(self, publication, LOOKUP_BYTES)? else {
                continue;
            };
            let _descriptor_reservation =
                self.reserve_resident(self.retained_estimate(&descriptor)?)?;
            for (id, expected) in descriptor.records {
                if self.read_binding(&id)? != Some(RunBinding::Study { publication }) {
                    return Err("committed publication lost its run reservation".into());
                }
                let bytes = self
                    .fs_read(self.committed_record_path(publication, &id)?)
                    .map_err(AnalysisPublicationError::from_io)?;
                let record = self.decode_record(&bytes, &id)?;
                let _record_reservation =
                    self.reserve_resident(self.admit_json_metadata(&bytes)?)?;
                if record_digest(&record)? != expected {
                    return Err("run differs from its committed publication".into());
                }
                visit(record)?;
            }
        }
        Ok(())
    }
}
