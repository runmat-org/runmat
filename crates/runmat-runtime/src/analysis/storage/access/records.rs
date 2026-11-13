//! Scoped artifact record reads and bounded authenticated metadata caching.
use super::*;

impl FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn decode_record(
        &self,
        bytes: &[u8],
        run_id: &str,
    ) -> Result<PersistedAnalysisArtifact, AnalysisPublicationError> {
        self.admit_json_metadata(bytes)?;
        decode_persisted_artifact(bytes, run_id).map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn admit_json_metadata(
        &self,
        bytes: &[u8],
    ) -> Result<u64, AnalysisPublicationError> {
        runmat_canonical_codec::json_materialization_resident_bytes_estimate(
            bytes,
            runmat_canonical_codec::CanonicalLimits::content_bounded(
                usize::try_from(self.remaining_bytes()).map_err(|_| {
                    AnalysisPublicationError::ResourceLimit {
                        message: "artifact budget exceeds address space".into(),
                    }
                })?,
                65_536,
                64,
            ),
            self.remaining_bytes(),
        )
        .map_err(AnalysisPublicationError::from)
    }

    pub(in crate::analysis::storage) fn decode_metadata<T: serde::de::DeserializeOwned>(
        &self,
        bytes: &[u8],
    ) -> Result<T, AnalysisPublicationError> {
        self.admit_json_metadata(bytes)?;
        serde_json::from_slice(bytes).map_err(AnalysisPublicationError::from)
    }

    pub(in crate::analysis::storage) fn ensure_runs_directory(
        &self,
    ) -> Result<(), AnalysisPublicationError> {
        self.ensure_owned_directory(&self.root, "analysis artifact root")?;
        let runs = self.root.join("runs");
        self.ensure_owned_directory(&runs, "analysis artifact runs directory")?;
        Ok(())
    }

    pub(in crate::analysis::storage) fn ensure_objects_directory(
        &self,
    ) -> Result<(), AnalysisPublicationError> {
        self.ensure_owned_directory(
            &self.objects_directory(),
            "analysis artifact objects directory",
        )
    }

    pub(in crate::analysis::storage) fn read_chunk(
        &self,
        digest: Digest,
    ) -> Result<Option<Vec<u8>>, AnalysisPublicationError> {
        let path = self.object_path(digest);
        match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {}
            Ok(metadata) => {
                return Err(format!(
                    "analysis field chunk must be a regular file, found {:?}",
                    metadata.file_type()
                )
                .into());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!("failed to inspect analysis field chunk: {error}").into())
            }
        }
        let bytes = self.fs_read(path).map_err(|error| {
            AnalysisPublicationError::from_io(error).context("failed to read field chunk")
        })?;
        if Digest::sha256(&bytes) != digest {
            return Err("analysis field chunk differs from its content address"
                .to_string()
                .into());
        }
        Ok(Some(bytes))
    }

    pub(in crate::analysis::storage) fn read_standard(
        &self,
        run_id: &str,
    ) -> Result<Option<Arc<PersistedRunArtifact>>, AnalysisPublicationError> {
        validate_run_id_token(run_id)?;
        let Some((path, expected)) = self.publication_record_path(run_id)? else {
            return Ok(None);
        };
        if let Some(expected) = expected {
            let cache = self
                .manifests
                .read()
                .map_err(|_| "analysis manifest cache lock poisoned")?;
            if cache.identities.get(run_id) == Some(&expected) {
                if let Some(run) = cache.entries.get(run_id) {
                    return Ok(Some(run.clone()));
                }
            }
        }
        let bytes = match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                self.fs_read(&path).map_err(|error| {
                    AnalysisPublicationError::from_io(error).context("failed to read run artifact")
                })?
            }
            Ok(_) => return Err("analysis run artifact must be a regular file".into()),
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}").into()),
        };
        match self.decode_bound_record(&bytes, run_id)? {
            PersistedAnalysisArtifact::Standard(run) => {
                let run: Arc<PersistedRunArtifact> = Arc::from(run);
                let mut cache = self
                    .manifests
                    .write()
                    .map_err(|_| "analysis manifest cache lock poisoned")?;
                cache.entries.remove(run_id);
                cache.order.retain(|id| id != run_id);
                if cache.entries.len() == 8 {
                    if let Some(oldest) = cache.order.pop_front() {
                        cache.entries.remove(&oldest);
                        cache.identities.remove(&oldest);
                    }
                }
                cache.order.push_back(run_id.into());
                cache.entries.insert(run_id.into(), run.clone());
                if let Some(digest) = expected {
                    cache.identities.insert(run_id.into(), digest);
                } else {
                    cache.identities.remove(run_id);
                }
                Ok(Some(run))
            }
            PersistedAnalysisArtifact::StructuralFrequency(_)
            | PersistedAnalysisArtifact::AcousticFrequency(_) => Ok(None),
        }
    }

    pub(in crate::analysis::storage) fn list_persisted_runs(
        &self,
    ) -> Result<Vec<PersistedAnalysisArtifact>, AnalysisPublicationError> {
        let mut records = Vec::new();
        self.visit_persisted_runs(|record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    pub(in crate::analysis::storage) fn visit_persisted_runs(
        &self,
        mut visit: impl FnMut(PersistedAnalysisArtifact) -> Result<(), AnalysisPublicationError>,
    ) -> Result<(), AnalysisPublicationError> {
        discovery::visit_supported_artifacts_admitted(
            self,
            |bytes| self.reserve_resident(self.admit_json_metadata(bytes)?),
            decode_persisted_artifact,
            &mut visit,
        )?;
        self.visit_publication_records(visit)
    }
}
