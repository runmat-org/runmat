//! Captured provider-owned immutable evidence I/O, bounded before decode and visibility.
use super::*;
use std::sync::Arc;
use storage::AnalysisPublicationError as Error;

/// Resident workspace for persisting immutable objects of at most `largest` bytes: the
/// object's own buffer plus one existing-content comparison buffer of the same size. Admission
/// funds this amount and [`StudyEvidenceAccess::persist_immutable_batch`] requires it.
pub(in crate::analysis) fn immutable_batch_workspace(largest: u64) -> Option<u64> {
    largest.checked_mul(2)
}

pub(in crate::analysis) struct StudyEvidenceAccess {
    root: PathBuf,
    provider: Arc<dyn runmat_filesystem::FsProvider>,
}
impl StudyEvidenceAccess {
    pub(in crate::analysis) fn capture() -> Self {
        Self::new(
            study_evidence_root(),
            storage::scoped_artifact_provider().unwrap_or_else(runmat_filesystem::current_provider),
        )
    }
    pub(in crate::analysis) fn new(
        root: PathBuf,
        provider: Arc<dyn runmat_filesystem::FsProvider>,
    ) -> Self {
        Self { root, provider }
    }
    pub(in crate::analysis) fn persist_bytes(
        &self,
        relative: &str,
        bytes: &[u8],
        maximum: u64,
    ) -> Result<String, Error> {
        self.persist_immutable_batch(maximum, bytes.len() as u64, |batch| {
            batch.persist(relative, bytes)
        })
    }

    /// One workspace scope and one group commit for a run of immutable objects staged one at
    /// a time, none larger than `largest` bytes: each object's buffer and its existing-content
    /// comparison stay within `maximum`, as for a single object. `maximum` must cover
    /// [`immutable_batch_workspace`]`(largest)`, the same law admission charges. The objects
    /// become durable and visible together once `stage` returns; nothing is published if it
    /// fails.
    pub(in crate::analysis) fn persist_immutable_batch<T>(
        &self,
        maximum: u64,
        largest: u64,
        stage: impl FnOnce(&mut ImmutableObjectBatch<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let limited = || {
            Error::ResourceLimit {
            message: format!(
                "immutable object buffers exceed their phase allowance: object {largest} bytes, allowance {maximum} bytes"
            ),
        }
        };
        let required = immutable_batch_workspace(largest).ok_or_else(limited)?;
        // The scope quota is the comparison buffer: everything beyond the object's own buffer.
        let available = maximum
            .checked_sub(largest)
            .filter(|available| maximum >= required && *available > 0)
            .ok_or_else(limited)?;
        let scope = self
            .provider
            .workspace_access(&self.root, true, available)?;
        let mut batch = ImmutableObjectBatch {
            root: &self.root,
            batch: scope.immutable_batch()?,
            available,
        };
        let value = stage(&mut batch)?;
        batch
            .batch
            .commit()
            .map_err(|error| Error::EvidenceDurabilityIndeterminate {
                path: self.root.clone(),
                cause: error.to_string(),
            })?;
        Ok(value)
    }
    pub(super) fn persist(
        &self,
        relative: &str,
        evidence: &PersistedStudyEvidence,
        input_resident: u64,
        maximum: u64,
    ) -> Result<String, Error> {
        let encoded = runmat_canonical_codec::profile_json_bytes(evidence, maximum)?;
        let own = input_resident
            .checked_add(encoded)
            .ok_or_else(|| limited("evidence phase overflow"))?;
        let available = maximum
            .checked_sub(own)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| limited("evidence encoding exceeds its phase allowance"))?;
        let bytes = runmat_canonical_codec::encode_json_bounded(evidence, available)?;
        let scope = self
            .provider
            .workspace_access(&self.root, true, available)?;
        let path = std::path::Path::new(relative);
        scope.create_dir_all(path.parent().ok_or("evidence path has no parent")?)?;
        match scope.atomic_create_new(path, &bytes, available)? {
            runmat_filesystem::AtomicCreateResult::Published { .. } => {}
            runmat_filesystem::AtomicCreateResult::Existing(existing_bytes) => {
                let decoded = runmat_canonical_codec::json_materialization_resident_bytes_estimate(
                    &existing_bytes,
                    runmat_canonical_codec::CanonicalLimits::content_bounded(
                        usize::try_from(available)
                            .map_err(|_| limited("evidence allowance exceeds address space"))?,
                        65_536,
                        64,
                    ),
                    available,
                )?;
                let required = (existing_bytes.len() as u64)
                    .checked_add(decoded)
                    .ok_or_else(|| limited("evidence collision decode overflow"))?;
                if required > available {
                    return Err(limited(
                        "evidence collision decode exceeds its phase allowance",
                    ));
                }
                let existing: PersistedStudyEvidence = serde_json::from_slice(&existing_bytes)
                    .map_err(|error| format!("failed to parse existing study evidence: {error}"))?;
                validate_evidence(
                    &existing,
                    &evidence.binding.study_fingerprint,
                    &evidence.binding.stage,
                )?;
                if existing != *evidence {
                    return Err(format!(
                        "study evidence identity already contains different content at {}",
                        self.root.join(path).display()
                    )
                    .into());
                }
            }
        }
        scope.synchronize_immutable(path).map_err(|error| {
            Error::EvidenceDurabilityIndeterminate {
                path: self.root.join(path),
                cause: error.to_string(),
            }
        })?;
        Ok(self.root.join(path).display().to_string())
    }
}
fn limited(message: &str) -> Error {
    Error::ResourceLimit {
        message: message.into(),
    }
}

pub(in crate::analysis) struct ImmutableObjectBatch<'a> {
    root: &'a std::path::Path,
    batch: Box<dyn runmat_filesystem::workspace_access::ImmutableBatch + 'a>,
    available: u64,
}

impl ImmutableObjectBatch<'_> {
    /// Stages one immutable object; its returned path is visible once the batch commits.
    pub(in crate::analysis) fn persist(
        &mut self,
        relative: &str,
        bytes: &[u8],
    ) -> Result<String, Error> {
        if bytes.len() as u64 > self.available {
            return Err(Error::ResourceLimit {
                message: format!(
                    "immutable object of {} bytes exceeds its scope allowance of {} bytes",
                    bytes.len(),
                    self.available
                ),
            });
        }
        let path = std::path::Path::new(relative);
        if path.parent().is_none() {
            return Err("immutable object path has no parent".into());
        }
        match self.batch.stage(path, bytes, self.available)? {
            runmat_filesystem::workspace_access::ImmutableStage::Staged => {}
            runmat_filesystem::workspace_access::ImmutableStage::Existing(existing)
                if existing == bytes => {}
            runmat_filesystem::workspace_access::ImmutableStage::Existing(_) => {
                return Err("immutable object identity already contains different content".into())
            }
        }
        Ok(self.root.join(path).display().to_string())
    }
}
