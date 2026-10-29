//! Captured provider-owned immutable evidence I/O, bounded before decode and visibility.
use super::*;
use std::sync::Arc;
use storage::AnalysisPublicationError as Error;

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
        self.immutable_objects(maximum, bytes.len() as u64)?
            .persist(relative, bytes)
    }

    /// One workspace scope (one journal lock and recovery) for a run of immutable objects
    /// written one at a time, none larger than `largest` bytes: each object's buffer and its
    /// existing-content comparison stay within `maximum`, as for a single object. The
    /// allowance is checked here; the workspace opens with the first object written, so a
    /// run rejected before any write leaves no directory behind.
    pub(in crate::analysis) fn immutable_objects(
        &self,
        maximum: u64,
        largest: u64,
    ) -> Result<ImmutableObjectScope<'_>, Error> {
        let available = maximum
            .checked_sub(largest)
            .filter(|available| *available > 0 && *available >= largest)
            .ok_or_else(|| {
                Error::ResourceLimit {
                    message: format!(
                        "immutable object buffers exceed their phase allowance: object {largest} bytes, allowance {maximum} bytes"
                    ),
                }
            })?;
        Ok(ImmutableObjectScope {
            access: self,
            scope: None,
            available,
        })
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
                    runmat_canonical_codec::CanonicalLimits::new(
                        usize::try_from(available)
                            .map_err(|_| limited("evidence allowance exceeds address space"))?,
                        100_000_000,
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

pub(in crate::analysis) struct ImmutableObjectScope<'a> {
    access: &'a StudyEvidenceAccess,
    scope: Option<Box<dyn runmat_filesystem::workspace_access::WorkspaceAccess>>,
    available: u64,
}

impl ImmutableObjectScope<'_> {
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
        let scope = match &mut self.scope {
            Some(scope) => scope,
            empty => empty.insert(self.access.provider.workspace_access(
                &self.access.root,
                true,
                self.available,
            )?),
        };
        scope.create_dir_all(path.parent().ok_or("immutable object path has no parent")?)?;
        match scope.atomic_create_new(path, bytes, self.available)? {
            runmat_filesystem::AtomicCreateResult::Published { .. } => {}
            runmat_filesystem::AtomicCreateResult::Existing(existing) if existing == bytes => {}
            runmat_filesystem::AtomicCreateResult::Existing(_) => {
                return Err("immutable object identity already contains different content".into())
            }
        }
        scope.synchronize_immutable(path).map_err(|error| {
            Error::EvidenceDurabilityIndeterminate {
                path: self.access.root.join(path),
                cause: error.to_string(),
            }
        })?;
        Ok(self.access.root.join(path).display().to_string())
    }
}
