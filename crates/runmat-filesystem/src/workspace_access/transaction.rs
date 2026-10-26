//! A bounded asynchronous provider transaction for existing synchronous algorithms.
//! The private view has no persistence authority. Only the original provider's
//! conditional publication makes any files visible; it owns conflicts and receipts.
use crate::{workspace::*, FsProvider, MemoryFsProvider};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct WorkspaceTransaction {
    provider: Arc<dyn FsProvider>,
    view: Arc<MemoryFsProvider>,
    root: PathBuf,
    revision: WorkspaceRevision,
    before: BTreeMap<PathBuf, [u8; 32]>,
    maximum: u64,
    writable: bool,
}
impl WorkspaceTransaction {
    pub async fn capture(
        provider: Arc<dyn FsProvider>,
        root: PathBuf,
        writable: bool,
        maximum: u64,
    ) -> io::Result<Self> {
        // Snapshot views use a virtual rooted namespace on every operating
        // system. Native providers already supply their own coherent scopes.
        if provider.path_namespace() != crate::FsPathNamespace::VirtualRooted {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "snapshot transactions require a virtual-rooted provider",
            ));
        }
        let mut resolved_root = PathBuf::from("/");
        for part in root.components() {
            match part {
                std::path::Component::RootDir | std::path::Component::CurDir => {}
                std::path::Component::Normal(part) => resolved_root.push(part),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "invalid workspace transaction root",
                    ))
                }
            }
        }
        let root = resolved_root;
        let maximum = maximum.min(MAX_WORKSPACE_PUBLICATION_BYTES as u64);
        if maximum == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "workspace transaction requires a byte budget",
            ));
        }
        if writable {
            provider.create_dir_all(&root).await?;
        }
        let request = WorkspaceCaptureRequest {
            root: root.clone(),
            excluded_subtrees: vec![],
            max_files: MAX_WORKSPACE_PUBLICATION_EDITS as u32,
            max_bytes: maximum,
        };
        let snapshot = match provider.capture_workspace(request).await {
            Ok(snapshot) => snapshot,
            Err(error) if !writable && error.kind() == io::ErrorKind::NotFound => {
                WorkspaceSnapshot {
                    root: root.clone(),
                    excluded_subtrees: vec![],
                    revision: WorkspaceRevision([0; 32]),
                    files: vec![],
                }
            }
            Err(error) => return Err(error),
        };
        if snapshot.root != root
            || !snapshot.excluded_subtrees.is_empty()
            || snapshot.files.len() > MAX_WORKSPACE_PUBLICATION_EDITS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "provider returned an invalid transaction inventory",
            ));
        }
        let mut total = 0u64;
        let mut before = BTreeMap::new();
        for file in &snapshot.files {
            if file.path.as_os_str().is_empty()
                || file.path.as_os_str().len() > 1024
                || file
                    .path
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid captured workspace path",
                ));
            }
            total = total
                .checked_add(file.bytes.len() as u64)
                .ok_or_else(|| io::Error::other("workspace inventory overflow"))?;
            if total > maximum {
                return Err(limited(total, maximum));
            }
            if before
                .insert(file.path.clone(), Sha256::digest(&file.bytes).into())
                .is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate captured file",
                ));
            }
        }
        let revision = snapshot.revision;
        let root = snapshot.root.clone();
        let view = Arc::new(crate::memory::scope::from_snapshot(snapshot)?);
        Ok(Self {
            provider,
            view,
            root,
            revision,
            before,
            maximum,
            writable,
        })
    }
    pub fn view(&self) -> Arc<dyn FsProvider> {
        self.view.clone()
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// No operation result may be exposed until this outcome is admitted.
    pub async fn commit(self) -> io::Result<Option<WorkspacePublicationOutcome>> {
        if !self.writable {
            return Ok(None);
        }
        let after = self
            .view
            .capture_workspace(WorkspaceCaptureRequest {
                root: self.root.clone(),
                excluded_subtrees: vec![],
                max_files: MAX_WORKSPACE_PUBLICATION_EDITS as u32,
                max_bytes: self.maximum,
            })
            .await?;
        let mut remaining = self.before;
        let mut edits = Vec::new();
        let mut bytes = 0u64;
        for file in after.files {
            let hash: [u8; 32] = Sha256::digest(&file.bytes).into();
            if remaining.remove(&file.path) == Some(hash) {
                continue;
            }
            bytes = bytes
                .checked_add(file.bytes.len() as u64)
                .ok_or_else(|| io::Error::other("workspace edit overflow"))?;
            if bytes > self.maximum {
                return Err(limited(bytes, self.maximum));
            }
            edits.push(WorkspaceEdit::Write {
                path: file.path,
                bytes: file.bytes,
            });
        }
        for path in remaining.into_keys() {
            edits.push(WorkspaceEdit::Remove { path });
        }
        if edits.is_empty() {
            return Ok(None);
        }
        if edits.len() > MAX_WORKSPACE_PUBLICATION_EDITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "workspace edit inventory exceeds its budget",
            ));
        }
        let request = WorkspacePublicationRequest {
            id: WorkspacePublicationId::new(),
            root: self.root,
            excluded_subtrees: vec![],
            expected_revision: self.revision,
            edits,
        };
        // Recover an acknowledgement through the same authority. Never replay
        // immutable writes or infer success from a subsequent directory read.
        match self.provider.publish_workspace(request.clone()).await {
            Ok(outcome) => Ok(Some(outcome)),
            Err(error) => match self.provider.lookup_workspace_publication(&request).await {
                Ok(Some(outcome)) => Ok(Some(outcome)),
                _ => Err(error),
            },
        }
    }
}
fn limited(required_bytes: u64, maximum_bytes: u64) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        super::WorkspaceResourceLimit {
            required_bytes,
            maximum_bytes,
        },
    )
}
