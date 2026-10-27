//! Coherent project capture and conditional multi-file publication. Providers
//! own arbitration and durability; callers must never emulate either operation
//! with a sequence of ordinary reads or writes.
pub mod execution;
pub mod wire;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRevision(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkspacePublicationId(pub [u8; 16]);

impl WorkspacePublicationId {
    pub fn new() -> Self {
        Self(*uuid::Uuid::new_v4().as_bytes())
    }
}
impl Default for WorkspacePublicationId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceFile {
    /// Path relative to the authorized capture root.
    pub path: PathBuf,
    pub version: String,
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceCaptureRequest {
    /// A host-admitted provider path, never a model-selected physical mount.
    pub root: PathBuf,
    pub excluded_subtrees: Vec<PathBuf>,
    pub max_files: u32,
    #[serde(with = "runmat_types::wire_integer")]
    pub max_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceSnapshot {
    pub root: PathBuf,
    pub excluded_subtrees: Vec<PathBuf>,
    pub revision: WorkspaceRevision,
    pub files: Vec<WorkspaceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkspaceEdit {
    Write {
        path: PathBuf,
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    Remove {
        path: PathBuf,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspacePublicationRequest {
    pub id: WorkspacePublicationId,
    pub root: PathBuf,
    pub excluded_subtrees: Vec<PathBuf>,
    /// Guards the complete captured inventory, including deleted/new imports,
    /// content, read-only flags and dependencies not directly edited.
    pub expected_revision: WorkspaceRevision,
    pub edits: Vec<WorkspaceEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspacePublicationReceipt {
    pub id: WorkspacePublicationId,
    pub revision: WorkspaceRevision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkspacePublicationOutcome {
    Published {
        receipt: WorkspacePublicationReceipt,
    },
    /// The immutable journal and receipt are durable. Participating access must
    /// recover materialization before returning files; this is not a failed commit.
    CommittedPendingRecovery {
        receipt: WorkspacePublicationReceipt,
    },
    /// A commit marker is visible, but directory durability is unestablished.
    /// Retain this original receipt; participating access must confirm durability
    /// and recover before exposing files. This is neither rejection nor success.
    DurabilityIndeterminate {
        receipt: WorkspacePublicationReceipt,
    },
    Conflict {
        current: WorkspaceRevision,
    },
    /// Reusing an idempotency key for different contents is an error.
    IdReused,
}

pub const MAX_WORKSPACE_PUBLICATION_EDITS: usize = 10_000;
pub const MAX_WORKSPACE_PUBLICATION_BYTES: usize = 96 * 1024 * 1024;

pub fn publication_digest(request: &WorkspacePublicationRequest) -> std::io::Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    struct HashWriter(Sha256);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, request)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    Ok(writer.0.finalize().into())
}
