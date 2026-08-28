use super::ArchiveError;
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArchiveEntryKind {
    File,
    Directory,
    Symlink,
    Hardlink,
    BlockDevice,
    CharacterDevice,
    Fifo,
    Socket,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveEntryHeader {
    pub path: String,
    pub kind: ArchiveEntryKind,
    pub expanded_bytes: u64,
    pub compressed_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_target: Option<String>,
    pub executable: bool,
}

/// Narrow header-only port. Hosts keep archive codecs and byte streaming outside policy.
pub trait ArchiveHeaderReader {
    fn read_headers(&mut self)
        -> LocalBoxFuture<'_, Result<Vec<ArchiveEntryHeader>, ArchiveError>>;
}
