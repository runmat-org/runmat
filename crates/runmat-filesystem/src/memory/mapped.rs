use super::{bump_identity, normalize_path, MemoryEntry, MemoryFsProvider};
use crate::mapped_backing::check_range;
use crate::{FsMappedBacking, FsMappedWriteFailure, FsMappedWriteOutcome};
use async_trait::async_trait;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicU64, Arc};

pub(super) fn open(
    provider: MemoryFsProvider,
    path: &Path,
    writable: bool,
) -> io::Result<Box<dyn FsMappedBacking>> {
    if !crate::FsPathNamespace::VirtualRooted.is_absolute(path) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "memory mapped backing requires a resolved absolute path",
        ));
    }
    let path = normalize_path(path)?;
    let (identity, extent) = {
        let tree = provider
            .inner
            .lock()
            .map_err(|_| io::Error::other("memory filesystem lock poisoned"))?;
        match tree.entries.get(&path) {
            Some(MemoryEntry::File {
                bytes,
                readonly,
                identity,
                ..
            }) => {
                if writable && *readonly {
                    return Err(io::Error::new(
                        ErrorKind::PermissionDenied,
                        "memory mapped source is read-only",
                    ));
                }
                (identity.clone(), bytes.len() as u64)
            }
            Some(MemoryEntry::Directory) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "mapped source is not a file",
                ));
            }
            None => return Err(super::not_found(&path)),
        }
    };
    Ok(Box::new(MemoryMappedBacking {
        provider,
        path,
        identity,
        extent,
        writable,
    }))
}

struct MemoryMappedBacking {
    provider: MemoryFsProvider,
    path: PathBuf,
    identity: Arc<AtomicU64>,
    extent: u64,
    writable: bool,
}

impl MemoryMappedBacking {
    fn with_entry<T>(&self, action: impl FnOnce(&MemoryEntry) -> io::Result<T>) -> io::Result<T> {
        let tree = self
            .provider
            .inner
            .lock()
            .map_err(|_| io::Error::other("memory filesystem lock poisoned"))?;
        let entry = tree.entries.get(&self.path).ok_or_else(stale)?;
        self.validate(entry)?;
        action(entry)
    }

    fn validate(&self, entry: &MemoryEntry) -> io::Result<()> {
        match entry {
            MemoryEntry::File {
                identity, readonly, ..
            } if Arc::ptr_eq(identity, &self.identity) => {
                if self.writable && *readonly {
                    Err(io::Error::new(
                        ErrorKind::PermissionDenied,
                        "memory mapped source became read-only",
                    ))
                } else {
                    Ok(())
                }
            }
            MemoryEntry::File { .. } | MemoryEntry::Directory => Err(stale()),
        }
    }
}

fn stale() -> io::Error {
    io::Error::new(
        ErrorKind::PermissionDenied,
        "memory mapped source identity was removed or replaced",
    )
}

#[async_trait(?Send)]
impl FsMappedBacking for MemoryMappedBacking {
    fn opened_extent(&self) -> u64 {
        self.extent
    }

    fn writable(&self) -> bool {
        self.writable
    }

    fn check_current(&self) -> io::Result<()> {
        self.with_entry(|_| Ok(()))
    }

    async fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        check_range(offset, buffer.len(), self.extent)?;
        self.with_entry(|entry| {
            let MemoryEntry::File { bytes, .. } = entry else {
                unreachable!("validated file entry")
            };
            let start = usize::try_from(offset)
                .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "offset is too large"))?;
            if start >= bytes.len() {
                return Ok(0);
            }
            let count = buffer.len().min(bytes.len() - start);
            buffer[..count].copy_from_slice(&bytes[start..start + count]);
            Ok(count)
        })
    }

    async fn write_at(
        &mut self,
        offset: u64,
        bytes: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        if !self.writable {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::PermissionDenied,
                "memory mapped backing is read-only",
            )));
        }
        check_range(offset, bytes.len(), self.extent)
            .map_err(FsMappedWriteFailure::before_effect)?;
        let mut tree = self.provider.inner.lock().map_err(|_| {
            FsMappedWriteFailure::before_effect(io::Error::other("memory filesystem lock poisoned"))
        })?;
        let entry = tree
            .entries
            .get_mut(&self.path)
            .ok_or_else(|| FsMappedWriteFailure::before_effect(stale()))?;
        self.validate(entry)
            .map_err(FsMappedWriteFailure::before_effect)?;
        let MemoryEntry::File {
            bytes: target,
            identity,
            revision,
            ..
        } = entry
        else {
            unreachable!("validated file entry")
        };
        let start = usize::try_from(offset).map_err(|_| {
            FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::InvalidInput,
                "offset is too large",
            ))
        })?;
        let end = start.checked_add(bytes.len()).ok_or_else(|| {
            FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::InvalidInput,
                "range is too large",
            ))
        })?;
        if end > target.len() {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::UnexpectedEof,
                "memory mapped source was truncated after opening",
            )));
        }
        // A retained bounded reader may share the old immutable snapshot.
        // Make the new writable owner before bumping the visible revision.
        let writable = Arc::make_mut(target);
        bump_identity(identity).map_err(FsMappedWriteFailure::before_effect)?;
        writable[start..end].copy_from_slice(bytes);
        *revision = uuid::Uuid::new_v4().to_string();
        Ok(FsMappedWriteOutcome {
            accepted_bytes: bytes.len(),
            requested_bytes: bytes.len(),
        })
    }
}
