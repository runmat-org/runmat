//! Captured, opened range I/O. An opened extent is a range limit, not a
//! resource/allocator certificate or a transaction across multiple writes.

use crate::{current_provider, current_provider_with_current_dir, FsProvider};
use async_trait::async_trait;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod write;
pub use write::{FsMappedWriteDisposition, FsMappedWriteFailure, FsMappedWriteOutcome};

/// One provider-issued opened file identity. Implementations must not stage
/// writes for an implicit Drop flush or silently follow a replaced path.
#[async_trait(?Send)]
pub trait FsMappedBacking: Send {
    fn opened_extent(&self) -> u64;
    fn writable(&self) -> bool;
    fn check_current(&self) -> io::Result<()>;

    /// Read directly into the caller's buffer. A short read exposes an
    /// externally truncated source; it must not be filled with invented data.
    async fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize>;

    /// Write at most one caller-provided slice. A short write is observable;
    /// callers own retries and must not assume a multi-write transaction.
    /// An implementation may return `BeforeEffect` only when it proves no
    /// bytes committed; uncertain provider errors must be `Indeterminate`.
    async fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<usize, FsMappedWriteFailure>;
}

/// Strongly retains the selected provider, CWD-resolved path, and its opened
/// backing across suspension and ambient provider/CWD replacement.
pub struct CapturedMappedBacking {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    backing: Box<dyn FsMappedBacking>,
}

impl CapturedMappedBacking {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub fn opened_extent(&self) -> u64 {
        self.backing.opened_extent()
    }

    pub fn writable(&self) -> bool {
        self.backing.writable()
    }

    pub fn check_current(&self) -> io::Result<()> {
        self.backing.check_current()
    }

    pub async fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        check_range(offset, buffer.len(), self.opened_extent())?;
        self.check_current()?;
        let result = self.backing.read_at(offset, buffer).await;
        let count = result?;
        if count > buffer.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "mapped provider read reported more than the supplied buffer",
            ));
        }
        self.check_current()?;
        Ok(count)
    }

    pub async fn write_at(
        &mut self,
        offset: u64,
        bytes: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        check_range(offset, bytes.len(), self.opened_extent())
            .map_err(FsMappedWriteFailure::before_effect)?;
        if !self.writable() {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::PermissionDenied,
                "mapped backing is read-only",
            )));
        }
        self.check_current()
            .map_err(FsMappedWriteFailure::before_effect)?;
        let result = self.backing.write_at(offset, bytes).await;
        let count = result?;
        if count > bytes.len() {
            return Err(FsMappedWriteFailure::indeterminate(
                bytes.len(),
                io::Error::new(
                    ErrorKind::InvalidData,
                    "mapped provider write reported more than the supplied slice",
                ),
            ));
        }
        // A revoked post-write owner reports failure, not rollback. Bytes may
        // already have committed; the caller must not replay this operation.
        self.check_current().map_err(|source| {
            if count == 0 {
                FsMappedWriteFailure::before_effect(source)
            } else {
                FsMappedWriteFailure::committed(count, source)
            }
        })?;
        Ok(FsMappedWriteOutcome {
            accepted_bytes: count,
            requested_bytes: bytes.len(),
        })
    }
}

pub(crate) fn check_range(offset: u64, length: usize, extent: u64) -> io::Result<()> {
    let length = u64::try_from(length)
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "mapped range is too large"))?;
    if offset.checked_add(length).is_none_or(|end| end > extent) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "mapped range exceeds opened extent",
        ));
    }
    Ok(())
}

pub async fn capture_mapped_backing(
    path: impl AsRef<Path>,
    writable: bool,
) -> io::Result<Option<CapturedMappedBacking>> {
    let (provider, path) = if path.as_ref().is_absolute() {
        (current_provider(), path.as_ref().to_path_buf())
    } else {
        let (provider, cwd) = current_provider_with_current_dir()?;
        (provider, cwd.join(path))
    };
    let Some(backing) = provider.open_mapped_backing(&path, writable).await? else {
        return Ok(None);
    };
    backing.check_current()?;
    if writable && !backing.writable() {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "provider issued a read-only backing for a writable request",
        ));
    }
    Ok(Some(CapturedMappedBacking {
        provider,
        path,
        backing,
    }))
}

#[cfg(test)]
#[path = "mapped_backing/tests.rs"]
mod tests;
