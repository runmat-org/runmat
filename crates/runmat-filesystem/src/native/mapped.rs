use crate::mapped_backing::check_range;
use crate::{FsMappedBacking, FsMappedWriteFailure, FsMappedWriteOutcome};
use async_trait::async_trait;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(super) fn open(path: &Path, writable: bool) -> io::Result<Box<dyn FsMappedBacking>> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "native mapped backing requires a resolved absolute path",
        ));
    }
    let file = OpenOptions::new().read(true).write(writable).open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "mapped source is not a regular file",
        ));
    }
    Ok(Box::new(NativeMappedBacking {
        file,
        extent: metadata.len(),
        writable,
    }))
}

struct NativeMappedBacking {
    file: File,
    extent: u64,
    writable: bool,
}

#[async_trait(?Send)]
impl FsMappedBacking for NativeMappedBacking {
    fn opened_extent(&self) -> u64 {
        self.extent
    }

    fn writable(&self) -> bool {
        self.writable
    }

    fn check_current(&self) -> io::Result<()> {
        // The opened descriptor, not its former path, remains the authority
        // after rename/replacement. External in-place edits are visible.
        self.file.metadata().map(|_| ())
    }

    async fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        check_range(offset, buffer.len(), self.extent)?;
        let actual = self.file.seek(SeekFrom::Start(offset))?;
        if actual != offset {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "native mapped read seek returned a different offset",
            ));
        }
        self.file.read(buffer)
    }

    async fn write_at(
        &mut self,
        offset: u64,
        bytes: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        check_range(offset, bytes.len(), self.extent)
            .map_err(FsMappedWriteFailure::before_effect)?;
        if !self.writable {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::PermissionDenied,
                "native mapped backing is read-only",
            )));
        }
        // A pre-write truncation is an error, not permission to extend a map.
        // Concurrent external truncation after this check is not an atomic
        // transaction; the caller must treat such interference as unsupported.
        if self
            .file
            .metadata()
            .map_err(FsMappedWriteFailure::before_effect)?
            .len()
            < self.extent
        {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::UnexpectedEof,
                "native mapped source was truncated after opening",
            )));
        }
        let actual = self
            .file
            .seek(SeekFrom::Start(offset))
            .map_err(FsMappedWriteFailure::before_effect)?;
        if actual != offset {
            return Err(FsMappedWriteFailure::before_effect(io::Error::new(
                ErrorKind::InvalidData,
                "native mapped write seek returned a different offset",
            )));
        }
        let accepted_bytes = self
            .file
            .write(bytes)
            .map_err(|source| FsMappedWriteFailure::indeterminate(bytes.len(), source))?;
        Ok(FsMappedWriteOutcome {
            accepted_bytes,
            requested_bytes: bytes.len(),
        })
    }
}
