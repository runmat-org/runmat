//! Immutable MemoryFS opened snapshot. The source shares already-owned file
//! bytes and copies only into the caller's buffer; ordinary writable handles
//! keep their existing private Vec semantics.

use super::{MemoryEntry, MemoryFsProvider};
use crate::{
    FileHandle, FsBoundedInput, FsFilePayloadContract, FsFileType, FsManagedOpenedInputCost,
    FsManagedReadBacking, FsManagedReadContract, FsMetadata, FsResourceContract, FsResourceLease,
    FsResourceLeaseHandle, FsResourceRevision,
};
use async_trait::async_trait;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path};
use std::sync::Arc;

struct MemoryDirectLease;
static MEMORY_DIRECT_LEASE: MemoryDirectLease = MemoryDirectLease;
struct MemoryProviderLease;
static MEMORY_PROVIDER_LEASE: MemoryProviderLease = MemoryProviderLease;

fn revision() -> FsResourceRevision {
    FsResourceRevision::for_provider(
        "runmat.filesystem.memory",
        b"immutable-arc-snapshot-direct-input-v1",
    )
}

pub(super) fn managed_read_contract() -> FsManagedReadContract {
    FsManagedReadContract {
        revision: revision(),
        backing: FsManagedReadBacking::RetainedManagedPayloadUnaccounted,
        capture_requested_heap_bytes: 0,
        opened_requested_heap_bytes: std::mem::size_of::<MemoryReadHandle>() as u64,
    }
}

pub(super) fn resource_lease_handle() -> FsResourceLeaseHandle {
    FsResourceLeaseHandle::Static(&MEMORY_PROVIDER_LEASE)
}

impl FsResourceLease for MemoryDirectLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: revision(),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        true // Snapshot bytes are immutable even if the tree is replaced.
    }

    fn managed_opened_input_cost(&self) -> Option<FsManagedOpenedInputCost> {
        Some(FsManagedOpenedInputCost {
            requested_heap_bytes: managed_read_contract().opened_requested_heap_bytes,
        })
    }
}

impl FsResourceLease for MemoryProviderLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: revision(),
            // Ordinary MemoryFS open clones the file Vec. Only the separate
            // bounded-input handle has the direct-read payload contract.
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn managed_opened_input_cost(&self) -> Option<FsManagedOpenedInputCost> {
        Some(FsManagedOpenedInputCost {
            requested_heap_bytes: managed_read_contract().opened_requested_heap_bytes,
        })
    }
}

pub(super) fn open(provider: &MemoryFsProvider, path: &Path) -> io::Result<FsBoundedInput> {
    if !crate::FsPathNamespace::VirtualRooted.is_absolute(path)
        || path.components().any(|part| {
            matches!(
                part,
                Component::Prefix(_) | Component::CurDir | Component::ParentDir
            )
        })
    {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "bounded memory input requires a resolved rooted path",
        ));
    }
    let (bytes, readonly) = {
        let tree = provider
            .inner
            .lock()
            .map_err(|_| io::Error::other("memory filesystem lock poisoned"))?;
        match tree.entries.get(path) {
            Some(MemoryEntry::File {
                bytes, readonly, ..
            }) => (Arc::clone(bytes), *readonly),
            Some(MemoryEntry::Directory) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "source is a directory",
                ));
            }
            None => return Err(super::not_found(path)),
        }
    };
    let metadata = FsMetadata::new(FsFileType::File, bytes.len() as u64, None, readonly);
    FsBoundedInput::new_with_lease(
        Box::new(MemoryReadHandle {
            bytes,
            position: 0,
            readonly,
        }),
        metadata,
        FsResourceLeaseHandle::Static(&MEMORY_DIRECT_LEASE),
    )
}

struct MemoryReadHandle {
    bytes: Arc<Vec<u8>>,
    position: u64,
    readonly: bool,
}

impl Read for MemoryReadHandle {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let start = usize::try_from(self.position).unwrap_or(usize::MAX);
        if start >= self.bytes.len() {
            return Ok(0);
        }
        let count = buffer.len().min(self.bytes.len() - start);
        buffer[..count].copy_from_slice(&self.bytes[start..start + count]);
        self.position += count as u64;
        Ok(count)
    }
}

impl Write for MemoryReadHandle {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "snapshot is read-only",
        ))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for MemoryReadHandle {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let target = match from {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
            SeekFrom::End(delta) => self.bytes.len() as i128 + i128::from(delta),
        };
        self.position = u64::try_from(target)
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "invalid snapshot seek"))?;
        Ok(self.position)
    }
}

#[async_trait(?Send)]
impl FileHandle for MemoryReadHandle {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        Ok(FsMetadata::new(
            FsFileType::File,
            self.bytes.len() as u64,
            None,
            self.readonly,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FsProvider;

    #[test]
    fn opened_snapshot_survives_replacement_without_payload_copy() {
        let provider = MemoryFsProvider::new();
        provider.write_project_path("/source.csv", b"old!").unwrap();
        let mut opened =
            futures::executor::block_on(provider.open_bounded_input(Path::new("/source.csv")))
                .unwrap()
                .unwrap();
        assert_eq!(opened.opened_extent(), 4);
        assert_eq!(
            opened.resource_revision(),
            resource_lease_handle().lease().contract().revision
        );
        provider
            .write_project_path("/source.csv", b"new-and-larger")
            .unwrap();
        let mut bytes = Vec::new();
        opened.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"old!");
        assert!(opened.is_valid());
        assert_eq!(opened.seek(SeekFrom::End(0)).unwrap(), 4);
        assert_eq!(managed_read_contract().capture_requested_heap_bytes, 0);
        assert_eq!(
            managed_read_contract().backing,
            FsManagedReadBacking::RetainedManagedPayloadUnaccounted
        );
        assert_eq!(
            managed_read_contract().opened_requested_heap_bytes,
            std::mem::size_of::<MemoryReadHandle>() as u64
        );
    }

    #[test]
    fn bounded_open_requires_resolved_rooted_path_before_tree_access() {
        let provider = MemoryFsProvider::new();
        for path in ["relative.csv", "/a/../source.csv"] {
            assert_eq!(
                futures::executor::block_on(provider.open_bounded_input(Path::new(path)))
                    .err()
                    .unwrap()
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn mapped_mutation_does_not_change_retained_snapshot() {
        let provider = MemoryFsProvider::new();
        provider
            .write_project_path("/source.csv", b"first")
            .unwrap();
        let mut opened =
            futures::executor::block_on(provider.open_bounded_input(Path::new("/source.csv")))
                .unwrap()
                .unwrap();
        let mut mapped = futures::executor::block_on(
            provider.open_mapped_backing(Path::new("/source.csv"), true),
        )
        .unwrap()
        .unwrap();
        futures::executor::block_on(mapped.write_at(0, b"other")).unwrap();
        let mut original = [0u8; 5];
        opened.read_exact(&mut original).unwrap();
        assert_eq!(&original, b"first");
        assert_eq!(provider.read_project_path("/source.csv").unwrap(), b"other");
    }

    #[test]
    fn provider_reset_does_not_retarget_retained_opened_source() {
        let provider = MemoryFsProvider::new();
        provider
            .write_project_path("/source.csv", b"original")
            .unwrap();
        let mut opened =
            futures::executor::block_on(provider.open_bounded_input(Path::new("/source.csv")))
                .unwrap()
                .unwrap();
        provider.reset();
        provider
            .write_project_path("/source.csv", b"replacement")
            .unwrap();
        let mut bytes = [0u8; 8];
        opened.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"original");
        assert_eq!(opened.opened_extent(), 8);
    }
}
