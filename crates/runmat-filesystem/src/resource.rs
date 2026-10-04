//! Narrow filesystem resource claims. A direct file handle certifies only
//! payload staging in the provider, never the memory used by its caller.

use crate::{FileHandle, FsMetadata, FsProvider};
use sha2::{Digest as _, Sha256};
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

/// Revision of the implementation and resource behavior being certified.
/// A changed claim must use a new revision before placement profiles can be
/// reused; no archive codec or HTTP revision is implied by this value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FsResourceRevision([u8; 32]);

impl FsResourceRevision {
    /// `authority` identifies the provider implementation/configuration owner;
    /// `revision_material` changes whenever its resource claim changes. The
    /// length-delimited, domain-separated encoding prevents ambiguous pairs.
    pub fn for_provider(authority: &str, revision_material: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"runmat.filesystem.resource-revision.v1\0");
        hasher.update((authority.len() as u64).to_le_bytes());
        hasher.update(authority.as_bytes());
        hasher.update((revision_material.len() as u64).to_le_bytes());
        hasher.update(revision_material);
        Self(hasher.finalize().into())
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Display for FsResourceRevision {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl serde::Serialize for FsResourceRevision {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// The provider reads/writes the supplied slice directly, without retaining
/// or constructing a payload-sized user-space copy. Fixed handle/OS overhead,
/// path processing, codecs, and caller-owned buffers remain outside this
/// claim and require separate bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum FsFilePayloadContract {
    CallerOwnedBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct FsResourceContract {
    pub revision: FsResourceRevision,
    pub opened_input_handle: Option<FsFilePayloadContract>,
    pub opened_output_handle: Option<FsFilePayloadContract>,
}

/// An opt-in provider may retain a configuration guard here. `is_valid` must
/// become false when its claim can no longer govern calls on that provider.
/// A mutable provider must synchronize this check with its own operations;
/// a point-in-time revision check alone is not a lease.
pub trait FsResourceLease: Send + Sync + 'static {
    fn contract(&self) -> FsResourceContract;
    fn is_valid(&self) -> bool;
}

/// Keeps the precise provider and its resource authority together across
/// suspension. Callers must perform I/O through `provider`, never the ambient
/// process-wide provider, and recheck `is_valid` before effect boundaries.
pub struct CapturedFsResourceLease {
    provider: Arc<dyn FsProvider>,
    lease: Arc<dyn FsResourceLease>,
    contract: FsResourceContract,
}

impl CapturedFsResourceLease {
    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub fn contract(&self) -> FsResourceContract {
        self.contract
    }

    pub fn is_valid(&self) -> bool {
        self.lease.is_valid() && self.lease.contract() == self.contract
    }
}

pub fn capture_resource_lease(provider: Arc<dyn FsProvider>) -> Option<CapturedFsResourceLease> {
    let lease = provider.resource_lease()?;
    let contract = lease.contract();
    if !lease.is_valid() {
        return None;
    }
    Some(CapturedFsResourceLease {
        provider,
        lease,
        contract,
    })
}

/// One retained opened source, bounded by the extent observed on that same
/// handle. Growth or path replacement cannot expand the bytes exposed to a
/// caller; truncation or in-place mutation may still change contents and can
/// cause a short read. This is a source-byte bound, not a memory-peak claim.
pub struct FsBoundedInput {
    handle: Box<dyn FileHandle>,
    opened_metadata: FsMetadata,
    position: u64,
    lease: Arc<dyn FsResourceLease>,
    contract: FsResourceContract,
}

impl FsBoundedInput {
    /// Providers may opt in only when `handle` reads directly into the caller
    /// buffer and does not retain a payload-sized hidden staging allocation.
    pub fn new(
        mut handle: Box<dyn FileHandle>,
        opened_metadata: FsMetadata,
        lease: Arc<dyn FsResourceLease>,
    ) -> io::Result<Self> {
        let contract = lease.contract();
        if !opened_metadata.is_file()
            || contract.opened_input_handle != Some(FsFilePayloadContract::CallerOwnedBuffer)
            || !lease.is_valid()
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "opened source has no valid direct-read resource authority",
            ));
        }
        handle.seek(SeekFrom::Start(0))?;
        Ok(Self {
            handle,
            opened_metadata,
            position: 0,
            lease,
            contract,
        })
    }

    pub fn opened_extent(&self) -> u64 {
        self.opened_metadata.len()
    }

    pub fn resource_revision(&self) -> FsResourceRevision {
        self.contract.revision
    }

    pub fn is_valid(&self) -> bool {
        self.lease.is_valid() && self.lease.contract() == self.contract
    }

    pub async fn metadata_async(&self) -> io::Result<FsMetadata> {
        self.check_valid()?;
        Ok(self.opened_metadata.clone())
    }

    fn check_valid(&self) -> io::Result<()> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "opened source resource authority was revoked",
            ))
        }
    }
}

impl Read for FsBoundedInput {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.check_valid()?;
        let remaining = self.opened_extent().saturating_sub(self.position);
        let allowed = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        if allowed == 0 {
            return Ok(0);
        }
        let count = self.handle.read(&mut buffer[..allowed])?;
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "source offset overflow"))?;
        Ok(count)
    }
}

impl Seek for FsBoundedInput {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.check_valid()?;
        let base = match position {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
            SeekFrom::End(delta) => i128::from(self.opened_extent()) + i128::from(delta),
        };
        if base < 0 || base > i128::from(self.opened_extent()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek exceeds opened source extent",
            ));
        }
        let offset = base as u64;
        self.handle.seek(SeekFrom::Start(offset))?;
        self.position = offset;
        Ok(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryFsProvider;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::NativeFsProvider;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_opened_extent_bounds_growth_and_keeps_small_range_reads_small() {
        use std::fs::{self, OpenOptions};
        use std::io::Write;

        let path =
            std::env::temp_dir().join(format!("runmat-bounded-input-{}", uuid::Uuid::new_v4()));
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("isolated input");
        writer.write_all(b"abcd").expect("initial bytes");
        writer.flush().expect("initial flush");
        let provider = NativeFsProvider;
        let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
            .expect("native open")
            .expect("direct opened source");
        assert_eq!(opened.opened_extent(), 4);
        assert!(opened.is_valid());
        writer.write_all(b"efgh").expect("growth after open");
        writer.flush().expect("growth flush");
        let mut bytes = Vec::new();
        opened.read_to_end(&mut bytes).expect("bounded read");
        assert_eq!(bytes, b"abcd");
        assert_eq!(opened.seek(SeekFrom::End(0)).expect("bounded end"), 4);
        assert!(opened.seek(SeekFrom::Start(5)).is_err());
        assert_eq!(
            futures::executor::block_on(opened.metadata_async())
                .expect("opened metadata")
                .len(),
            4
        );
        drop(opened);
        drop(writer);
        fs::remove_file(path).expect("remove isolated input");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sparse_large_native_source_needs_only_a_small_caller_buffer() {
        use std::fs::{self, OpenOptions};

        let path =
            std::env::temp_dir().join(format!("runmat-bounded-large-{}", uuid::Uuid::new_v4()));
        let writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("isolated sparse input");
        writer.set_len(1_000_000_000).expect("sparse extent");
        let provider = NativeFsProvider;
        let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
            .expect("native open")
            .expect("direct opened source");
        assert_eq!(opened.opened_extent(), 1_000_000_000);
        opened.seek(SeekFrom::Start(999_999_996)).expect("tail");
        let mut tail = [1u8; 8];
        assert_eq!(opened.read(&mut tail).expect("tail read"), 4);
        assert_eq!(&tail[..4], &[0; 4]);
        assert_eq!(opened.read(&mut tail).expect("bounded eof"), 0);
        drop(opened);
        drop(writer);
        fs::remove_file(path).expect("remove isolated sparse input");
    }

    #[cfg(unix)]
    #[test]
    fn path_replacement_does_not_redirect_the_opened_source() {
        use std::fs;

        let root =
            std::env::temp_dir().join(format!("runmat-bounded-rename-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).expect("isolated root");
        let path = root.join("source.wav");
        let old_path = root.join("old.wav");
        fs::write(&path, b"old!").expect("original source");
        let provider = NativeFsProvider;
        let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
            .expect("native open")
            .expect("direct opened source");
        fs::rename(&path, &old_path).expect("move original after open");
        fs::write(&path, b"new-and-larger").expect("replace pathname");
        let mut bytes = Vec::new();
        opened
            .read_to_end(&mut bytes)
            .expect("retained source read");
        assert_eq!(bytes, b"old!");
        assert_eq!(opened.opened_extent(), 4);
        drop(opened);
        fs::remove_file(path).expect("remove replacement");
        fs::remove_file(old_path).expect("remove original");
        fs::remove_dir(root).expect("remove root");
    }

    #[test]
    fn provider_revisions_are_domain_separated_and_extensible() {
        let native = FsResourceRevision::for_provider("runmat.native", b"direct-files-v1");
        assert_ne!(
            native,
            FsResourceRevision::for_provider("third-party.provider", b"direct-files-v1")
        );
        assert_ne!(
            native,
            FsResourceRevision::for_provider("runmat.native", b"direct-files-v2")
        );
        assert_ne!(
            FsResourceRevision::for_provider("ab", b"c"),
            FsResourceRevision::for_provider("a", b"bc")
        );
        assert_eq!(
            serde_json::to_value(native).expect("typed revision serializes"),
            serde_json::Value::String(native.to_string())
        );
    }

    #[test]
    fn memory_provider_does_not_inherit_native_payload_claim() {
        let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
        assert!(capture_resource_lease(provider).is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_claim_retains_exact_provider_without_claiming_codec_memory() {
        let provider: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
        let captured = capture_resource_lease(provider.clone()).expect("native direct handles");
        assert!(Arc::ptr_eq(captured.provider(), &provider));
        drop(provider); // The exact provider remains alive with the lease.
        assert!(captured.is_valid());
        assert_eq!(
            captured.contract(),
            FsResourceContract {
                revision: FsResourceRevision::for_provider(
                    "runmat.filesystem.native",
                    b"direct-file-handles-bounded-open-v2",
                ),
                opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
                opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            }
        );
    }

    struct RevocableLease(Arc<AtomicBool>);

    impl FsResourceLease for RevocableLease {
        fn contract(&self) -> FsResourceContract {
            FsResourceContract {
                revision: FsResourceRevision::for_provider(
                    "test.revocable",
                    b"direct-file-handles-v1",
                ),
                opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
                opened_output_handle: None,
            }
        }

        fn is_valid(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn captured_lease_detects_revocation_after_suspension() {
        let valid = Arc::new(AtomicBool::new(true));
        let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
        let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(valid.clone()));
        let captured = CapturedFsResourceLease {
            provider,
            contract: lease.contract(),
            lease,
        };
        assert!(captured.is_valid());
        valid.store(false, Ordering::Release);
        assert!(!captured.is_valid());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn bounded_opened_read_rechecks_owner_before_each_effect() {
        use std::fs;

        let path =
            std::env::temp_dir().join(format!("runmat-revoked-input-{}", uuid::Uuid::new_v4()));
        fs::write(&path, b"sound").expect("isolated input");
        let file = fs::File::open(&path).expect("opened input");
        let metadata = FsMetadata::from(file.metadata().expect("opened metadata"));
        let valid = Arc::new(AtomicBool::new(true));
        let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(valid.clone()));
        let mut opened =
            FsBoundedInput::new(Box::new(file), metadata, lease).expect("direct opened input");
        let mut first = [0u8; 1];
        assert_eq!(opened.read(&mut first).expect("first read"), 1);
        valid.store(false, Ordering::Release);
        assert_eq!(
            opened.read(&mut first).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            opened.seek(SeekFrom::Start(0)).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        drop(opened);
        fs::remove_file(path).expect("remove isolated input");
    }
}
