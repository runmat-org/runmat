use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use rand::RngCore as _;
use sha2::{Digest as _, Sha256};

use super::descriptor::hex_nonce;
use super::{SharedMemoryDescriptor, SharedMemoryKind};
use crate::{ProcessHostError, ProcessHostResult};

enum StoreRoot {
    Owned(tempfile::TempDir),
    Borrowed(PathBuf),
}

/// Private file-backed snapshots shared by a driver and its local child host.
///
/// Descriptors contain only a relative, nonce-derived name. The driver keeps
/// the owning store alive for the complete child session, so abandoned files
/// are removed even when the child exits unexpectedly.
pub struct SharedSnapshotStore {
    root: StoreRoot,
}

impl std::fmt::Debug for SharedSnapshotStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedSnapshotStore")
            .field("root", &self.root_path())
            .finish_non_exhaustive()
    }
}

impl SharedSnapshotStore {
    pub fn create() -> ProcessHostResult<Self> {
        let root = tempfile::Builder::new()
            .prefix("runmat-extension-")
            .tempdir()?;
        restrict_store_root(root.path())?;
        Ok(Self {
            root: StoreRoot::Owned(root),
        })
    }

    pub fn open_existing(path: impl Into<PathBuf>) -> ProcessHostResult<Self> {
        let path = path.into();
        validate_store_root(&path)?;
        Ok(Self {
            root: StoreRoot::Borrowed(path),
        })
    }

    pub fn root_path(&self) -> &Path {
        match &self.root {
            StoreRoot::Owned(root) => root.path(),
            StoreRoot::Borrowed(root) => root,
        }
    }

    pub fn publish(&self, bytes: &[u8]) -> ProcessHostResult<SharedMemoryDescriptor> {
        if bytes.is_empty() {
            return Err(ProcessHostError::Configuration(
                "shared snapshot must not be empty".into(),
            ));
        }
        let mut nonce = [0_u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let name = hex_nonce(nonce);
        let path = self.root_path().join(&name);
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(SharedMemoryDescriptor {
            kind: SharedMemoryKind::FileBacked,
            name,
            byte_length: bytes.len() as u64,
            nonce,
            sha256: Sha256::digest(bytes).into(),
        })
    }

    pub fn consume(
        &self,
        descriptor: &SharedMemoryDescriptor,
        max_bytes: u64,
    ) -> ProcessHostResult<Vec<u8>> {
        descriptor.validate()?;
        if descriptor.kind != SharedMemoryKind::FileBacked {
            return Err(ProcessHostError::Protocol(
                "shared snapshot kind is not available through the file-backed store".into(),
            ));
        }
        if descriptor.byte_length > max_bytes {
            return Err(ProcessHostError::Protocol(format!(
                "shared snapshot exceeds the {max_bytes}-byte limit"
            )));
        }
        let path = self.root_path().join(&descriptor.name);
        let outcome = consume_file(&path, descriptor);
        let _ = fs::remove_file(path);
        outcome
    }
}

fn consume_file(path: &Path, descriptor: &SharedMemoryDescriptor) -> ProcessHostResult<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() != descriptor.byte_length {
        return Err(ProcessHostError::Protocol(
            "shared snapshot metadata does not match its descriptor".into(),
        ));
    }
    let capacity = usize::try_from(descriptor.byte_length).map_err(|_| {
        ProcessHostError::Protocol("shared snapshot does not fit in host memory".into())
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    OpenOptions::new()
        .read(true)
        .open(path)?
        .take(descriptor.byte_length.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != descriptor.byte_length
        || <[u8; 32]>::from(Sha256::digest(&bytes)) != descriptor.sha256
    {
        return Err(ProcessHostError::Protocol(
            "shared snapshot content does not match its descriptor".into(),
        ));
    }
    Ok(bytes)
}

fn validate_store_root(path: &Path) -> ProcessHostResult<()> {
    if !path.is_absolute() {
        return Err(ProcessHostError::Configuration(
            "shared snapshot root must be absolute".into(),
        ));
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() {
        return Err(ProcessHostError::Configuration(
            "shared snapshot root must be a directory, not a link".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(ProcessHostError::Configuration(
                "shared snapshot root must not grant group or other access".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn restrict_store_root(path: &Path) -> ProcessHostResult<()> {
    use std::os::unix::fs::PermissionsExt as _;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    validate_store_root(path)
}

#[cfg(not(unix))]
fn restrict_store_root(path: &Path) -> ProcessHostResult<()> {
    validate_store_root(path)
}
