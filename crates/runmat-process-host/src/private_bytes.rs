//! Native private-byte ownership shared by process hosts and artifact facades.
//! This module knows no executable formats, package digests, or runtime ports.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX: &str = "runmat-graph-attempt-";
const RETIREMENT_MARKER_SCHEMA_VERSION: u16 = 1;
const MAX_RETIREMENT_MARKER_BYTES: u64 = 1024;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RetirementMarker {
    schema_version: u16,
    process_id: Option<u32>,
    reason: String,
}

impl RetirementMarker {
    fn is_valid(&self) -> bool {
        self.schema_version == RETIREMENT_MARKER_SCHEMA_VERSION
            && !self.reason.is_empty()
            && self.reason.chars().count() <= 96
            && !self.reason.chars().any(char::is_control)
    }
}

fn existing_marker_matches(marker: &Path, process_id: Option<u32>) -> bool {
    let Ok(metadata) = std::fs::symlink_metadata(marker) else {
        return false;
    };
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() > MAX_RETIREMENT_MARKER_BYTES
    {
        return false;
    }
    let Ok(file) = std::fs::File::open(marker) else {
        return false;
    };
    let mut contents = Vec::new();
    if file
        .take(MAX_RETIREMENT_MARKER_BYTES + 1)
        .read_to_end(&mut contents)
        .is_err()
        || u64::try_from(contents.len()).map_or(true, |size| size > MAX_RETIREMENT_MARKER_BYTES)
    {
        return false;
    }
    serde_json::from_slice::<RetirementMarker>(&contents)
        .is_ok_and(|record| record.is_valid() && record.process_id == process_id)
}

struct PrivateArtifactStorage {
    path: PathBuf,
    ownership: PrivateArtifactOwnership,
}

enum PrivateArtifactOwnership {
    Owned(Mutex<Option<tempfile::TempDir>>),
    /// The parent deletes this directory after its whole child tree retires.
    ParentOwned,
}

/// Clone-cheap exact native byte owner. This is storage lifetime, not an
/// executable-artifact admission receipt.
#[derive(Clone)]
pub struct PrivateByteLease {
    storage: Arc<PrivateArtifactStorage>,
}

impl PrivateByteLease {
    pub fn new_attempt_scratch() -> Result<Self, String> {
        Self::new_private(SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX)
    }

    pub fn new_attempt_scratch_in(parent: &Path) -> Result<Self, String> {
        if !parent.is_absolute() {
            return Err("selected Graph store root is not absolute".into());
        }
        let metadata = std::fs::symlink_metadata(parent)
            .map_err(|error| format!("inspect selected Graph store root: {error}"))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err("selected Graph store root is not a real directory".into());
        }
        let root = tempfile::Builder::new()
            .prefix(SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX)
            .tempdir_in(parent)
            .map_err(|error| format!("create selected Graph attempt scratch: {error}"))?;
        Self::from_owned(root)
    }

    /// Own one private byte root. The caller still validates artifact bytes
    /// before writing or installing anything under it.
    pub fn new_private(prefix: &str) -> Result<Self, String> {
        let root = tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .map_err(|error| format!("create private artifact root: {error}"))?;
        Self::from_owned(root)
    }

    fn from_owned(root: tempfile::TempDir) -> Result<Self, String> {
        make_private(root.path())?;
        let path = std::fs::canonicalize(root.path())
            .map_err(|error| format!("resolve private artifact root: {error}"))?;
        Ok(Self {
            storage: Arc::new(PrivateArtifactStorage {
                path,
                ownership: PrivateArtifactOwnership::Owned(Mutex::new(Some(root))),
            }),
        })
    }

    pub fn in_parent_scratch(parent: &Path, unit_index: usize) -> Result<Self, String> {
        validate_parent_scratch(parent)?;
        let unit = parent.join(format!("unit-{unit_index}"));
        create_private_directory(&unit)?;
        let path = std::fs::canonicalize(&unit)
            .map_err(|error| format!("resolve parent-owned private artifact root: {error}"))?;
        Ok(Self {
            storage: Arc::new(PrivateArtifactStorage {
                path,
                ownership: PrivateArtifactOwnership::ParentOwned,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.storage.path
    }

    /// Persist before any unverified child can outlive the owner. Marker I/O
    /// errors still report the exact kept directory rather than deleting it.
    pub fn persist_unverified(&self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        let PrivateArtifactOwnership::Owned(owner) = &self.storage.ownership else {
            return Err("parent-owned private root cannot be persisted by a child".into());
        };
        let mut owner = owner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(root) = owner.take() {
            let _preserved = root.keep();
        }
        let reason = reason
            .chars()
            .take(96)
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>();
        let record = RetirementMarker {
            schema_version: RETIREMENT_MARKER_SCHEMA_VERSION,
            process_id,
            reason: if reason.is_empty() {
                "unverified retirement".into()
            } else {
                reason
            },
        };
        let encoded = serde_json::to_vec(&record).map_err(|error| {
            format!(
                "private bytes retained at {:?}, encode marker: {error}",
                self.path()
            )
        })?;
        if u64::try_from(encoded.len()).map_or(true, |size| size > MAX_RETIREMENT_MARKER_BYTES) {
            return Err(format!(
                "private bytes retained at {:?}, marker exceeds bound",
                self.path()
            ));
        }
        let marker = self.path().join("RETIREMENT_UNVERIFIED");
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = match options.open(&marker) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if existing_marker_matches(&marker, process_id) {
                    return Ok(());
                }
                return Err(format!(
                    "private bytes retained at {:?}, marker is incomplete or not regular",
                    self.path()
                ));
            }
            Err(error) => {
                return Err(format!(
                    "private bytes retained at {:?}, create marker: {error}",
                    self.path()
                ));
            }
        };
        file.write_all(&encoded)
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                format!(
                    "private bytes retained at {:?}, write marker: {error}",
                    self.path()
                )
            })?;
        Ok(())
    }
}

pub fn make_private(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("protect private artifact directory: {error}"))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn validate_parent_scratch(parent: &Path) -> Result<(), String> {
    if !parent.is_absolute()
        || !parent
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX))
    {
        return Err("selected Graph scratch is not a parent-created attempt directory".into());
    }
    let metadata = std::fs::symlink_metadata(parent)
        .map_err(|error| format!("inspect selected Graph scratch: {error}"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("selected Graph scratch is not a real directory".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err("selected Graph scratch grants access outside its owner".into());
        }
    }
    Ok(())
}

fn create_private_directory(path: &Path) -> Result<(), String> {
    let builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt as _;
        let mut builder = builder;
        builder.mode(0o700);
        builder
    };
    builder
        .create(path)
        .map_err(|error| format!("create selected Graph unit scratch: {error}"))?;
    make_private(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unverified_owned_bytes_persist_and_marker_is_typed() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PrivateByteLease>();
        let store = tempfile::tempdir().unwrap();
        let lease = PrivateByteLease::new_attempt_scratch_in(store.path()).unwrap();
        let clone = lease.clone();
        let path = lease.path().to_path_buf();
        assert!(path.exists());
        assert!(path.starts_with(std::fs::canonicalize(store.path()).unwrap()));
        assert_eq!(std::fs::canonicalize(&path).unwrap(), path);
        lease
            .persist_unverified(Some(7), "tree retirement unverified")
            .unwrap();
        lease
            .persist_unverified(Some(7), "tree retirement unverified")
            .unwrap();
        assert!(lease
            .persist_unverified(Some(9), "different child")
            .is_err());
        drop(lease);
        assert!(clone.path().exists());
        drop(clone);
        assert!(path.exists());
        let marker = std::fs::read(path.join("RETIREMENT_UNVERIFIED")).unwrap();
        let marker: RetirementMarker = serde_json::from_slice(&marker).unwrap();
        assert!(marker.is_valid());
        assert_eq!(marker.process_id, Some(7));
        assert_eq!(marker.reason, "tree retirement unverified");
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn marker_failure_keeps_the_exact_owned_directory() {
        let store = tempfile::tempdir().unwrap();
        let lease = PrivateByteLease::new_attempt_scratch_in(store.path()).unwrap();
        let path = lease.path().to_path_buf();
        std::fs::write(
            path.join("RETIREMENT_UNVERIFIED"),
            b"process_id=Some(8) reason=partial\n",
        )
        .unwrap();
        assert!(lease
            .persist_unverified(Some(8), "unverified")
            .unwrap_err()
            .contains("private bytes retained"));
        drop(lease);
        assert!(path.exists());
        std::fs::remove_dir_all(path).unwrap();
    }
}
