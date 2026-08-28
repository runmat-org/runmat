//! Verified filesystem object storage shared by a native driver and its child hosts.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{ArtifactError, ArtifactResult, LogicalObject};

#[derive(Clone, Debug)]
pub struct NativeObjectStore {
    root: PathBuf,
    max_object_bytes: u64,
}

impl NativeObjectStore {
    pub fn open(root: impl Into<PathBuf>, max_object_bytes: u64) -> ArtifactResult<Self> {
        let root = root.into();
        if !root.is_absolute() || max_object_bytes == 0 {
            return Err(ArtifactError::Invalid(
                "native object store requires an absolute root and non-zero object bound".into(),
            ));
        }
        fs::create_dir_all(&root)?;
        restrict_directory(&root)?;
        if !root.is_dir() {
            return Err(ArtifactError::Invalid(
                "native object store root is not a directory".into(),
            ));
        }
        Ok(Self {
            root,
            max_object_bytes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, digest: Digest) -> PathBuf {
        let mut name = String::with_capacity(64);
        for byte in digest.bytes() {
            use std::fmt::Write as _;
            write!(&mut name, "{byte:02x}").expect("writing to a string cannot fail");
        }
        self.root.join(name)
    }

    fn read_required(&self, digest: Digest) -> ArtifactResult<Vec<u8>> {
        let path = self.path(digest);
        let file = OpenOptions::new().read(true).open(path)?;
        let length = file.metadata()?.len();
        if length > self.max_object_bytes {
            return Err(ArtifactError::Limit(
                "native object exceeds its configured byte bound".into(),
            ));
        }
        let capacity = usize::try_from(length).map_err(|_| {
            ArtifactError::Limit("native object length does not fit this host".into())
        })?;
        let mut bytes = Vec::with_capacity(capacity);
        file.take(self.max_object_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != length || Digest::sha256(&bytes) != digest {
            return Err(ArtifactError::Identity(
                "native object bytes do not match their content identity".into(),
            ));
        }
        Ok(bytes)
    }
}

impl CacheImport for NativeObjectStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        match self.read_required(digest) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(ArtifactError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

impl CacheExport for NativeObjectStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        if object.bytes.len() as u64 > self.max_object_bytes {
            return Err(ArtifactError::Limit(
                "native object exceeds its configured byte bound".into(),
            ));
        }
        let target = self.path(object.descriptor.digest);
        if target.exists() {
            return verify_existing(self, object);
        }

        let mut temporary = tempfile::NamedTempFile::new_in(&self.root)?;
        temporary.write_all(&object.bytes)?;
        temporary.as_file_mut().sync_all()?;
        match fs::hard_link(temporary.path(), &target) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                verify_existing(self, object)
            }
            Err(error) => Err(ArtifactError::Io(error)),
        }
    }
}

fn verify_existing(store: &NativeObjectStore, object: &LogicalObject) -> ArtifactResult<()> {
    if store.read_required(object.descriptor.digest)? != object.bytes {
        return Err(ArtifactError::Identity(
            "native object store identity collision".into(),
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn restrict_directory(path: &Path) -> ArtifactResult<()> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn restrict_directory(_path: &Path) -> ArtifactResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_execution_artifact::{LogicalObject, ObjectNamespace};

    use super::*;

    #[test]
    fn store_is_atomic_idempotent_bounded_and_rehashes_reads() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = NativeObjectStore::open(directory.path().join("objects"), 16).unwrap();
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "test/object",
            "application/vnd.runmat.test",
            b"canonical".to_vec(),
        )
        .unwrap();
        store.write_verified(&object).unwrap();
        store.write_verified(&object).unwrap();
        assert_eq!(
            store.read_verified(object.descriptor.digest).unwrap(),
            Some(object.bytes.clone())
        );

        fs::write(store.path(object.descriptor.digest), b"poisoned").unwrap();
        assert!(matches!(
            store.read_verified(object.descriptor.digest),
            Err(ArtifactError::Identity(_))
        ));

        let oversized = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "test/oversized",
            "application/vnd.runmat.test",
            vec![0; 17],
        )
        .unwrap();
        assert!(matches!(
            store.write_verified(&oversized),
            Err(ArtifactError::Limit(_))
        ));
    }
}
