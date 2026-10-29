//! Atomic verified content-addressed storage for native execution-artifact consumers.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use runmat_execution::Digest;

use super::{CacheExport, CacheImport};
use crate::{ArtifactError, ArtifactResult, LogicalObject};

#[derive(Clone, Debug)]
pub struct FilesystemObjectStore {
    root: PathBuf,
    max_object_bytes: u64,
}

impl FilesystemObjectStore {
    pub fn open(root: impl Into<PathBuf>, max_object_bytes: u64) -> ArtifactResult<Self> {
        let root = root.into();
        if !root.is_absolute() || max_object_bytes == 0 {
            return Err(ArtifactError::Invalid(
                "filesystem object store requires an absolute root and non-zero object bound"
                    .into(),
            ));
        }
        fs::create_dir_all(&root)?;
        restrict_directory(&root)?;
        if !root.is_dir() {
            return Err(ArtifactError::Invalid(
                "filesystem object store root is not a directory".into(),
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
        self.read_required_bounded(digest, self.max_object_bytes)
    }

    fn read_required_bounded(&self, digest: Digest, maximum_bytes: u64) -> ArtifactResult<Vec<u8>> {
        self.read_required_admitted(digest, maximum_bytes, &mut |_| Ok(()))
    }

    fn read_required_admitted(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Vec<u8>> {
        let bound = maximum_bytes.min(self.max_object_bytes);
        if bound == 0 {
            return Err(ArtifactError::Limit(
                "filesystem read has no admitted payload allowance".into(),
            ));
        }
        let mut file = OpenOptions::new().read(true).open(self.path(digest))?;
        let length = file.metadata()?.len();
        if length > bound {
            return Err(ArtifactError::Limit(
                "filesystem object exceeds the admitted byte bound".into(),
            ));
        }
        let capacity = usize::try_from(length).map_err(|_| {
            ArtifactError::Limit("filesystem object length does not fit this host".into())
        })?;
        admit(length)?;
        let mut bytes = vec![0; capacity];
        file.read_exact(&mut bytes)?;
        // File growth never grows the admitted buffer. Inspect at most one stack byte.
        let mut extra = [0];
        if file.read(&mut extra)? != 0 || Digest::sha256(&bytes) != digest {
            return Err(ArtifactError::Identity(
                "filesystem object bytes do not match their content identity".into(),
            ));
        }
        Ok(bytes)
    }
}

impl CacheImport for FilesystemObjectStore {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        match self.read_required_admitted(digest, maximum_bytes, admit) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(ArtifactError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        match self.read_required_bounded(digest, maximum_bytes) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(ArtifactError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

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

impl CacheExport for FilesystemObjectStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.write_verified_batch(std::iter::once(object))
    }

    /// Stages every new object, sends each to the device, issues one device flush for the
    /// batch and only then links the content names, so a visible name always holds complete
    /// durable bytes. Staged files are closed after their write to bound open descriptors.
    fn write_verified_batch<'a>(
        &mut self,
        objects: impl IntoIterator<Item = &'a LogicalObject>,
    ) -> ArtifactResult<()> {
        let mut staged: Vec<(PathBuf, tempfile::TempPath, &LogicalObject)> = Vec::new();
        for object in objects {
            object.validate()?;
            if object.bytes.len() as u64 > self.max_object_bytes {
                return Err(ArtifactError::Limit(
                    "filesystem object exceeds its configured byte bound".into(),
                ));
            }
            let target = self.path(object.descriptor.digest);
            if target.exists() {
                verify_existing(self, object)?;
                continue;
            }
            if staged
                .iter()
                .any(|(staged_target, _, _)| *staged_target == target)
            {
                // Same digest, already validated: identical canonical bytes.
                continue;
            }
            let mut temporary = tempfile::NamedTempFile::new_in(&self.root)?;
            temporary.write_all(&object.bytes)?;
            send_to_device(temporary.as_file())?;
            staged.push((target, temporary.into_temp_path(), object));
        }
        let Some((_, last, _)) = staged.last() else {
            return Ok(());
        };
        device_barrier(&fs::File::open(last)?)?;
        for (target, temporary, object) in &staged {
            match fs::hard_link(temporary, target) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    verify_existing(self, object)?
                }
                Err(error) => return Err(ArtifactError::Io(error)),
            }
        }
        Ok(())
    }
}

/// Sends the file's data and metadata to the device. On Apple platforms `fsync(2)` does not
/// flush the device cache; the batch's single `device_barrier` does that for every file.
fn send_to_device(file: &fs::File) -> std::io::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use std::os::fd::AsRawFd as _;
        // SAFETY: the descriptor is owned by `file` and stays open for the call.
        if unsafe { libc::fsync(file.as_raw_fd()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        file.sync_all()
    }
}

/// One full device flush covering everything previously sent (`F_FULLFSYNC` on Apple).
fn device_barrier(file: &fs::File) -> std::io::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        file.sync_all()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = file;
        Ok(())
    }
}

fn verify_existing(store: &FilesystemObjectStore, object: &LogicalObject) -> ArtifactResult<()> {
    if store.read_required(object.descriptor.digest)? != object.bytes {
        return Err(ArtifactError::Identity(
            "filesystem object store identity collision".into(),
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
    use crate::{LogicalObject, ObjectNamespace};

    use super::*;

    #[test]
    fn store_is_atomic_idempotent_bounded_and_rehashes_reads() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = FilesystemObjectStore::open(directory.path().join("objects"), 16).unwrap();
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

#[cfg(test)]
mod batch_tests {
    use crate::{LogicalObject, ObjectNamespace};

    use super::*;

    fn object(text: &str) -> LogicalObject {
        LogicalObject::new(
            ObjectNamespace::ResultValue,
            "test/object",
            "application/vnd.runmat.test",
            text.as_bytes().to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn a_batch_publishes_every_object_once_and_leaves_no_staging() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("objects");
        let mut store = FilesystemObjectStore::open(&root, 64).unwrap();
        let existing = object("existing");
        store.write_verified(&existing).unwrap();
        let batch = [object("a"), object("b"), object("a"), existing.clone()];
        store.write_verified_batch(&batch).unwrap();
        for object in &batch {
            assert_eq!(
                store.read_verified(object.descriptor.digest).unwrap(),
                Some(object.bytes.clone())
            );
        }
        assert_eq!(fs::read_dir(&root).unwrap().count(), 3);

        fs::write(store.path(existing.descriptor.digest), b"poisoned").unwrap();
        assert!(store
            .write_verified_batch([&object("c"), &existing])
            .is_err());
    }
}

#[cfg(test)]
#[path = "filesystem/bounded_tests.rs"]
mod bounded_tests;
