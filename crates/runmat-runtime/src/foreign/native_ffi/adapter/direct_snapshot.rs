//! Private byte capture for the public path-based isolated install API.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use runmat_process_host::private_bytes::PrivateByteLease;

use super::{foreign_error_with_source, ForeignErrorKind, RuntimeError};

pub(super) struct DirectNativeSnapshot {
    staged: StagedPaths,
    pub(super) lease: PrivateByteLease,
    pub(super) library: PathBuf,
    pub(super) manifest: PathBuf,
}

#[derive(Default)]
struct StagedPaths {
    created: Vec<PathBuf>,
    committed: bool,
}

impl Drop for StagedPaths {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for path in &self.created {
            if let Err(error) = std::fs::remove_file(path) {
                eprintln!(
                    "remove unpublished native snapshot {}: {error}",
                    path.display()
                );
            }
        }
    }
}

impl DirectNativeSnapshot {
    pub(super) fn capture(
        current: Option<PrivateByteLease>,
        sequence: u64,
        library: &Path,
        manifest: &Path,
    ) -> Result<Self, RuntimeError> {
        let lease = current
            .map(Ok)
            .unwrap_or_else(|| PrivateByteLease::new_private("runmat-native-direct-"))
            .map_err(|error| {
                foreign_error_with_source(ForeignErrorKind::LoadFailed, io::Error::other(error))
            })?;
        let library_copy = lease.path().join(format!("library-{sequence}.bin"));
        let manifest_copy = lease.path().join(format!("manifest-{sequence}.json"));
        let mut staged = StagedPaths::default();
        copy_exclusive(library, &library_copy, &mut staged)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        copy_exclusive(manifest, &manifest_copy, &mut staged)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        Ok(Self {
            staged,
            lease,
            library: library_copy,
            manifest: manifest_copy,
        })
    }

    pub(super) fn commit(&mut self) {
        self.staged.committed = true;
    }
}

fn copy_exclusive(source: &Path, target_path: &Path, staged: &mut StagedPaths) -> io::Result<()> {
    let mut source = File::open(source)?;
    if !source.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "native interface input is not a regular file",
        ));
    }
    let mut target = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target_path)?;
    staged.created.push(target_path.to_path_buf());
    io::copy(&mut source, &mut target)?;
    target.sync_all()
}
