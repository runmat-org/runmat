use super::CopyFilesystem;
use crate::execution::ExecutionServiceError;
use async_trait::async_trait;
use runmat_filesystem::{self as vfs, FsMetadata};
use std::cell::RefMut;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
enum CopyCheckpointError {
    #[error(transparent)]
    Filesystem(#[from] io::Error),
    #[error(transparent)]
    Execution(#[from] ExecutionServiceError),
}

struct OwnerCheckpoint<'a> {
    owner: &'a CopyFilesystem,
    cooperation: RefMut<'a, super::super::cooperation::CopyCooperation>,
}

#[async_trait(?Send)]
impl vfs::FsCopyCheckpoint for OwnerCheckpoint<'_> {
    type Error = CopyCheckpointError;

    async fn checkpoint(&mut self, completed_bytes: u64) -> Result<(), Self::Error> {
        self.owner.check()?;
        self.cooperation.raw_checkpoint(completed_bytes).await?;
        self.owner.check()?;
        Ok(())
    }
}

impl CopyFilesystem {
    pub(in super::super) async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.check()?;
        let result = self.provider.metadata(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn read_dir(&self, path: &Path) -> io::Result<Vec<vfs::DirEntry>> {
        self.check()?;
        let result = self.provider.read_dir(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.check()?;
        let result = self.provider.canonicalize(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.create_dir_all(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.remove_file(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.remove_dir_all(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn set_readonly(
        &self,
        path: &Path,
        readonly: bool,
    ) -> io::Result<()> {
        self.check()?;
        let result = self.provider.set_readonly(&self.path(path), readonly).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn copy_file(&self, from: &Path, to: &Path) -> io::Result<u64> {
        self.check()?;
        let mut checkpoint = OwnerCheckpoint {
            owner: self,
            cooperation: self.cooperation.borrow_mut(),
        };
        let result = vfs::copy_file_on_async(
            self.provider.as_ref(),
            &self.path(from),
            &self.path(to),
            &mut checkpoint,
        )
        .await
        .map_err(|error| match error {
            vfs::FsCopyError::Io(error) => error,
            vfs::FsCopyError::Checkpoint(error) => io::Error::other(error),
        });
        if result.is_ok() {
            self.check()?;
        }
        result
    }
}
