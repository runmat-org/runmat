use super::{CapturedFilesystemAuthority, TransferFilesystem};
use crate::execution::ExecutionServiceError;
use async_trait::async_trait;
use runmat_filesystem::{self as vfs, FsMetadata};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub(in super::super) enum CopyOperationError {
    #[error(transparent)]
    Filesystem(#[from] io::Error),
    #[error(transparent)]
    Execution(#[from] ExecutionServiceError),
}

struct OwnerCheckpoint<'a> {
    authority: &'a CapturedFilesystemAuthority,
    cooperation: &'a mut super::super::cooperation::TransferCooperation,
}

#[async_trait(?Send)]
impl vfs::FsCopyCheckpoint for OwnerCheckpoint<'_> {
    type Error = CopyOperationError;

    async fn checkpoint(&mut self, completed_bytes: u64) -> Result<(), Self::Error> {
        self.authority.check()?;
        self.cooperation.raw_checkpoint(completed_bytes).await?;
        self.authority.check()?;
        Ok(())
    }
}

impl TransferFilesystem {
    pub(in super::super) async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.check()?;
        let result = self.authority.provider.metadata(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.check()?;
        let result = self
            .authority
            .provider
            .symlink_metadata(&self.path(path))
            .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.check()?;
        let result = self
            .authority
            .provider
            .rename(&self.path(from), &self.path(to))
            .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.check()?;
        let result = self
            .authority
            .provider
            .rename_replace(&self.path(from), &self.path(to))
            .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn read_dir(&self, path: &Path) -> io::Result<Vec<vfs::DirEntry>> {
        self.check()?;
        let result = self.authority.provider.read_dir(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.check()?;
        let result = self.authority.provider.canonicalize(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self
            .authority
            .provider
            .create_dir_all(&self.path(path))
            .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.authority.provider.remove_file(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self
            .authority
            .provider
            .remove_dir_all(&self.path(path))
            .await;
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
        let result = self
            .authority
            .provider
            .set_readonly(&self.path(path), readonly)
            .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(in super::super) async fn copy_file(
        &mut self,
        from: &Path,
        to: &Path,
    ) -> Result<u64, CopyOperationError> {
        self.check()?;
        let from = self.authority.path(from);
        let to = self.authority.path(to);
        let mut checkpoint = OwnerCheckpoint {
            authority: &self.authority,
            cooperation: &mut self.cooperation,
        };
        let result = vfs::copy_file_on_async(
            self.authority.provider.as_ref(),
            &from,
            &to,
            &mut checkpoint,
        )
        .await
        .map_err(|error| match error {
            vfs::FsCopyError::Io(error) => CopyOperationError::Filesystem(error),
            vfs::FsCopyError::Checkpoint(error) => error,
        });
        if result.is_ok() {
            self.check()?;
        }
        result
    }
}
