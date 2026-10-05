//! Captured filesystem owner for one copyfile invocation.

use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use async_trait::async_trait;
use runmat_execution::cancellation::CancellationToken;
use runmat_filesystem::{self as vfs, FsMetadata, FsProvider};
use std::cell::{RefCell, RefMut};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(super) struct CopyFilesystem {
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
    cancellation: Option<CancellationToken>,
    cooperation: RefCell<super::cooperation::CopyCooperation>,
}

#[derive(Debug, thiserror::Error)]
enum CopyCheckpointError {
    #[error(transparent)]
    Filesystem(#[from] io::Error),
    #[error(transparent)]
    Execution(#[from] ExecutionServiceError),
}

struct OwnerCheckpoint<'a> {
    owner: &'a CopyFilesystem,
    cooperation: RefMut<'a, super::cooperation::CopyCooperation>,
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
    pub(super) fn capture(runtime: Option<&RuntimeContext>) -> crate::BuiltinResult<Self> {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|error| {
            crate::build_runtime_error(format!(
                "copyfile: unable to determine current directory ({error})"
            ))
            .with_builtin("copyfile")
            .with_source(error)
            .build()
        })?;
        let lease = vfs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            cwd,
            lease,
            cancellation: runtime.map(RuntimeContext::cancellation),
            cooperation: RefCell::new(super::cooperation::CopyCooperation::new(runtime)),
        })
    }

    fn path(&self, path: &Path) -> PathBuf {
        if vfs::is_absolute_provider_path(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    pub(super) fn check(&self) -> io::Result<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "copyfile: invocation cancelled",
            ));
        }
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            Err(io::Error::other(
                "copyfile: filesystem resource authority changed",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.check()?;
        let result = self.provider.metadata(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn read_dir(&self, path: &Path) -> io::Result<Vec<vfs::DirEntry>> {
        self.check()?;
        let result = self.provider.read_dir(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.check()?;
        let result = self.provider.canonicalize(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.create_dir_all(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.remove_file(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.check()?;
        let result = self.provider.remove_dir_all(&self.path(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.check()?;
        let result = self.provider.set_readonly(&self.path(path), readonly).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn copy_file(&self, from: &Path, to: &Path) -> io::Result<u64> {
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

    pub(super) async fn same_path(&self, left: &Path, right: &Path) -> bool {
        if self.path(left) == self.path(right) {
            return true;
        }
        match (
            self.canonicalize(left).await,
            self.canonicalize(right).await,
        ) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    }

    pub(super) async fn is_descendant(&self, parent: &Path, candidate: &Path) -> bool {
        let parent = self.path(parent);
        let candidate = self.path(candidate);
        if candidate.starts_with(&parent) && candidate != parent {
            return true;
        }
        match (
            self.canonicalize(&parent).await,
            self.canonicalize(&candidate).await,
        ) {
            (Ok(parent), Ok(candidate)) => candidate.starts_with(&parent) && candidate != parent,
            _ => false,
        }
    }

    pub(super) async fn wildcard_matches(&self, pattern: &str) -> io::Result<Vec<PathBuf>> {
        let path = Path::new(pattern);
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "pattern has no file name")
        })?;
        let matcher = glob::Pattern::new(&name.to_string_lossy())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.msg))?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut matches = self
            .read_dir(parent)
            .await?
            .into_iter()
            .filter(|entry| matcher.matches(&entry.file_name().to_string_lossy()))
            .map(|entry| entry.path().to_path_buf())
            .collect::<Vec<_>>();
        matches.sort();
        Ok(matches)
    }

    pub(super) async fn remove_existing(&self, path: &Path, directory: bool) -> io::Result<()> {
        let result = if directory {
            self.remove_dir_all(path).await
        } else {
            self.remove_file(path).await
        };
        match result {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }
}
