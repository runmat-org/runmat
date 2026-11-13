use runmat_filesystem as vfs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use crate::BuiltinResult;

pub(super) struct DirFilesystem {
    provider: Arc<dyn vfs::FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
}

impl DirFilesystem {
    pub(super) fn capture() -> BuiltinResult<Self> {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|error| {
            super::super::error::io(
                format!("dir: unable to determine current directory ({error})"),
                error,
            )
        })?;
        let lease = vfs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            cwd,
            lease,
        })
    }

    pub(super) fn check(&self) -> std::io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            Err(std::io::Error::other(
                "filesystem resource authority changed during directory listing",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn current_directory(&self) -> &Path {
        &self.cwd
    }

    fn resolved(&self, path: &Path) -> PathBuf {
        if is_rooted_path(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    pub(super) async fn metadata(&self, path: &Path) -> std::io::Result<vfs::FsMetadata> {
        self.check()?;
        let result = self.provider.metadata(&self.resolved(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf> {
        self.check()?;
        let result = self.provider.canonicalize(&self.resolved(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn directory(
        &self,
        path: &Path,
    ) -> std::io::Result<Vec<crate::builtins::common::fs::enumeration::ListedEntry>> {
        self.check()?;
        let result = crate::builtins::common::fs::enumeration::directory_in(
            self.provider.as_ref(),
            &self.cwd,
            path,
        )
        .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    pub(super) async fn matching(
        &self,
        path: &Path,
    ) -> std::io::Result<Vec<crate::builtins::common::fs::enumeration::ListedEntry>> {
        self.check()?;
        let result = crate::builtins::common::fs::enumeration::matching_in(
            self.provider.as_ref(),
            &self.cwd,
            path,
        )
        .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }
}
