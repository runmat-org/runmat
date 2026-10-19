//! Invocation-owned filesystem authority for `gzip` and `gunzip`.

use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as vfs, FsMetadata, FsProvider};

use super::codec;
use crate::builtins::common::fs::enumeration;
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::BuiltinResult;

pub(super) struct ArchiveFilesystem {
    operation: codec::Operation,
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
    cooperation: Option<CpuCooperationBudget>,
}

impl ArchiveFilesystem {
    pub(super) fn capture(
        operation: codec::Operation,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(ExecutionServiceError::Cancelled.into_runtime_error());
        }
        let name = operation.builtin_name();
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|source| {
            super::archive_error_with_source(
                name,
                format!("{name}: unable to determine current directory ({source})"),
                source,
            )
        })?;
        let lease =
            vfs::capture_resource_lease_for_effect(Arc::clone(&provider)).map_err(|source| {
                super::archive_error_with_source(
                    name,
                    format!("{name}: filesystem resource authority is unavailable ({source})"),
                    source,
                )
            })?;
        let owner = Self {
            operation,
            provider,
            cwd,
            lease,
            cooperation: runtime.map(RuntimeContext::cpu_cooperation),
        };
        owner.check()?;
        Ok(owner)
    }

    fn resolved(&self, path: &Path) -> PathBuf {
        if self.provider.path_namespace().is_absolute(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        self.check_resource().map_err(|source| {
            let name = self.operation.builtin_name();
            super::archive_error_with_source(
                name,
                format!("{name}: filesystem resource authority was revoked"),
                source,
            )
        })
    }

    fn check_resource(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "filesystem resource authority was revoked",
            ));
        }
        Ok(())
    }

    async fn await_io<T>(
        &self,
        work: impl Future<Output = io::Result<T>>,
    ) -> BuiltinResult<io::Result<T>> {
        self.check()?;
        let result = match self.cooperation.as_ref() {
            Some(budget) => budget
                .await_cancellable(work)
                .await
                .map_err(ExecutionServiceError::into_runtime_error)?,
            None => work.await,
        };
        if result.is_ok() {
            self.check()?;
        }
        Ok(result)
    }

    fn io_error(&self, source: io::Error) -> crate::RuntimeError {
        let name = self.operation.builtin_name();
        super::archive_error_with_source(name, format!("{name}: {source}"), source)
    }

    pub(super) async fn metadata(&self, path: &Path) -> BuiltinResult<io::Result<FsMetadata>> {
        let path = self.resolved(path);
        self.await_io(self.provider.metadata(&path)).await
    }

    pub(super) async fn directory(
        &self,
        path: &Path,
    ) -> BuiltinResult<io::Result<Vec<enumeration::ListedEntry>>> {
        self.await_io(enumeration::directory_in_guarded(
            self.provider.as_ref(),
            &self.cwd,
            path,
            self,
        ))
        .await
    }

    pub(super) async fn matching(
        &self,
        path: &Path,
    ) -> BuiltinResult<io::Result<Vec<enumeration::ListedEntry>>> {
        self.await_io(enumeration::matching_in_guarded(
            self.provider.as_ref(),
            &self.cwd,
            path,
            self,
        ))
        .await
    }

    pub(super) async fn read(&self, path: &Path) -> BuiltinResult<Vec<u8>> {
        let path = self.resolved(path);
        self.await_io(self.provider.read(&path))
            .await?
            .map_err(|source| self.io_error(source))
    }

    pub(super) async fn create_dir_all(&self, path: &Path) -> BuiltinResult<()> {
        let path = self.resolved(path);
        self.await_io(self.provider.create_dir_all(&path))
            .await?
            .map_err(|source| self.io_error(source))
    }

    pub(super) async fn write(&self, path: &Path, bytes: &[u8]) -> BuiltinResult<()> {
        let path = self.resolved(path);
        self.await_io(self.provider.write(&path, bytes))
            .await?
            .map_err(|source| self.io_error(source))
    }

    pub(super) async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()> {
        self.check()?;
        codec::checkpoint(&mut self.cooperation.as_mut(), completed_work).await?;
        self.check()
    }

    pub(super) async fn transform(&mut self, bytes: &[u8]) -> BuiltinResult<Vec<u8>> {
        self.check()?;
        let transformed =
            codec::transform(bytes, self.operation, self.cooperation.as_mut()).await?;
        self.check()?;
        Ok(transformed)
    }
}

impl enumeration::EnumerationAuthority for ArchiveFilesystem {
    fn check(&self) -> io::Result<()> {
        self.check_resource()
    }
}

#[cfg(test)]
#[path = "filesystem/tests.rs"]
mod tests;
