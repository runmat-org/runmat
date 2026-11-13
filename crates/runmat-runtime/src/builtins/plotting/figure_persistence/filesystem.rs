//! Captured filesystem authority for one figure persistence operation.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider, FsStagedOutput};

use super::errors::{persistence_error, persistence_error_with_source};
use super::{ERROR_IMPORT, ERROR_IO, MAX_FIGURE_ARCHIVE_BYTES};
use crate::BuiltinResult;

pub(super) struct FigureFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

impl FigureFsPath {
    /// Resolve the path and retain its provider/resource authority before the
    /// first render, scene export, read, or other suspension.
    pub(super) fn capture(path: &Path) -> io::Result<Self> {
        let (provider, path) = fs::capture_provider_path(path)?;
        let lease = fs::capture_resource_lease_for_effect(provider.clone())?;
        let selected = Self {
            provider,
            path,
            lease,
        };
        selected.check()?;
        Ok(selected)
    }

    pub(super) fn check(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "figure persistence filesystem authority was revoked",
            ));
        }
        Ok(())
    }

    pub(super) fn begin_staged_replace(&self) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        self.check()?;
        let stage = self.provider.begin_staged_replace(&self.path)?;
        self.check()?;
        Ok(stage)
    }

    pub(super) async fn read(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
        builtin: &'static str,
    ) -> BuiltinResult<Vec<u8>> {
        self.check().map_err(|source| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!("{builtin}: figure archive authority is unavailable"),
                source,
            )
        })?;
        let mut source = crate::execution::staged_file::await_cancellable(
            runtime,
            builtin,
            self.provider.open_bounded_input(&self.path),
        )
        .await?
        .map_err(|source| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!("{builtin}: unable to open bounded figure archive"),
                source,
            )
        })?
        .ok_or_else(|| {
            persistence_error(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: selected filesystem provider cannot certify bounded archive input"
                ),
            )
        })?;
        if source.opened_extent() > MAX_FIGURE_ARCHIVE_BYTES as u64 {
            return Err(persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!(
                    "{builtin}: archive exceeds the {} byte input bound",
                    MAX_FIGURE_ARCHIVE_BYTES
                ),
            ));
        }
        if let Some(lease) = self.lease.as_ref() {
            if source.resource_revision() != lease.contract().revision {
                return Err(persistence_error(
                    &ERROR_IO,
                    builtin,
                    format!("{builtin}: opened archive authority changed after admission"),
                ));
            }
        }
        let capacity = usize::try_from(source.opened_extent()).map_err(|_| {
            persistence_error(&ERROR_IMPORT, builtin, "archive extent does not fit memory")
        })?;
        let mut bytes = Vec::with_capacity(capacity);
        let mut chunk = [0u8; 8 * 1024];
        let mut cooperation =
            crate::execution::staged_file::StagedCpuCooperation::new(runtime, builtin);
        loop {
            crate::execution::staged_file::StagedWriteCooperation::completed(
                &mut cooperation,
                chunk.len(),
            )
            .await?;
            self.check().map_err(|source| {
                persistence_error_with_source(
                    &ERROR_IO,
                    builtin,
                    format!("{builtin}: archive authority was revoked during read"),
                    source,
                )
            })?;
            let read = source.read(&mut chunk).map_err(|source| {
                persistence_error_with_source(
                    &ERROR_IO,
                    builtin,
                    format!("{builtin}: unable to read bounded figure archive"),
                    source,
                )
            })?;
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
        }
        self.check().map_err(|source| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!("{builtin}: archive authority was revoked after read"),
                source,
            )
        })?;
        Ok(bytes)
    }

    pub(super) async fn canonical_path(&self) -> io::Result<PathBuf> {
        self.check()?;
        let path = self.provider.canonicalize(&self.path).await?;
        self.check()?;
        Ok(path)
    }

    #[cfg(test)]
    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn provider_identity(&self) -> usize {
        Arc::as_ptr(&self.provider) as *const () as usize
    }

    pub(super) fn provider_lease(&self) -> Weak<dyn FsProvider> {
        Arc::downgrade(&self.provider)
    }
}
