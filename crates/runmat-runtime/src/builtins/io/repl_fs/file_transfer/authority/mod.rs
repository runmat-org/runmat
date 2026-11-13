//! Captured filesystem owner for one file-transfer invocation.

use crate::context::RuntimeContext;
use runmat_execution::cancellation::CancellationToken;
use runmat_filesystem::{self as vfs, FsProvider};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod operations;
pub(super) use operations::CopyOperationError;
mod selection;

#[derive(Clone, Copy)]
pub(super) enum TransferOperation {
    Copy,
    Move,
}

impl TransferOperation {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Copy => "copyfile",
            Self::Move => "movefile",
        }
    }
}

pub(super) struct TransferFilesystem {
    authority: CapturedFilesystemAuthority,
    cooperation: super::cooperation::TransferCooperation,
}

struct CapturedFilesystemAuthority {
    operation: TransferOperation,
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
    cancellation: Option<CancellationToken>,
}

impl TransferFilesystem {
    pub(super) fn capture(
        operation: TransferOperation,
        runtime: Option<&RuntimeContext>,
    ) -> crate::BuiltinResult<Self> {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|error| {
            crate::build_runtime_error(format!(
                "{}: unable to determine current directory ({error})",
                operation.name()
            ))
            .with_builtin(operation.name())
            .with_source(error)
            .build()
        })?;
        let lease = vfs::capture_resource_lease(provider.clone());
        Ok(Self {
            authority: CapturedFilesystemAuthority {
                operation,
                provider,
                cwd,
                lease,
                cancellation: runtime.map(RuntimeContext::cancellation),
            },
            cooperation: super::cooperation::TransferCooperation::new(operation, runtime),
        })
    }

    #[cfg(test)]
    pub(super) fn capture_with_budget(
        operation: TransferOperation,
        budget: crate::execution::CpuCooperationBudget,
    ) -> crate::BuiltinResult<Self> {
        let mut owner = Self::capture(operation, None)?;
        owner.cooperation =
            super::cooperation::TransferCooperation::with_test_budget(operation, budget);
        Ok(owner)
    }

    fn path(&self, path: &Path) -> PathBuf {
        self.authority.path(path)
    }

    pub(super) fn check(&self) -> io::Result<()> {
        self.authority.check()
    }

    pub(super) async fn checkpoint(&mut self) -> crate::BuiltinResult<()> {
        self.cooperation.checkpoint().await
    }
}

impl CapturedFilesystemAuthority {
    fn path(&self, path: &Path) -> PathBuf {
        if self.provider.path_namespace().is_absolute(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    fn check(&self) -> io::Result<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                format!("{}: invocation cancelled", self.operation.name()),
            ));
        }
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            Err(io::Error::other(format!(
                "{}: filesystem resource authority changed",
                self.operation.name()
            )))
        } else {
            Ok(())
        }
    }
}
