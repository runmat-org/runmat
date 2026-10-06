//! Captured filesystem owner for one copyfile invocation.

use crate::context::RuntimeContext;
use runmat_execution::cancellation::CancellationToken;
use runmat_filesystem::{self as vfs, FsProvider};
use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod operations;
mod selection;

pub(super) struct CopyFilesystem {
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
    cancellation: Option<CancellationToken>,
    cooperation: RefCell<super::cooperation::CopyCooperation>,
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
        if self.provider.path_namespace().is_absolute(path) {
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
}
