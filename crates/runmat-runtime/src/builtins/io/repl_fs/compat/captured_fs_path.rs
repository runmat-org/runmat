use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use runmat_filesystem as vfs;

use crate::{build_runtime_error, BuiltinResult};

/// One invocation's filesystem namespace, path, and revocable resource authority.
pub(super) struct CapturedFsPath {
    provider: Arc<dyn vfs::FsProvider>,
    pub(super) path: PathBuf,
    resource: Option<vfs::CapturedFsResourceLease>,
}

impl CapturedFsPath {
    pub(super) fn capture(path: PathBuf, builtin: &'static str) -> BuiltinResult<Self> {
        let (provider, path) = vfs::capture_provider_path(&path).map_err(|source| {
            build_runtime_error(format!(
                "{builtin}: cannot resolve current folder: {source}"
            ))
            .with_builtin(builtin)
            .with_source(source)
            .build()
        })?;
        let resource = vfs::capture_resource_lease(Arc::clone(&provider));
        Ok(Self {
            provider,
            path,
            resource,
        })
    }

    pub(super) async fn read(&self) -> io::Result<Vec<u8>> {
        self.check_current()?;
        let bytes = self.provider.read(&self.path).await?;
        self.check_current()?;
        Ok(bytes)
    }

    pub(super) async fn open_mapped(
        &self,
        writable: bool,
    ) -> io::Result<Option<vfs::CapturedMappedBacking>> {
        self.check_current()?;
        let backing =
            vfs::capture_mapped_backing_on(Arc::clone(&self.provider), self.path.clone(), writable)
                .await?;
        self.check_current()?;
        Ok(backing)
    }

    pub(super) async fn metadata(&self) -> io::Result<vfs::FsMetadata> {
        self.check_current()?;
        let metadata = self.provider.metadata(&self.path).await?;
        self.check_current()?;
        Ok(metadata)
    }

    pub(super) async fn set_readonly(&self, readonly: bool) -> io::Result<()> {
        self.check_current()?;
        self.provider.set_readonly(&self.path, readonly).await?;
        self.check_current()
    }

    pub(super) fn check_current(&self) -> io::Result<()> {
        if self
            .resource
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured filesystem authority was revoked",
            ));
        }
        Ok(())
    }
}
