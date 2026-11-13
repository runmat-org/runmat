//! Exact filesystem owner for one importdata invocation.

use super::*;
use std::io;
use std::sync::Arc;

pub(super) struct ImportInput {
    pub(super) provider: Arc<dyn fs::FsProvider>,
    pub(super) path: PathBuf,
    resource: Option<fs::CapturedFsResourceLease>,
}

impl ImportInput {
    pub(super) fn capture(path: PathBuf) -> BuiltinResult<Self> {
        let (provider, path) = fs::capture_provider_path(&path).map_err(|err| {
            importdata_error_with_source(
                &IMPORTDATA_ERROR_IO,
                "importdata: unable to capture current folder",
                err,
            )
        })?;
        let has_resource_lease = provider.resource_lease().is_some();
        let resource = fs::capture_resource_lease(provider.clone());
        if has_resource_lease && resource.is_none() {
            return Err(importdata_error_with(
                &IMPORTDATA_ERROR_IO,
                "importdata: filesystem resource authority was invalidated",
            ));
        }
        Ok(Self {
            provider,
            path,
            resource,
        })
    }

    pub(super) fn check_resource(&self) -> BuiltinResult<()> {
        if self
            .resource
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            return Err(importdata_error_with_source(
                &IMPORTDATA_ERROR_IO,
                "importdata: filesystem resource authority was invalidated",
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "captured filesystem resource lease is no longer valid",
                ),
            ));
        }
        Ok(())
    }
}
