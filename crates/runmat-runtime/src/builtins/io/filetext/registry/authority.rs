//! Session and exact filesystem authority for registered file identifiers.

use crate::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_filesystem as fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

#[derive(Clone)]
pub(crate) enum FileIssuer {
    Process,
    Session {
        identity: RuntimeContextLocalId,
        lease: Weak<()>,
    },
}

impl FileIssuer {
    pub(crate) fn from_runtime(runtime: Option<&RuntimeContext>) -> Self {
        match runtime {
            Some(runtime) => Self::Session {
                identity: runtime.local_identity(),
                lease: runtime.session_lease(),
            },
            None => Self::Process,
        }
    }

    pub(crate) fn admits(&self, runtime: Option<&RuntimeContext>) -> bool {
        match self {
            Self::Process => runtime.is_none(),
            Self::Session { identity, lease } => runtime.is_some_and(|runtime| {
                *identity == runtime.local_identity()
                    && lease.strong_count() != 0
                    && Weak::ptr_eq(lease, &runtime.session_lease())
            }),
        }
    }

    pub(crate) fn session_identity(&self) -> Option<RuntimeContextLocalId> {
        match self {
            Self::Process => None,
            Self::Session { identity, .. } => Some(*identity),
        }
    }

    pub(crate) fn live(&self) -> bool {
        match self {
            Self::Process => true,
            Self::Session { lease, .. } => lease.strong_count() != 0,
        }
    }
}

pub(crate) struct FileFilesystem {
    pub(crate) provider: Arc<dyn fs::FsProvider>,
    pub(crate) path: PathBuf,
    lease: Option<fs::CapturedFsResourceLease>,
}

impl FileFilesystem {
    pub(crate) fn capture(path: &Path) -> io::Result<Self> {
        let (provider, path) = fs::capture_provider_path(path)?;
        let has_lease = provider.resource_lease().is_some();
        let lease = fs::capture_resource_lease(provider.clone());
        if has_lease && lease.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "filesystem resource authority was invalidated",
            ));
        }
        Ok(Self {
            provider,
            path,
            lease,
        })
    }

    pub(crate) fn check(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "filesystem resource authority was invalidated",
            ));
        }
        Ok(())
    }
}
