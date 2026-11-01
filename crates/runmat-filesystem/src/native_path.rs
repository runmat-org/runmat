//! Provider-issued direct OS-path authority for native libraries that cannot
//! consume a [`crate::FileHandle`]. This is not a memory-resource certificate.

use crate::{current_provider_with_current_dir, FsProvider, NativePathIntent};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

/// Provider-owned guard for an exported native path. While retained, the
/// provider must keep the mapping and permissions for the issued path/intent
/// stable. A mutable provider must coordinate revocation with its own mapping
/// changes; a consumer's checkpoint alone is not an atomic operation lock.
/// This does not pin an inode against unrelated OS namespace mutation.
pub trait NativePathAuthority {
    fn is_valid(&self) -> bool;

    /// Process-directory grants must declare immutable mapping or supply a
    /// wakeable revocation witness. File-only grants need not implement this.
    fn process_revocation(&self) -> NativeProcessRevocation<'_> {
        NativeProcessRevocation::Unavailable
    }
}

pub enum NativeProcessRevocation<'a> {
    Stable,
    Watch(Pin<Box<dyn Future<Output = ()> + 'a>>),
    Unavailable,
}

/// An opt-in provider issues this grant; ordinary callers must obtain a
/// captured grant through [`capture_native_path`], not manufacture one.
pub struct NativePathGrant {
    path: PathBuf,
    intent: NativePathIntent,
    authority: Box<dyn NativePathAuthority>,
}

impl NativePathGrant {
    /// Providers must issue only a path their own authority can service for
    /// `intent`. No provider-independent path translation is performed here.
    pub fn issued_by_provider(
        path: PathBuf,
        intent: NativePathIntent,
        authority: Box<dyn NativePathAuthority>,
    ) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "issued native path must be absolute",
            ));
        }
        if !authority.is_valid() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "issued native path authority is revoked",
            ));
        }
        Ok(Self {
            path,
            intent,
            authority,
        })
    }
}

/// Captured provider, working-directory resolution, and native-path issuer.
/// Retain this owner for at least the lifetime of the native library's file
/// handle. Check before effects and after suspension/publication. If revocation
/// occurs during a synchronous native call, a later check can report failure
/// but cannot undo bytes already committed by that call.
pub struct CapturedNativePath {
    provider: Arc<dyn FsProvider>,
    grant: NativePathGrant,
}

impl CapturedNativePath {
    pub fn path(&self) -> &Path {
        &self.grant.path
    }

    pub fn intent(&self) -> NativePathIntent {
        self.grant.intent
    }

    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub fn check_current(&self) -> io::Result<()> {
        if self.grant.authority.is_valid() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured native path authority was revoked",
            ))
        }
    }

    pub fn process_revocation(&self) -> NativeProcessRevocation<'_> {
        self.grant.authority.process_revocation()
    }
}

/// Capture the installed provider and effective CWD as one invocation owner.
/// `Ok(None)` is a real unavailable route, not permission to use `std::fs`.
pub fn capture_native_path(
    requested: impl AsRef<Path>,
    intent: NativePathIntent,
) -> io::Result<Option<CapturedNativePath>> {
    let (provider, cwd) = current_provider_with_current_dir()?;
    let namespace = provider.path_namespace();
    let requested = requested.as_ref();
    let absolute = if namespace.is_absolute(requested) {
        requested.to_path_buf()
    } else {
        cwd.join(requested)
    };
    if !namespace.is_absolute(&absolute) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "native path capture requires an absolute effective working directory",
        ));
    }
    let Some(grant) = provider.acquire_native_path(&absolute, intent)? else {
        return Ok(None);
    };
    if grant.intent != intent {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "provider issued native path for a different access intent",
        ));
    }
    let captured = CapturedNativePath { provider, grant };
    // An issuer may revoke after constructing its grant but before returning
    // from acquire_native_path. Never publish that already-stale authority.
    captured.check_current()?;
    Ok(Some(captured))
}

#[cfg(test)]
#[path = "native_path/tests.rs"]
mod tests;
