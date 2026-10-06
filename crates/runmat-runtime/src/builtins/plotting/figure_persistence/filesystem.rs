//! Captured filesystem authority for one figure persistence operation.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider, FsStagedOutput};

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

    pub(super) async fn read(&self) -> io::Result<Vec<u8>> {
        self.check()?;
        let bytes = self.provider.read(&self.path).await?;
        self.check()?;
        Ok(bytes)
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(test)]
    pub(super) fn provider_identity(&self) -> usize {
        Arc::as_ptr(&self.provider) as *const () as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_filesystem::MemoryFsProvider;

    #[test]
    fn staged_write_keeps_captured_provider_and_working_directory() {
        let _serial = fs::provider_override_lock();
        let first = Arc::new(MemoryFsProvider::with_current_dir("/first"));
        let second = Arc::new(MemoryFsProvider::with_current_dir("/second"));
        let _selected = fs::replace_provider(first.clone() as Arc<dyn FsProvider>);
        let path = FigureFsPath::capture(Path::new("scene.fig")).expect("capture first path");
        assert_eq!(path.path(), Path::new("/first/scene.fig"));
        assert_eq!(
            path.provider_identity(),
            Arc::as_ptr(&(first.clone() as Arc<dyn FsProvider>)) as *const () as usize
        );
        fs::set_provider(second.clone() as Arc<dyn FsProvider>);

        block_on(crate::builtins::plotting::figure_persistence::write_bytes(
            None, &path, b"scene", "savefig",
        ))
        .expect("publish on captured provider");
        assert_eq!(
            block_on(first.read(Path::new("/first/scene.fig"))).unwrap(),
            b"scene"
        );
        assert!(block_on(second.read(Path::new("/second/scene.fig"))).is_err());
    }
}
