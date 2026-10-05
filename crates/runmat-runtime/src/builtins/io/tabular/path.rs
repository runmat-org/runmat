//! Exact filesystem authority for one tabular invocation.

use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{
    self as fs, CapturedFsResourceLease, FsProvider, FsStagedOutput, OpenOptions,
};

pub(super) struct TabularFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

pub(super) enum TabularReadSource {
    Bounded(fs::FsBoundedInput),
    Ordinary(fs::File),
}

impl TabularReadSource {
    pub(super) fn opened_extent(&self) -> Option<u64> {
        match self {
            Self::Bounded(source) => Some(source.opened_extent()),
            Self::Ordinary(_) => None,
        }
    }
}

impl Read for TabularReadSource {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Bounded(source) => source.read(buffer),
            Self::Ordinary(source) => source.read(buffer),
        }
    }
}

pub(super) struct TabularFsOwner {
    provider: Arc<dyn FsProvider>,
    cwd: Option<PathBuf>,
    lease: Option<CapturedFsResourceLease>,
}

impl TabularFsOwner {
    /// A known absolute path needs no working-directory service. For a path
    /// that will be materialized later, retain the current directory now.
    pub(super) fn capture(known_path: Option<&Path>) -> io::Result<Self> {
        let (provider, cwd) = if let Some(path) = known_path {
            fs::capture_provider_for_paths(&[path])?
        } else {
            let (provider, cwd) = fs::current_provider_with_current_dir()?;
            (provider, Some(cwd))
        };
        let lease = fs::capture_resource_lease_for_effect(provider.clone())?;
        let owner = Self {
            provider,
            cwd,
            lease,
        };
        owner.check()?;
        Ok(owner)
    }

    pub(super) fn path(self, path: PathBuf) -> io::Result<TabularFsPath> {
        let path = if self.provider.path_namespace().is_absolute(&path) {
            path
        } else {
            self.cwd
                .as_ref()
                .ok_or_else(|| io::Error::other("relative path has no captured directory"))?
                .join(path)
        };
        let captured = TabularFsPath {
            provider: self.provider,
            path,
            lease: self.lease,
        };
        captured.check()?;
        Ok(captured)
    }

    fn check(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "tabular filesystem authority was invalidated",
            ));
        }
        Ok(())
    }
}

impl TabularFsPath {
    /// Open one direct-read source on the captured provider. A finite route
    /// requires the opened handle's resource revision to match the captured
    /// pre-open lease; ordinary file opening is not a finite substitute.
    pub(super) async fn open_bounded_input(&self) -> io::Result<Option<fs::FsBoundedInput>> {
        self.check()?;
        let opened = self.provider.open_bounded_input(&self.path).await?;
        self.check()?;
        if let Some(source) = opened.as_ref() {
            let captured = self.lease.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "opened tabular source has no captured resource authority",
                )
            })?;
            if !source.is_valid() || source.resource_revision() != captured.contract().revision {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "opened tabular source changed resource authority",
                ));
            }
        }
        Ok(opened)
    }

    pub(super) fn begin_staged_replace(&self) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        self.check()?;
        let staged = self.provider.begin_staged_replace(&self.path)?;
        self.check()?;
        Ok(staged)
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn provider_identity(&self) -> usize {
        Arc::as_ptr(&self.provider) as *const () as usize
    }

    pub(super) fn check(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "tabular filesystem authority was invalidated",
            ));
        }
        Ok(())
    }

    pub(super) async fn open_read(&self) -> io::Result<fs::File> {
        self.check()?;
        let file = fs::File::open_async_on(self.provider.as_ref(), &self.path).await?;
        self.check()?;
        Ok(file)
    }

    pub(super) async fn open_read_preferred(&self) -> io::Result<TabularReadSource> {
        if let Some(source) = self.open_bounded_input().await? {
            return Ok(TabularReadSource::Bounded(source));
        }
        self.open_read().await.map(TabularReadSource::Ordinary)
    }

    pub(super) async fn open_with(&self, options: &OpenOptions) -> io::Result<fs::File> {
        self.check()?;
        let file = options
            .open_async_on(self.provider.as_ref(), &self.path)
            .await?;
        self.check()?;
        Ok(file)
    }

    pub(super) async fn canonicalize_on_owner(&self, path: &Path) -> io::Result<PathBuf> {
        self.check()?;
        let canonical = self.provider.canonicalize(path).await?;
        self.check()?;
        Ok(canonical)
    }

    pub(super) async fn metadata(&self) -> io::Result<fs::FsMetadata> {
        self.check()?;
        let result = self.provider.metadata(&self.path).await;
        self.check()?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::io::Read;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_selected_source_retains_real_opened_extent_for_same_stage_promotion() {
        use crate::context::host_effect::{HostEffectStageReservation, ManagedHostEffectLedger};

        let directory = tempfile::tempdir().expect("isolated native fixture");
        let source_path = directory.path().join("numbers.csv");
        std::fs::write(&source_path, b"1,2\n3,4\n").expect("fixture payload");
        let provider: Arc<dyn FsProvider> = Arc::new(fs::NativeFsProvider);
        let selected = TabularFsPath {
            lease: fs::capture_resource_lease(provider.clone()),
            provider,
            path: source_path,
        };
        let ledger = ManagedHostEffectLedger::new(8);
        let stage = ledger.reserve_stage(4).expect("pre-open prefix envelope");
        let mut source = block_on(selected.open_bounded_input())
            .expect("original native owner")
            .expect("direct opened source");
        assert_eq!(source.opened_extent(), 8);
        stage
            .promote_scan_envelope(source.opened_extent())
            .expect("same authority promotes after opened extent");
        let mut content = [0; 8];
        source
            .read_exact(&mut content)
            .expect("direct bounded read");
        assert_eq!(&content, b"1,2\n3,4\n");
        assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
        drop(stage);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    fn selected_directory() -> &'static str {
        "/tabular-selected"
    }

    fn replacement_directory() -> &'static str {
        "/tabular-replacement"
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_owner_keeps_provider_and_relative_directory_across_await() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(selected_directory()));
        selected
            .write_project_path(Path::new(selected_directory()).join("a.csv"), b"1,2\n")
            .expect("selected file");
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir(
            replacement_directory(),
        ));
        replacement
            .write_project_path(Path::new(replacement_directory()).join("a.csv"), b"9,8\n")
            .expect("replacement file");
        let _selected = fs::replace_provider(selected);
        fs::set_current_dir(selected_directory()).expect("selected cwd");
        let owner = TabularFsOwner::capture(None).expect("capture before suspension");
        let _replacement = fs::replace_provider(replacement);
        fs::set_current_dir(replacement_directory()).expect("replacement cwd");
        let path = owner.path(PathBuf::from("a.csv")).expect("selected path");
        assert_eq!(path.path(), Path::new(selected_directory()).join("a.csv"));
        assert_eq!(
            block_on(path.metadata()).expect("selected metadata").len(),
            4
        );
        let mut opened = block_on(path.open_read()).expect("original owner");
        let mut content = Vec::new();
        opened
            .read_to_end(&mut content)
            .expect("read selected file");
        assert_eq!(content, b"1,2\n");
    }
}
