//! Exact filesystem authority for one tabular invocation.

use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{
    self as fs, CapturedFsResourceLease, FsProvider, FsStagedOutput, OpenOptions,
};

pub(in crate::builtins) struct TabularFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

/// Read-side authority used by tabular parsers. A finite absolute-path call
/// may borrow the caller's host String until its stage owns every allocation.
pub(super) trait TabularReadPath {
    fn path(&self) -> &Path;
    fn check(&self) -> io::Result<()>;
    async fn open_bounded_input(&self) -> io::Result<Option<fs::FsBoundedInput>>;
    async fn open_read_preferred(&self) -> io::Result<TabularReadSource>;
}

pub(super) struct BorrowedTabularReadPath<'a> {
    provider: Arc<dyn FsProvider>,
    path: &'a Path,
    lease: CapturedFsResourceLease,
}

impl<'a> BorrowedTabularReadPath<'a> {
    pub(super) fn capture_absolute_on(
        provider: Arc<dyn FsProvider>,
        path: &'a Path,
    ) -> io::Result<Self> {
        if !provider.path_namespace().is_absolute(path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "borrowed tabular source must be absolute in its provider namespace",
            ));
        }
        let lease = fs::capture_resource_lease_for_effect(provider.clone())?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "borrowed bounded source has no retained resource authority",
            )
        })?;
        let captured = Self {
            provider,
            path,
            lease,
        };
        captured.check()?;
        Ok(captured)
    }

    pub(super) fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }
}

impl TabularReadPath for BorrowedTabularReadPath<'_> {
    fn path(&self) -> &Path {
        self.path
    }

    fn check(&self) -> io::Result<()> {
        check_resource_lease(Some(&self.lease))
    }

    async fn open_bounded_input(&self) -> io::Result<Option<fs::FsBoundedInput>> {
        open_bounded_input_on(&self.provider, self.path, Some(&self.lease)).await
    }

    async fn open_read_preferred(&self) -> io::Result<TabularReadSource> {
        self.open_bounded_input()
            .await?
            .map(TabularReadSource::Bounded)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "selected source has no bounded read authority",
                )
            })
    }
}

impl TabularReadPath for TabularFsPath {
    fn path(&self) -> &Path {
        TabularFsPath::path(self)
    }
    fn check(&self) -> io::Result<()> {
        TabularFsPath::check(self)
    }
    async fn open_bounded_input(&self) -> io::Result<Option<fs::FsBoundedInput>> {
        TabularFsPath::open_bounded_input(self).await
    }
    async fn open_read_preferred(&self) -> io::Result<TabularReadSource> {
        TabularFsPath::open_read_preferred(self).await
    }
}

fn check_resource_lease(lease: Option<&CapturedFsResourceLease>) -> io::Result<()> {
    if lease.is_some_and(|lease| !lease.is_valid()) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "tabular filesystem authority was invalidated",
        ));
    }
    Ok(())
}

async fn open_bounded_input_on(
    provider: &Arc<dyn FsProvider>,
    path: &Path,
    lease: Option<&CapturedFsResourceLease>,
) -> io::Result<Option<fs::FsBoundedInput>> {
    check_resource_lease(lease)?;
    let opened = provider.open_bounded_input(path).await?;
    check_resource_lease(lease)?;
    if let Some(source) = opened.as_ref() {
        let captured = lease.ok_or_else(|| {
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

pub(in crate::builtins) struct TabularFsOwner {
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
        open_bounded_input_on(&self.provider, &self.path, self.lease.as_ref()).await
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
        check_resource_lease(self.lease.as_ref())
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

    #[test]
    fn borrowed_absolute_source_keeps_exact_provider_without_cwd_lookup() {
        let original = Arc::new(fs::MemoryFsProvider::with_current_dir("/unrelated"));
        original
            .write_project_path("/selected/numbers.csv", b"1,2\n")
            .expect("original source");
        let provider: Arc<dyn FsProvider> = original;
        let path = Path::new("/selected/numbers.csv");
        let selected = BorrowedTabularReadPath::capture_absolute_on(provider, path)
            .expect("absolute path does not consult virtual CWD");
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/other"));
        replacement
            .write_project_path(path, b"9,9\n")
            .expect("replacement source");
        let _lock = fs::provider_override_lock();
        let _restore = fs::replace_provider(replacement);
        let mut source = block_on(selected.open_bounded_input())
            .expect("original owner read")
            .expect("bounded source");
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes).expect("direct read");
        assert_eq!(bytes, b"1,2\n");
        assert_eq!(selected.path(), path);
    }

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
