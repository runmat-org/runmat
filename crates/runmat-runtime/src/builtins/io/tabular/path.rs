//! Exact filesystem authority for one tabular invocation.

use std::io;
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

pub(super) struct TabularFsOwner {
    provider: Arc<dyn FsProvider>,
    cwd: Option<PathBuf>,
    lease: Option<CapturedFsResourceLease>,
}

impl TabularFsOwner {
    /// A known absolute path needs no working-directory service. For a path
    /// that will be materialized later, retain the current directory now.
    pub(super) fn capture(known_path: Option<&Path>) -> io::Result<Self> {
        let (provider, cwd) = if known_path.is_some_and(Path::is_absolute) {
            (fs::current_provider(), None)
        } else {
            let (provider, cwd) = fs::current_provider_with_current_dir()?;
            (provider, Some(cwd))
        };
        let advertised = provider.resource_lease().is_some();
        let lease = fs::capture_resource_lease(provider.clone());
        if advertised && lease.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "tabular filesystem authority was invalidated",
            ));
        }
        let owner = Self {
            provider,
            cwd,
            lease,
        };
        owner.check()?;
        Ok(owner)
    }

    pub(super) fn path(self, path: PathBuf) -> io::Result<TabularFsPath> {
        let path = if path.is_absolute() {
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

    fn selected_directory() -> &'static str {
        if cfg!(windows) {
            r"C:\tabular-selected"
        } else {
            "/tabular-selected"
        }
    }

    fn replacement_directory() -> &'static str {
        if cfg!(windows) {
            r"C:\tabular-replacement"
        } else {
            "/tabular-replacement"
        }
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
