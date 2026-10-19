//! Filesystem fixtures preserving native OS coverage and isolated browser bytes.

use std::io;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
pub type TestDirectory = tempfile::TempDir;

#[cfg(not(target_arch = "wasm32"))]
pub fn tempdir() -> io::Result<TestDirectory> {
    tempfile::tempdir()
}

/// A fresh virtual filesystem retained for this browser fixture. Callers
/// serialize ambient provider mutation through their domain test guard.
#[cfg(target_arch = "wasm32")]
pub struct TestDirectory {
    path: std::path::PathBuf,
    _provider: runmat_filesystem::ProviderGuard,
}

#[cfg(target_arch = "wasm32")]
impl TestDirectory {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(target_arch = "wasm32")]
pub fn tempdir() -> io::Result<TestDirectory> {
    let path = std::path::PathBuf::from("/fixture");
    let provider =
        std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(&path));
    let guard = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(&path)?;
    Ok(TestDirectory {
        path,
        _provider: guard,
    })
}

pub fn exists(path: &Path) -> bool {
    runmat_filesystem::metadata(path).is_ok()
}

#[cfg(target_arch = "wasm32")]
pub fn unique_path(prefix: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
    let sequence = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    runmat_filesystem::current_dir()
        .expect("fixture working directory")
        .join(format!("{prefix}_{sequence}.tmp"))
}
