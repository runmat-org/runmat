use once_cell::sync::OnceCell;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub mod remote;
#[cfg(not(target_arch = "wasm32"))]
pub mod sandbox;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeFsProvider;
#[cfg(not(target_arch = "wasm32"))]
pub use remote::{RemoteFsConfig, RemoteFsProvider};
#[cfg(not(target_arch = "wasm32"))]
pub use sandbox::SandboxFsProvider;
#[cfg(target_arch = "wasm32")]
pub use wasm::PlaceholderFsProvider;

pub trait FileHandle: Read + Write + Seek + Send + Sync {}

impl<T> FileHandle for T where T: Read + Write + Seek + Send + Sync + 'static {}

#[derive(Clone, Debug, Default)]
pub struct OpenFlags {
    pub read: bool,
    pub write: bool,
    pub append: bool,
    pub truncate: bool,
    pub create: bool,
    pub create_new: bool,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    flags: OpenFlags,
}

impl OpenOptions {
    pub fn new() -> Self {
        Self {
            flags: OpenFlags::default(),
        }
    }

    pub fn read(&mut self, value: bool) -> &mut Self {
        self.flags.read = value;
        self
    }

    pub fn write(&mut self, value: bool) -> &mut Self {
        self.flags.write = value;
        self
    }

    pub fn append(&mut self, value: bool) -> &mut Self {
        self.flags.append = value;
        self
    }

    pub fn truncate(&mut self, value: bool) -> &mut Self {
        self.flags.truncate = value;
        self
    }

    pub fn create(&mut self, value: bool) -> &mut Self {
        self.flags.create = value;
        self
    }

    pub fn create_new(&mut self, value: bool) -> &mut Self {
        self.flags.create_new = value;
        self
    }

    pub fn open(&self, path: impl AsRef<Path>) -> io::Result<File> {
        with_provider(|provider| provider.open(path.as_ref(), &self.flags)).map(File::from_handle)
    }

    pub fn flags(&self) -> &OpenFlags {
        &self.flags
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsFileType {
    Directory,
    File,
    Symlink,
    Other,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct FsMetadata {
    file_type: FsFileType,
    len: u64,
    modified: Option<SystemTime>,
    readonly: bool,
}

impl FsMetadata {
    pub fn new(
        file_type: FsFileType,
        len: u64,
        modified: Option<SystemTime>,
        readonly: bool,
    ) -> Self {
        Self {
            file_type,
            len,
            modified,
            readonly,
        }
    }

    pub fn file_type(&self) -> FsFileType {
        self.file_type
    }

    pub fn is_dir(&self) -> bool {
        matches!(self.file_type, FsFileType::Directory)
    }

    pub fn is_file(&self) -> bool {
        matches!(self.file_type, FsFileType::File)
    }

    pub fn is_symlink(&self) -> bool {
        matches!(self.file_type, FsFileType::Symlink)
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn modified(&self) -> Option<SystemTime> {
        self.modified
    }

    pub fn is_readonly(&self) -> bool {
        self.readonly
    }
}

#[derive(Clone, Debug)]
pub struct DirEntry {
    path: PathBuf,
    file_name: OsString,
    file_type: FsFileType,
}

impl DirEntry {
    pub fn new(path: PathBuf, file_name: OsString, file_type: FsFileType) -> Self {
        Self {
            path,
            file_name,
            file_type,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_name(&self) -> &OsString {
        &self.file_name
    }

    pub fn file_type(&self) -> FsFileType {
        self.file_type
    }

    pub fn is_dir(&self) -> bool {
        matches!(self.file_type, FsFileType::Directory)
    }
}

pub trait FsProvider: Send + Sync + 'static {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>>;
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn metadata(&self, path: &Path) -> io::Result<FsMetadata>;
    fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata>;
    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>>;
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    fn create_dir(&self, path: &Path) -> io::Result<()>;
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    fn remove_dir(&self, path: &Path) -> io::Result<()>;
    fn remove_dir_all(&self, path: &Path) -> io::Result<()>;
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()>;
}

pub struct File {
    inner: Box<dyn FileHandle>,
}

impl File {
    fn from_handle(handle: Box<dyn FileHandle>) -> Self {
        Self { inner: handle }
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        opts.open(path)
    }

    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        opts.open(path)
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").finish_non_exhaustive()
    }
}

impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for File {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for File {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

static PROVIDER: OnceCell<RwLock<Arc<dyn FsProvider>>> = OnceCell::new();

fn provider_lock() -> &'static RwLock<Arc<dyn FsProvider>> {
    PROVIDER.get_or_init(|| RwLock::new(default_provider()))
}

fn with_provider<T>(f: impl FnOnce(&dyn FsProvider) -> T) -> T {
    let guard = provider_lock()
        .read()
        .expect("filesystem provider lock poisoned");
    f(&**guard)
}

pub fn set_provider(provider: Arc<dyn FsProvider>) {
    let mut guard = provider_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    *guard = provider;
}

/// Temporarily replace the active provider and return a guard that restores the
/// previous provider when dropped. Useful for tests that need to install a mock
/// filesystem without permanently mutating global state.
pub fn replace_provider(provider: Arc<dyn FsProvider>) -> ProviderGuard {
    let mut guard = provider_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    let previous = guard.clone();
    *guard = provider;
    ProviderGuard { previous }
}

/// Run a closure with the supplied provider installed, restoring the previous
/// provider automatically afterwards.
pub fn with_provider_override<R>(provider: Arc<dyn FsProvider>, f: impl FnOnce() -> R) -> R {
    let guard = replace_provider(provider);
    let result = f();
    drop(guard);
    result
}

/// Returns the currently installed provider.
pub fn current_provider() -> Arc<dyn FsProvider> {
    provider_lock()
        .read()
        .expect("filesystem provider lock poisoned")
        .clone()
}

pub struct ProviderGuard {
    previous: Arc<dyn FsProvider>,
}

impl Drop for ProviderGuard {
    fn drop(&mut self) {
        set_provider(self.previous.clone());
    }
}

pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    with_provider(|provider| provider.read(path.as_ref()))
}

pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
    let bytes = read(path)?;
    String::from_utf8(bytes).map_err(|err| io::Error::new(ErrorKind::InvalidData, err.utf8_error()))
}

pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
    with_provider(|provider| provider.write(path.as_ref(), data.as_ref()))
}

pub fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.remove_file(path.as_ref()))
}

pub fn metadata(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    with_provider(|provider| provider.metadata(path.as_ref()))
}

pub fn symlink_metadata(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    with_provider(|provider| provider.symlink_metadata(path.as_ref()))
}

pub fn read_dir(path: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    with_provider(|provider| provider.read_dir(path.as_ref()))
}

pub fn canonicalize(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    with_provider(|provider| provider.canonicalize(path.as_ref()))
}

pub fn create_dir(path: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.create_dir(path.as_ref()))
}

pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.create_dir_all(path.as_ref()))
}

pub fn remove_dir(path: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.remove_dir(path.as_ref()))
}

pub fn remove_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.remove_dir_all(path.as_ref()))
}

pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    with_provider(|provider| provider.rename(from.as_ref(), to.as_ref()))
}

/// Update the readonly flag for a file or directory if the provider supports it.
pub fn set_readonly(path: impl AsRef<Path>, readonly: bool) -> io::Result<()> {
    with_provider(|provider| provider.set_readonly(path.as_ref(), readonly))
}

/// Copy a file from `from` to `to`, truncating the destination when it exists.
/// Returns the number of bytes written, matching `std::fs::copy`.
pub fn copy_file(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<u64> {
    let mut reader = OpenOptions::new().read(true).open(from.as_ref())?;
    let mut writer = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(to.as_ref())?;
    io::copy(&mut reader, &mut writer)
}

fn default_provider() -> Arc<dyn FsProvider> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Arc::new(NativeFsProvider::default())
    }
    #[cfg(target_arch = "wasm32")]
    {
        Arc::new(PlaceholderFsProvider::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::io::{Read, Write};
    use std::sync::Mutex;
    use tempfile::tempdir;

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct UnsupportedProvider;

    impl FsProvider for UnsupportedProvider {
        fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            Err(unsupported())
        }

        fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Err(unsupported())
        }

        fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
            Err(unsupported())
        }

        fn remove_file(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
            Err(unsupported())
        }

        fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
            Err(unsupported())
        }

        fn create_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn remove_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
            Err(unsupported())
        }
    }

    fn unsupported() -> io::Error {
        io::Error::new(ErrorKind::Unsupported, "unsupported in test provider")
    }

    #[test]
    fn copy_file_round_trip() {
        let _guard = TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("src.bin");
        let dst = dir.path().join("dst.bin");
        {
            let mut file = std::fs::File::create(&src).expect("create src");
            file.write_all(b"hello filesystem").expect("write src");
        }

        copy_file(&src, &dst).expect("copy");
        let mut dst_file = File::open(&dst).expect("open dst");
        let mut contents = Vec::new();
        dst_file
            .read_to_end(&mut contents)
            .expect("read destination");
        assert_eq!(contents, b"hello filesystem");
    }

    #[test]
    fn set_readonly_flips_metadata_flag() {
        let _guard = TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("flag.txt");
        write(&path, b"flag").expect("write");

        set_readonly(&path, true).expect("set readonly");
        let meta = metadata(&path).expect("metadata");
        assert!(meta.is_readonly());

        set_readonly(&path, false).expect("unset readonly");
        let meta = metadata(&path).expect("metadata");
        assert!(!meta.is_readonly());
    }

    #[test]
    fn replace_provider_restores_previous() {
        let _guard = TEST_LOCK.lock().unwrap();
        let original = current_provider();
        let custom: Arc<dyn FsProvider> = Arc::new(UnsupportedProvider);
        {
            let _guard = replace_provider(custom.clone());
            let active = current_provider();
            assert!(Arc::ptr_eq(&active, &custom));
        }
        let final_provider = current_provider();
        assert!(Arc::ptr_eq(&final_provider, &original));
    }

    #[test]
    fn with_provider_restores_even_on_panic() {
        let _guard = TEST_LOCK.lock().unwrap();
        let original = current_provider();
        let custom: Arc<dyn FsProvider> = Arc::new(UnsupportedProvider);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_provider_override(custom.clone(), || {
                let active = current_provider();
                assert!(Arc::ptr_eq(&active, &custom));
                panic!("boom");
            })
        }));
        assert!(result.is_err());
        let final_provider = current_provider();
        assert!(Arc::ptr_eq(&final_provider, &original));
    }
}
