use async_trait::async_trait;
use log::warn;
use once_cell::sync::OnceCell;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
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

pub mod data_contract;

use data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};

#[async_trait(?Send)]
pub trait FileHandle: Read + Write + Seek + Send + Sync {
    async fn flush_async(&mut self) -> io::Result<()> {
        self.flush()
    }

    async fn sync_all_async(&mut self) -> io::Result<()> {
        self.flush_async().await
    }
}

#[async_trait(?Send)]
impl FileHandle for std::fs::File {
    async fn sync_all_async(&mut self) -> io::Result<()> {
        std::fs::File::sync_all(self)
    }
}

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
        let resolved = resolve_path(path.as_ref());
        with_provider(|provider| provider.open(&resolved, &self.flags)).map(File::from_handle)
    }

    pub async fn open_async(&self, path: impl AsRef<Path>) -> io::Result<File> {
        let resolved = resolve_path(path.as_ref());
        let provider = current_provider();
        provider
            .open_async(&resolved, &self.flags)
            .await
            .map(File::from_handle)
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
    hash: Option<String>,
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
            hash: None,
        }
    }

    pub fn new_with_hash(
        file_type: FsFileType,
        len: u64,
        modified: Option<SystemTime>,
        readonly: bool,
        hash: Option<String>,
    ) -> Self {
        Self {
            file_type,
            len,
            modified,
            readonly,
            hash,
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

    pub fn hash(&self) -> Option<&str> {
        self.hash.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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

#[derive(Clone, Debug)]
pub struct ReadManyEntry {
    path: PathBuf,
    bytes: Option<Vec<u8>>,
    error: Option<String>,
}

impl ReadManyEntry {
    pub fn new(path: PathBuf, bytes: Option<Vec<u8>>) -> Self {
        Self {
            path,
            bytes,
            error: None,
        }
    }

    pub fn with_error(path: PathBuf, error: String) -> Self {
        Self {
            path,
            bytes: None,
            error: Some(error),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }

    pub fn into_bytes(self) -> Option<Vec<u8>> {
        self.bytes
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogFilter {
    pub patterns: Vec<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenFileDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
    pub filters: Vec<OpenFileDialogFilter>,
    pub multiselect: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogSelection {
    pub paths: Vec<PathBuf>,
    pub filter_index: Option<usize>,
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

#[async_trait(?Send)]
pub trait FsProvider: Send + Sync + 'static {
    fn current_dir_override(&self) -> Option<PathBuf> {
        None
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>>;
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()>;
    async fn remove_file(&self, path: &Path) -> io::Result<()>;
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata>;
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata>;
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>>;
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    async fn create_dir(&self, path: &Path) -> io::Result<()>;
    async fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    async fn remove_dir(&self, path: &Path) -> io::Result<()>;
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()>;
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()>;

    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        let mut entries = Vec::with_capacity(paths.len());
        for path in paths {
            let entry = match self.read(path).await {
                Ok(payload) => ReadManyEntry::new(path.clone(), Some(payload)),
                Err(error) => {
                    warn!(
                        "fs.read_many.miss path={} kind={:?} error={}",
                        path.to_string_lossy(),
                        error.kind(),
                        error
                    );
                    ReadManyEntry::with_error(
                        path.clone(),
                        format!("kind={:?}; error={}", error.kind(), error),
                    )
                }
            };
            entries.push(entry);
        }
        Ok(entries)
    }

    async fn data_manifest_descriptor(
        &self,
        _request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "data manifest descriptor is unsupported by this provider",
        ))
    }

    async fn data_chunk_upload_targets(
        &self,
        _request: &DataChunkUploadRequest,
    ) -> io::Result<Vec<DataChunkUploadTarget>> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "data chunk upload targets are unsupported by this provider",
        ))
    }

    async fn data_upload_chunk(
        &self,
        _target: &DataChunkUploadTarget,
        _data: &[u8],
    ) -> io::Result<()> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "data chunk upload is unsupported by this provider",
        ))
    }

    async fn select_file_open(
        &self,
        _request: &OpenFileDialogRequest,
    ) -> io::Result<Option<OpenFileDialogSelection>> {
        Ok(None)
    }
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

    pub async fn open_async(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        opts.open_async(path).await
    }

    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        opts.open(path)
    }

    pub async fn create_async(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        opts.open_async(path).await
    }

    pub async fn flush_async(&mut self) -> io::Result<()> {
        self.inner.flush_async().await
    }

    pub async fn sync_all_async(&mut self) -> io::Result<()> {
        self.inner.sync_all_async().await
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

struct ProviderState {
    provider: Arc<dyn FsProvider>,
    current_dir_override: Option<PathBuf>,
}

static PROVIDER_STATE: OnceCell<RwLock<ProviderState>> = OnceCell::new();
static PROVIDER_OVERRIDE_LOCK: OnceCell<Mutex<()>> = OnceCell::new();

fn provider_state_lock() -> &'static RwLock<ProviderState> {
    PROVIDER_STATE.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        let current_dir_override = Some(PathBuf::from("/"));
        #[cfg(not(target_arch = "wasm32"))]
        let current_dir_override = None;

        RwLock::new(ProviderState {
            provider: default_provider(),
            current_dir_override,
        })
    })
}

/// Serializes tests and embedders that temporarily replace the process-wide
/// filesystem provider.
pub fn provider_override_lock() -> MutexGuard<'static, ()> {
    PROVIDER_OVERRIDE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn current_dir_override() -> Option<PathBuf> {
    provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned")
        .current_dir_override
        .clone()
}

fn replace_current_dir_override(value: Option<PathBuf>) -> Option<PathBuf> {
    let mut guard = provider_state_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    std::mem::replace(&mut guard.current_dir_override, value)
}

fn with_provider<T>(f: impl FnOnce(&dyn FsProvider) -> T) -> T {
    let guard = provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned");
    f(&*guard.provider)
}

fn resolve_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let state = provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned");
    if let Some(base) = &state.current_dir_override {
        return base.join(path);
    }
    path.to_path_buf()
}

fn next_current_dir_override(
    current: Option<&PathBuf>,
    provider_default: Option<PathBuf>,
) -> Option<PathBuf> {
    match provider_default {
        Some(default) => current.cloned().or(Some(default)),
        None => None,
    }
}

pub fn set_provider(provider: Arc<dyn FsProvider>) {
    let provider_default_current_dir = provider.current_dir_override();
    let mut guard = provider_state_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    let current_dir_override = next_current_dir_override(
        guard.current_dir_override.as_ref(),
        provider_default_current_dir,
    );
    guard.provider = provider;
    guard.current_dir_override = current_dir_override;
}

/// Temporarily replace the active provider and return a guard that restores the
/// previous provider when dropped. Useful for tests that need to install a mock
/// filesystem without permanently mutating global state.
pub fn replace_provider(provider: Arc<dyn FsProvider>) -> ProviderGuard {
    let provider_default_current_dir = provider.current_dir_override();
    let mut guard = provider_state_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    let previous = guard.provider.clone();
    let previous_current_dir = guard.current_dir_override.clone();
    let current_dir_override = next_current_dir_override(
        guard.current_dir_override.as_ref(),
        provider_default_current_dir,
    );
    guard.provider = provider;
    guard.current_dir_override = current_dir_override;
    ProviderGuard {
        previous,
        previous_current_dir,
    }
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
    provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned")
        .provider
        .clone()
}

pub fn current_dir() -> io::Result<PathBuf> {
    if let Some(current) = current_dir_override() {
        return Ok(current);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::current_dir()
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(PathBuf::from("/"))
    }
}

pub fn set_current_dir(path: impl AsRef<Path>) -> io::Result<()> {
    if current_dir_override().is_some() {
        futures::executor::block_on(set_current_dir_async(path.as_ref().to_path_buf()))
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::set_current_dir(path)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(())
        }
    }
}

pub async fn set_current_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    if current_dir_override().is_some() {
        let mut target = PathBuf::from(path.as_ref());
        if !target.is_absolute() {
            let base = current_dir()?;
            target = base.join(target);
        }
        let canonical = canonicalize_async(&target).await.unwrap_or(target.clone());
        let metadata = metadata_async(&canonical).await?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("Not a directory: {}", canonical.display()),
            ));
        }
        replace_current_dir_override(Some(canonical));
        Ok(())
    } else {
        set_current_dir(path)
    }
}

pub struct ProviderGuard {
    previous: Arc<dyn FsProvider>,
    previous_current_dir: Option<PathBuf>,
}

impl Drop for ProviderGuard {
    fn drop(&mut self) {
        let mut guard = provider_state_lock()
            .write()
            .expect("filesystem provider lock poisoned");
        guard.provider = self.previous.clone();
        guard.current_dir_override = self.previous_current_dir.clone();
    }
}

pub async fn read_many_async(paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
    let resolved = paths
        .iter()
        .map(|path| resolve_path(path.as_path()))
        .collect::<Vec<_>>();
    let provider = current_provider();
    provider.read_many(&resolved).await
}

pub async fn read_async(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.read(&resolved).await
}

pub async fn read_to_string_async(path: impl AsRef<Path>) -> io::Result<String> {
    let bytes = read_async(path).await?;
    String::from_utf8(bytes).map_err(|err| io::Error::new(ErrorKind::InvalidData, err.utf8_error()))
}

pub async fn write_async(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.write(&resolved, data.as_ref()).await
}

pub async fn remove_file_async(path: impl AsRef<Path>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.remove_file(&resolved).await
}

pub async fn metadata_async(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.metadata(&resolved).await
}

pub async fn symlink_metadata_async(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.symlink_metadata(&resolved).await
}

pub async fn read_dir_async(path: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.read_dir(&resolved).await
}

pub async fn canonicalize_async(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.canonicalize(&resolved).await
}

pub async fn create_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.create_dir(&resolved).await
}

pub async fn create_dir_all_async(path: impl AsRef<Path>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.create_dir_all(&resolved).await
}

pub async fn remove_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.remove_dir(&resolved).await
}

pub async fn remove_dir_all_async(path: impl AsRef<Path>) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.remove_dir_all(&resolved).await
}

pub async fn rename_async(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let resolved_from = resolve_path(from.as_ref());
    let resolved_to = resolve_path(to.as_ref());
    let provider = current_provider();
    provider.rename(&resolved_from, &resolved_to).await
}

pub async fn set_readonly_async(path: impl AsRef<Path>, readonly: bool) -> io::Result<()> {
    let resolved = resolve_path(path.as_ref());
    let provider = current_provider();
    provider.set_readonly(&resolved, readonly).await
}

pub async fn select_file_open_async(
    request: &OpenFileDialogRequest,
) -> io::Result<Option<OpenFileDialogSelection>> {
    let mut resolved = request.clone();
    if let Some(default_path) = resolved.default_path.as_mut() {
        *default_path = resolve_path(default_path);
    }
    let provider = current_provider();
    provider.select_file_open(&resolved).await
}

pub async fn data_manifest_descriptor_async(
    request: &DataManifestRequest,
) -> io::Result<DataManifestDescriptor> {
    let provider = current_provider();
    provider.data_manifest_descriptor(request).await
}

pub async fn data_chunk_upload_targets_async(
    request: &DataChunkUploadRequest,
) -> io::Result<Vec<DataChunkUploadTarget>> {
    let provider = current_provider();
    provider.data_chunk_upload_targets(request).await
}

pub async fn data_upload_chunk_async(
    target: &DataChunkUploadTarget,
    data: &[u8],
) -> io::Result<()> {
    let provider = current_provider();
    provider.data_upload_chunk(target, data).await
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
        Arc::new(NativeFsProvider)
    }
    #[cfg(target_arch = "wasm32")]
    {
        Arc::new(PlaceholderFsProvider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::collections::{HashMap, HashSet};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::sync::Mutex;
    use tempfile::tempdir;

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct UnsupportedProvider;

    struct AsyncOpenProvider {
        opened_async: Arc<Mutex<bool>>,
        flushed_async: Arc<Mutex<bool>>,
    }

    struct TestProviderStateGuard {
        previous_provider: Arc<dyn FsProvider>,
        previous_current_dir: Option<PathBuf>,
    }

    struct ProcessCwdGuard {
        previous: PathBuf,
    }

    struct VirtualFsProvider {
        default_current_dir: PathBuf,
        dirs: Mutex<HashSet<PathBuf>>,
        files: Mutex<HashMap<PathBuf, Vec<u8>>>,
    }

    impl Drop for ProcessCwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.previous);
        }
    }

    impl TestProviderStateGuard {
        fn capture() -> Self {
            let guard = provider_state_lock()
                .read()
                .expect("filesystem provider lock poisoned");
            Self {
                previous_provider: guard.provider.clone(),
                previous_current_dir: guard.current_dir_override.clone(),
            }
        }
    }

    impl Drop for TestProviderStateGuard {
        fn drop(&mut self) {
            let mut guard = provider_state_lock()
                .write()
                .expect("filesystem provider lock poisoned");
            guard.provider = self.previous_provider.clone();
            guard.current_dir_override = self.previous_current_dir.clone();
        }
    }

    impl VirtualFsProvider {
        fn new(default_current_dir: impl Into<PathBuf>, dirs: &[&str]) -> Self {
            let default_current_dir = default_current_dir.into();
            let mut all_dirs = HashSet::from([default_current_dir.clone()]);
            for dir in dirs {
                all_dirs.insert(PathBuf::from(dir));
            }
            Self {
                default_current_dir,
                dirs: Mutex::new(all_dirs),
                files: Mutex::new(HashMap::new()),
            }
        }

        fn file_bytes(&self, path: impl AsRef<Path>) -> Option<Vec<u8>> {
            self.files.lock().unwrap().get(path.as_ref()).cloned()
        }
    }

    struct AsyncTestHandle {
        cursor: usize,
        data: Vec<u8>,
        flushed_async: Arc<Mutex<bool>>,
    }

    impl Read for AsyncTestHandle {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let remaining = self.data.len().saturating_sub(self.cursor);
            let to_read = remaining.min(buf.len());
            buf[..to_read].copy_from_slice(&self.data[self.cursor..self.cursor + to_read]);
            self.cursor += to_read;
            Ok(to_read)
        }
    }

    impl Write for AsyncTestHandle {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let end = self.cursor + buf.len();
            if end > self.data.len() {
                self.data.resize(end, 0);
            }
            self.data[self.cursor..end].copy_from_slice(buf);
            self.cursor = end;
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Seek for AsyncTestHandle {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            let next = match pos {
                SeekFrom::Start(offset) => offset as i64,
                SeekFrom::End(offset) => self.data.len() as i64 + offset,
                SeekFrom::Current(offset) => self.cursor as i64 + offset,
            };
            if next < 0 {
                return Err(io::Error::new(ErrorKind::InvalidInput, "seek before start"));
            }
            self.cursor = next as usize;
            Ok(self.cursor as u64)
        }
    }

    #[async_trait(?Send)]
    impl FileHandle for AsyncTestHandle {
        async fn flush_async(&mut self) -> io::Result<()> {
            *self.flushed_async.lock().unwrap() = true;
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl FsProvider for UnsupportedProvider {
        fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            Err(unsupported())
        }

        async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Err(unsupported())
        }

        async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_file(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
            Err(unsupported())
        }

        async fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
            Err(unsupported())
        }

        async fn create_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
            Err(unsupported())
        }

        async fn data_manifest_descriptor(
            &self,
            _request: &DataManifestRequest,
        ) -> io::Result<DataManifestDescriptor> {
            Err(unsupported())
        }

        async fn data_chunk_upload_targets(
            &self,
            _request: &DataChunkUploadRequest,
        ) -> io::Result<Vec<DataChunkUploadTarget>> {
            Err(unsupported())
        }

        async fn data_upload_chunk(
            &self,
            _target: &DataChunkUploadTarget,
            _data: &[u8],
        ) -> io::Result<()> {
            Err(unsupported())
        }
    }

    #[async_trait(?Send)]
    impl FsProvider for VirtualFsProvider {
        fn current_dir_override(&self) -> Option<PathBuf> {
            Some(self.default_current_dir.clone())
        }

        fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            Err(unsupported())
        }

        async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            self.files
                .lock()
                .unwrap()
                .get(path)
                .cloned()
                .ok_or_else(|| io::Error::new(ErrorKind::NotFound, path.display().to_string()))
        }

        async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
            self.files
                .lock()
                .unwrap()
                .insert(path.to_path_buf(), data.to_vec());
            Ok(())
        }

        async fn remove_file(&self, path: &Path) -> io::Result<()> {
            self.files.lock().unwrap().remove(path);
            Ok(())
        }

        async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
            if self.dirs.lock().unwrap().contains(path) {
                return Ok(FsMetadata::new(FsFileType::Directory, 0, None, false));
            }
            if let Some(bytes) = self.files.lock().unwrap().get(path) {
                return Ok(FsMetadata::new(
                    FsFileType::File,
                    bytes.len() as u64,
                    None,
                    false,
                ));
            }
            Err(io::Error::new(
                ErrorKind::NotFound,
                path.display().to_string(),
            ))
        }

        async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
            self.metadata(path).await
        }

        async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
            Ok(Vec::new())
        }

        async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
            Ok(path.to_path_buf())
        }

        async fn create_dir(&self, path: &Path) -> io::Result<()> {
            self.dirs.lock().unwrap().insert(path.to_path_buf());
            Ok(())
        }

        async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
            let mut dirs = self.dirs.lock().unwrap();
            for ancestor in path.ancestors() {
                dirs.insert(ancestor.to_path_buf());
            }
            Ok(())
        }

        async fn remove_dir(&self, path: &Path) -> io::Result<()> {
            self.dirs.lock().unwrap().remove(path);
            Ok(())
        }

        async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
            self.dirs
                .lock()
                .unwrap()
                .retain(|dir| !dir.starts_with(path));
            Ok(())
        }

        async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            let mut files = self.files.lock().unwrap();
            let data = files
                .remove(from)
                .ok_or_else(|| io::Error::new(ErrorKind::NotFound, from.display().to_string()))?;
            files.insert(to.to_path_buf(), data);
            Ok(())
        }

        async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl FsProvider for AsyncOpenProvider {
        fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
            Err(unsupported())
        }

        async fn open_async(
            &self,
            _path: &Path,
            _flags: &OpenFlags,
        ) -> io::Result<Box<dyn FileHandle>> {
            *self.opened_async.lock().unwrap() = true;
            Ok(Box::new(AsyncTestHandle {
                cursor: 0,
                data: b"async contents".to_vec(),
                flushed_async: self.flushed_async.clone(),
            }))
        }

        async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Err(unsupported())
        }

        async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_file(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
            Err(unsupported())
        }

        async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
            Err(unsupported())
        }

        async fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
            Err(unsupported())
        }

        async fn create_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
            Err(unsupported())
        }

        async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
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
        futures::executor::block_on(write_async(&path, b"flag")).expect("write");

        futures::executor::block_on(set_readonly_async(&path, true)).expect("set readonly");
        let meta = futures::executor::block_on(metadata_async(&path)).expect("metadata");
        assert!(meta.is_readonly());

        futures::executor::block_on(set_readonly_async(&path, false)).expect("unset readonly");
        let meta = futures::executor::block_on(metadata_async(&path)).expect("metadata");
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
    #[cfg(not(target_arch = "wasm32"))]
    fn native_provider_replacement_preserves_process_cwd_resolution() {
        let _guard = TEST_LOCK.lock().unwrap();
        let temp = tempdir().expect("tempdir");
        let previous = std::env::current_dir().expect("current dir");
        let _cwd_guard = ProcessCwdGuard { previous };
        std::env::set_current_dir(temp.path()).expect("set temp cwd");

        let _provider_guard = replace_provider(Arc::new(NativeFsProvider));
        let current = current_dir().expect("vfs current dir");
        let expected = std::fs::canonicalize(temp.path()).expect("canonical temp");
        assert_eq!(current, expected);

        futures::executor::block_on(write_async("native-relative.txt", b"native"))
            .expect("write relative path");
        assert_eq!(
            std::fs::read_to_string(temp.path().join("native-relative.txt")).expect("read file"),
            "native"
        );

        std::fs::create_dir(temp.path().join("child")).expect("create child");
        set_current_dir("child").expect("set child cwd");
        assert_eq!(
            std::env::current_dir().expect("process cwd"),
            expected.join("child")
        );
    }

    #[test]
    fn set_provider_initializes_virtual_cwd_from_provider_default() {
        let _guard = TEST_LOCK.lock().unwrap();
        let _state_guard = TestProviderStateGuard::capture();
        replace_current_dir_override(None);

        set_provider(Arc::new(VirtualFsProvider::new("/sandbox", &[])));

        assert_eq!(
            current_dir().expect("virtual cwd"),
            PathBuf::from("/sandbox")
        );
    }

    #[test]
    fn set_provider_preserves_existing_virtual_cwd() {
        let _guard = TEST_LOCK.lock().unwrap();
        let _state_guard = TestProviderStateGuard::capture();
        let initial = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        set_provider(initial);
        set_current_dir("/workspace").expect("set virtual cwd");

        let replacement = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        set_provider(replacement.clone());

        assert_eq!(
            current_dir().expect("virtual cwd"),
            PathBuf::from("/workspace")
        );
        futures::executor::block_on(write_async("data.txt", b"virtual")).expect("write relative");
        assert_eq!(
            replacement.file_bytes("/workspace/data.txt").as_deref(),
            Some(&b"virtual"[..])
        );
        assert_eq!(replacement.file_bytes("data.txt"), None);
    }

    #[test]
    fn replace_provider_preserves_existing_virtual_cwd() {
        let _guard = TEST_LOCK.lock().unwrap();
        let initial = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        let _initial_guard = replace_provider(initial);
        set_current_dir("/workspace").expect("set virtual cwd");

        let replacement = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        {
            let _replacement_guard = replace_provider(replacement.clone());

            assert_eq!(
                current_dir().expect("virtual cwd"),
                PathBuf::from("/workspace")
            );
            futures::executor::block_on(write_async("nested.txt", b"replacement"))
                .expect("write relative");
        }

        assert_eq!(
            replacement.file_bytes("/workspace/nested.txt").as_deref(),
            Some(&b"replacement"[..])
        );
        assert_eq!(replacement.file_bytes("nested.txt"), None);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn native_provider_replacement_clears_virtual_cwd_override() {
        let _guard = TEST_LOCK.lock().unwrap();
        let temp = tempdir().expect("tempdir");
        let previous = std::env::current_dir().expect("current dir");
        let _cwd_guard = ProcessCwdGuard { previous };
        std::env::set_current_dir(temp.path()).expect("set temp cwd");

        let virtual_provider = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        let _virtual_guard = replace_provider(virtual_provider);
        set_current_dir("/workspace").expect("set virtual cwd");

        {
            let _native_guard = replace_provider(Arc::new(NativeFsProvider));
            let expected = std::fs::canonicalize(temp.path()).expect("canonical temp");
            assert_eq!(current_dir().expect("native cwd"), expected);

            futures::executor::block_on(write_async("native.txt", b"native"))
                .expect("write native relative");
            assert_eq!(
                std::fs::read_to_string(temp.path().join("native.txt")).expect("read native file"),
                "native"
            );
        }
    }

    #[test]
    fn open_async_and_flush_async_use_provider_async_paths() {
        let _guard = TEST_LOCK.lock().unwrap();
        let opened_async = Arc::new(Mutex::new(false));
        let flushed_async = Arc::new(Mutex::new(false));
        let provider = Arc::new(AsyncOpenProvider {
            opened_async: opened_async.clone(),
            flushed_async: flushed_async.clone(),
        });
        let _provider_guard = replace_provider(provider);

        let mut file =
            futures::executor::block_on(OpenOptions::new().read(true).open_async("data.txt"))
                .expect("async open");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("read contents");
        futures::executor::block_on(file.flush_async()).expect("async flush");

        assert_eq!(contents, "async contents");
        assert!(*opened_async.lock().unwrap());
        assert!(*flushed_async.lock().unwrap());
    }

    #[test]
    fn select_file_open_defaults_to_cancelled_selection() {
        let _guard = TEST_LOCK.lock().unwrap();
        let provider: Arc<dyn FsProvider> = Arc::new(UnsupportedProvider);
        let _provider_guard = replace_provider(provider);
        let request = OpenFileDialogRequest {
            title: Some("Open".to_string()),
            default_path: Some(PathBuf::from("data")),
            filters: vec![OpenFileDialogFilter {
                patterns: vec!["*.csv".to_string()],
                description: Some("CSV files".to_string()),
            }],
            multiselect: false,
        };

        let selection =
            futures::executor::block_on(select_file_open_async(&request)).expect("select file");

        assert_eq!(selection, None);
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
