use async_trait::async_trait;
use log::warn;
use once_cell::sync::OnceCell;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::SystemTime;

mod dialog;
#[cfg(not(target_arch = "wasm32"))]
mod file_lock;
mod invocation;
mod mapped_backing;
mod memory;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
mod native_path;
mod native_path_intent;
mod provider_path;
mod captured_read;
mod resource;
mod selected_provider;
mod staged_output;
mod unavailable;
#[cfg(not(target_arch = "wasm32"))]
pub use file_lock::ExclusiveFileLock;
mod publication;
pub use publication::{AtomicCreateResult, AtomicReplaceOutcome, PublicationDurability};
#[cfg(not(target_arch = "wasm32"))]
pub mod remote;
#[cfg(not(target_arch = "wasm32"))]
pub mod sandbox;
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(not(target_arch = "wasm32"))]
mod workspace_native;

pub use captured_read::{CapturedExecutionScope, CapturedPinnedScope, CapturedReadScope};
pub use dialog::*;
pub use invocation::{
    register_invocation_filesystem_resolver, InvocationFilesystem, InvocationFilesystemSelection,
};
pub use mapped_backing::{
    capture_mapped_backing, capture_mapped_backing_on, CapturedMappedBacking, FsMappedBacking,
    FsMappedWriteDisposition, FsMappedWriteFailure, FsMappedWriteOutcome,
};
pub use memory::MemoryFsProvider;
#[cfg(not(target_arch = "wasm32"))]
pub use native::NativeFsProvider;
#[cfg(not(target_arch = "wasm32"))]
pub use native_path::{
    capture_native_path, CapturedNativePath, NativePathAuthority, NativePathGrant,
    NativeProcessRevocation,
};
pub use native_path_intent::NativePathIntent;
pub use provider_path::{capture_provider_for_paths, capture_provider_path, FsPathNamespace};
use provider_path::{capture_provider_many, capture_provider_pair};
#[cfg(not(target_arch = "wasm32"))]
pub use remote::{RemoteFsConfig, RemoteFsProvider};
pub use resource::{
    capture_resource_lease, capture_resource_lease_for_effect, CapturedFsResourceLease,
    FsBoundedInput, FsFilePayloadContract, FsManagedOpenedInputCost, FsManagedReadBacking,
    FsManagedReadContract, FsResourceContract, FsResourceLease, FsResourceLeaseHandle,
    FsResourceRevision,
};
#[cfg(not(target_arch = "wasm32"))]
pub use sandbox::{CapturedIo, SandboxFsProvider};
pub use selected_provider::SelectedFsProvider;
pub use staged_output::{FsPublicationDisposition, FsPublicationFailure, FsStagedOutput};
pub use unavailable::UnavailableFsProvider;
#[cfg(target_arch = "wasm32")]
pub use wasm::PlaceholderFsProvider;

pub mod data_contract;
pub mod execution;
pub mod pinned;
pub mod workspace;
pub mod workspace_access;

use data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};

#[async_trait(?Send)]
pub trait FileHandle: Read + Write + Seek + Send + Sync {
    /// Providers may override these for suspension-aware streaming. The
    /// default retains their synchronous handle semantics for one bounded
    /// chunk; callers can still cooperate between chunks.
    async fn read_async(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.read(bytes)
    }

    async fn write_async(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.write(bytes)
    }

    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "file handle metadata is not supported by this provider",
        ))
    }

    async fn flush_async(&mut self) -> io::Result<()> {
        self.flush()
    }

    async fn sync_all_async(&mut self) -> io::Result<()> {
        self.flush_async().await
    }
}

#[async_trait(?Send)]
impl FileHandle for std::fs::File {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        let meta = std::fs::File::metadata(self)?;
        let file_type = meta.file_type();
        Ok(FsMetadata {
            file_type: if file_type.is_dir() {
                FsFileType::Directory
            } else if file_type.is_file() {
                FsFileType::File
            } else if file_type.is_symlink() {
                FsFileType::Symlink
            } else {
                FsFileType::Other
            },
            len: meta.len(),
            modified: meta.modified().ok(),
            readonly: meta.permissions().readonly(),
            hash: None,
        })
    }

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
        let (provider, resolved) = capture_provider_path(path.as_ref())?;
        provider.open(&resolved, &self.flags).map(File::from_handle)
    }

    /// Open an absolute path through a previously captured provider, without
    /// consulting the ambient provider or working directory again.
    pub fn open_on(&self, provider: &dyn FsProvider, path: impl AsRef<Path>) -> io::Result<File> {
        let path = path.as_ref();
        if !provider.path_namespace().is_absolute(path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "captured filesystem opens require an absolute path",
            ));
        }
        provider.open(path, &self.flags).map(File::from_handle)
    }

    pub async fn open_async(&self, path: impl AsRef<Path>) -> io::Result<File> {
        let (provider, path) = capture_provider_path(path.as_ref())?;
        self.open_async_on(provider.as_ref(), path).await
    }

    /// Open using the invocation's captured filesystem authority. The caller
    /// may pass an absolute path captured with that authority before suspension.
    pub async fn open_async_on(
        &self,
        provider: &dyn FsProvider,
        path: impl AsRef<Path>,
    ) -> io::Result<File> {
        let resolved = path.as_ref();
        if !provider.path_namespace().is_absolute(resolved) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "captured filesystem opens require an absolute path",
            ));
        }
        provider
            .open_async(resolved, &self.flags)
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

impl DirEntry {
    pub fn new(path: PathBuf, file_name: OsString, file_type: FsFileType) -> Self {
        Self {
            path,
            file_name,
            file_type,
        }
    }

    /// Requested storage of this response entry, including owned buffer capacities.
    /// Allocator rounding and OS iterator internals are excluded.
    pub fn resident_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64)
            .checked_add(self.path.capacity() as u64)?
            .checked_add(self.file_name.capacity() as u64)
    }

    pub fn into_path(self) -> PathBuf {
        self.path
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
    /// Capability metadata only; capture and publication still perform admission.
    fn supports_workspace_transactions(&self) -> bool {
        false
    }
    /// Coherent access through this provider's existing publication/recovery authority.
    fn workspace_access(
        &self,
        _root: &Path,
        _writable: bool,
        _maximum_resident_bytes: u64,
    ) -> io::Result<Box<dyn workspace_access::WorkspaceAccess>> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "provider coherent workspace access is unsupported",
        ))
    }
    /// Capture one coherent project inventory, including immutable file versions.
    async fn capture_workspace(
        &self,
        _request: workspace::WorkspaceCaptureRequest,
    ) -> io::Result<workspace::WorkspaceSnapshot> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "coherent workspace capture is unsupported",
        ))
    }

    /// Recover the result of an exact publication request without replaying edits.
    /// This supports lost acknowledgements and rename/delete retries after restart.
    async fn lookup_workspace_publication(
        &self,
        _request: &workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<workspace::WorkspacePublicationOutcome>> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "workspace publication receipts are unsupported",
        ))
    }

    /// Publish against exactly the captured revision. Successful retries return
    /// the same durable receipt even if later edits have changed the project.
    async fn publish_workspace(
        &self,
        _request: workspace::WorkspacePublicationRequest,
    ) -> io::Result<workspace::WorkspacePublicationOutcome> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "conditional workspace publication is unsupported",
        ))
    }
    /// Arbitrate execution ownership inside this provider's existing project
    /// scope. The authority supplies time and stores the transition atomically.
    /// A remote client must never emulate this with reads or a local clock.
    async fn transition_execution(
        &self,
        _work: &execution::ExecutionWorkId,
        _command: execution::ExecutionLeaseCommand,
    ) -> io::Result<execution::ExecutionLeaseOutcome> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "provider execution authority is unsupported",
        ))
    }
    /// Validate ownership at the publication commit boundary under the same
    /// provider lock as the files and receipt. A separate lease check is unsafe.
    async fn publish_workspace_fenced(
        &self,
        _request: workspace::WorkspacePublicationRequest,
        _lease: &execution::ExecutionLease,
    ) -> io::Result<execution::FencedWorkspacePublicationOutcome> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "fenced workspace publication is unsupported",
        ))
    }
    /// Path syntax owned by this provider. The default preserves native OS
    /// qualification for existing native embedders; a provider exposing a
    /// virtual rooted namespace must opt in even when hosted on Windows.
    fn path_namespace(&self) -> FsPathNamespace {
        #[cfg(target_arch = "wasm32")]
        {
            FsPathNamespace::VirtualRooted
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            FsPathNamespace::NativeOs
        }
    }
    /// Opt in to an opened, fixed-extent range owner for an existing file.
    /// `None` means the provider cannot supply live range semantics; it does
    /// not mean the source is absent. This is not a memory-peak certificate.
    async fn open_mapped_backing(
        &self,
        _resolved_absolute: &Path,
        _writable: bool,
    ) -> io::Result<Option<Box<dyn FsMappedBacking>>> {
        Ok(None)
    }

    /// Acquire an exclusive provider-owned staging object for one already
    /// resolved absolute destination. The returned owner must abort private
    /// staging on Drop before commit dispatch. Dropping an in-flight remote
    /// publication may leave its commit outcome indeterminate.
    /// `None` means this provider cannot supply that guarantee; callers must
    /// not reconstruct it from a public temporary path and `remove_file`.
    /// RemoteFsProvider currently does not opt in: its CRUD/versioned upload
    /// endpoints do not provide this private abort/commit-disposition contract.
    /// Callers requiring staged replacement (such as writecell/dlmwrite)
    /// therefore reject that provider before modifying the destination.
    fn begin_staged_replace(
        &self,
        _resolved_absolute_destination: &Path,
    ) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        Ok(None)
    }

    /// Opt in to direct OS-path access by a native library. The path is
    /// absolute and resolved with this provider's captured working directory.
    /// `None` means the provider has no safe direct native-path route; callers
    /// must not fall back to the process filesystem.
    #[cfg(not(target_arch = "wasm32"))]
    fn acquire_native_path(
        &self,
        _resolved_absolute: &Path,
        _intent: NativePathIntent,
    ) -> io::Result<Option<NativePathGrant>> {
        Ok(None)
    }

    /// Opt-in opened-source bound for an already resolved absolute path.
    /// `None` means this provider cannot certify direct, extent-limited reads;
    /// callers retain their ordinary route. The certificate and handle must
    /// refer to the same open, never to a separate path metadata lookup.
    async fn open_bounded_input(&self, _path: &Path) -> io::Result<Option<FsBoundedInput>> {
        Ok(None)
    }

    /// Pure, allocation-free pre-effect managed-byte contract for this exact
    /// provider's bounded input route. `None` is unavailable, never zero.
    /// Callers must still certify their path, codec, and output allocations.
    fn managed_read_contract(&self) -> Option<FsManagedReadContract> {
        None
    }

    /// Opt-in resource authority for this exact provider instance. The lease
    /// must keep its claims valid while retained by a caller. This does not
    /// certify codec, path, result, or whole-invocation allocations.
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        None
    }

    /// Allocation-free static leases can serve managed reads; existing
    /// instance-owned authorities use the Arc-valued compatibility method.
    fn resource_lease_handle(&self) -> Option<FsResourceLeaseHandle> {
        self.resource_lease().map(FsResourceLeaseHandle::Owned)
    }

    /// Whether this installed provider can service ordinary file reads.
    /// Hosts use this metadata for capability admission; it does not promise
    /// that a particular path exists or that a subsequent read will succeed.
    /// Providers whose read operation is unconditionally Unsupported must
    /// override this to return false.
    fn supports_read(&self) -> bool {
        true
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        None
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>>;
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
    /// Read an immutable version through the provider's normal authorization.
    /// Never fall back to a live head when the version is unavailable.
    async fn read_version(
        &self,
        _path: &Path,
        _version: &str,
        _size: u64,
        _snapshot: &str,
    ) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "immutable version reads are unsupported by this provider",
        ))
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()>;
    /// Atomically create (None) or replace the exact previously published version.
    /// Implementations must never emulate this with a read followed by a write.
    async fn publish_output(
        &self,
        _path: &Path,
        _data: &[u8],
        _expected: Option<&str>,
    ) -> io::Result<String> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "conditional output publication is unsupported",
        ))
    }
    async fn remove_output(&self, _path: &Path, _expected: &str) -> io::Result<()> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "conditional output removal is unsupported",
        ))
    }
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
    /// Replace a compatible destination as part of one provider-owned rename.
    ///
    /// Implementations must not delete the destination before attempting the move.
    /// Unsupported replacement, incompatible types, and nonempty destination
    /// directories must fail without discarding either existing tree. This does
    /// not require cross-volume emulation or a transaction across multiple moves.
    async fn rename_replace(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "filesystem provider does not support safe rename replacement",
        ))
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()>;

    /// Atomically publishes `bytes` if `path` does not exist.
    ///
    /// Implementations must provide a linearizable no-replace operation: exactly one competing
    /// publisher may return `Published`, and readers must observe either no entry or the complete
    /// winning payload, never a partially written destination. `Existing` contains a complete
    /// snapshot of the payload which won publication. The default deliberately fails instead of
    /// emulating these semantics with a racy existence check followed by a write.
    async fn atomic_create_new(
        &self,
        _path: &Path,
        _bytes: &[u8],
    ) -> io::Result<AtomicCreateResult> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "atomic create-if-absent is unsupported by this provider",
        ))
    }

    /// Atomically replaces a path with one complete payload. Concurrent callers
    /// are linearized at publication; ordering/fencing belongs to the caller.
    /// Err guarantees this invocation did not publish its replacement. After
    /// publication, uncertain crash durability is returned explicitly with its
    /// original diagnostic. Unsupported providers must not emulate this by write.
    async fn atomic_replace(
        &self,
        _path: &Path,
        _bytes: &[u8],
    ) -> io::Result<AtomicReplaceOutcome> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "atomic replacement is unsupported by this provider",
        ))
    }

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

    /// Open through the caller's retained filesystem authority. A global
    /// provider replacement between admission and execution cannot redirect
    /// this input to another owner.
    pub fn open_on(provider: &dyn FsProvider, path: impl AsRef<Path>) -> io::Result<Self> {
        let resolved = path.as_ref();
        if !provider.path_namespace().is_absolute(resolved) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "captured filesystem opens require an absolute path",
            ));
        }
        let flags = OpenFlags {
            read: true,
            ..OpenFlags::default()
        };
        provider.open(resolved, &flags).map(Self::from_handle)
    }

    pub async fn open_async(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        opts.open_async(path).await
    }

    /// Async counterpart of [`File::open_on`]; the borrowed owner is held
    /// through completion of the provider's open future.
    pub async fn open_async_on(
        provider: &dyn FsProvider,
        path: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let resolved = path.as_ref();
        if !provider.path_namespace().is_absolute(resolved) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "captured filesystem opens require an absolute path",
            ));
        }
        let flags = OpenFlags {
            read: true,
            ..OpenFlags::default()
        };
        provider
            .open_async(resolved, &flags)
            .await
            .map(Self::from_handle)
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

    pub async fn metadata_async(&self) -> io::Result<FsMetadata> {
        self.inner.metadata_async().await
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
    if let Some((provider, directory)) = invocation::snapshot() {
        if captured_read::check_selected(&provider).is_err() {
            return None;
        }
        return directory;
    }
    if let Some((_, directory)) = captured_read::snapshot() {
        return directory;
    }
    provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned")
        .current_dir_override
        .clone()
}

#[cfg(test)]
fn replace_current_dir_override(value: Option<PathBuf>) -> Option<PathBuf> {
    let mut guard = provider_state_lock()
        .write()
        .expect("filesystem provider lock poisoned");
    std::mem::replace(&mut guard.current_dir_override, value)
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
    if let Some((provider, _)) = invocation::snapshot() {
        return captured_read::selected_view(&provider)
            .unwrap_or_else(|_| Arc::new(UnavailableFsProvider));
    }
    if let Some((provider, _)) = captured_read::snapshot() {
        return provider;
    }
    provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned")
        .provider
        .clone()
}

/// Capture the installed provider and its effective working directory as one
/// invocation authority. A virtual CWD belongs to the same registry state as
/// the provider; native process CWD is copied after that state is released.
pub fn current_provider_with_current_dir() -> io::Result<(Arc<dyn FsProvider>, PathBuf)> {
    if let Some((provider, directory)) = invocation::snapshot() {
        let provider = captured_read::selected_view(&provider)?;
        return Ok((
            provider,
            directory.expect("selected filesystem has virtual CWD"),
        ));
    }
    if let Some((provider, directory)) = captured_read::snapshot() {
        return Ok((
            provider,
            directory.ok_or_else(captured_read::missing_directory)?,
        ));
    }
    let (provider, virtual_dir) = {
        let state = provider_state_lock()
            .read()
            .expect("filesystem provider lock poisoned");
        (state.provider.clone(), state.current_dir_override.clone())
    };
    let dir = match virtual_dir {
        Some(dir) => dir,
        None => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                std::env::current_dir()?
            }
            #[cfg(target_arch = "wasm32")]
            {
                PathBuf::from("/")
            }
        }
    };
    Ok((provider, dir))
}

pub fn current_provider_supports_read() -> bool {
    // Provider metadata is implementation code and may reenter the registry.
    // Clone its current owner while locked, then invoke outside the lock.
    current_provider().supports_read()
}

pub fn current_dir() -> io::Result<PathBuf> {
    if let Some((provider, _)) = invocation::snapshot() {
        captured_read::check_selected(&provider)?;
    }
    match invocation::selected() {
        Some(InvocationFilesystemSelection::Selected(owner)) => return owner.current_dir(),
        Some(InvocationFilesystemSelection::Denied {
            directory,
            environment_read,
        }) => {
            return directory
                .filter(|_| environment_read)
                .ok_or_else(invocation::denied_environment)
        }
        None => {}
    }
    if let Some((_, directory)) = captured_read::snapshot() {
        return directory.ok_or_else(captured_read::missing_directory);
    }
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
    futures::executor::block_on(set_current_dir_async(path.as_ref()))
}

pub async fn set_current_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    captured_read::require_directory_write()?;
    if let Some((provider, _)) = invocation::snapshot() {
        captured_read::check_selected(&provider)?;
    }
    match invocation::selected() {
        Some(InvocationFilesystemSelection::Selected(owner)) => {
            if !owner.environment_write_allowed() {
                return Err(invocation::denied_environment());
            }
            let mut target = PathBuf::from(path.as_ref());
            if !owner.provider().path_namespace().is_absolute(&target) {
                target = owner.captured_dir().join(target);
            }
            let canonical = owner
                .provider()
                .canonicalize(&target)
                .await
                .unwrap_or(target.clone());
            let metadata = owner.provider().metadata(&canonical).await?;
            if !metadata.is_dir() {
                return Err(io::Error::new(
                    ErrorKind::NotFound,
                    format!("Not a directory: {}", canonical.display()),
                ));
            }
            if !matches!(
                invocation::selected(),
                Some(InvocationFilesystemSelection::Selected(ref current))
                    if current.same_owner(&owner)
            ) {
                return Err(io::Error::new(
                    ErrorKind::Interrupted,
                    "filesystem invocation changed before working-directory publication",
                ));
            }
            owner.publish_current_dir(canonical)?;
            return Ok(());
        }
        Some(InvocationFilesystemSelection::Denied { .. }) => {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "selected invocation has no filesystem service",
            ));
        }
        None => {}
    }
    if captured_read::set_local_directory(path.as_ref()).await? {
        return Ok(());
    }
    let (provider, selected_dir) = {
        let state = provider_state_lock()
            .read()
            .expect("filesystem provider lock poisoned");
        (state.provider.clone(), state.current_dir_override.clone())
    };
    if let Some(base) = selected_dir {
        let mut target = PathBuf::from(path.as_ref());
        if !provider.path_namespace().is_absolute(&target) {
            target = base.join(target);
        }
        let canonical = provider
            .canonicalize(&target)
            .await
            .unwrap_or(target.clone());
        let metadata = provider.metadata(&canonical).await?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("Not a directory: {}", canonical.display()),
            ));
        }
        let mut state = provider_state_lock()
            .write()
            .expect("filesystem provider lock poisoned");
        if !Arc::ptr_eq(&state.provider, &provider) {
            return Err(io::Error::new(
                ErrorKind::Interrupted,
                "filesystem provider changed before working-directory publication",
            ));
        }
        state.current_dir_override = Some(canonical);
        Ok(())
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
    let (provider, resolved) = capture_provider_many(paths)?;
    provider.read_many(&resolved).await
}

pub async fn read_async(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.read(&resolved).await
}

pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { read_async(path).await })
}

pub async fn atomic_create_new_async(
    path: impl AsRef<Path>,
    bytes: impl AsRef<[u8]>,
) -> io::Result<AtomicCreateResult> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.atomic_create_new(&resolved, bytes.as_ref()).await
}

pub fn atomic_create_new(
    path: impl AsRef<Path>,
    bytes: impl AsRef<[u8]>,
) -> io::Result<AtomicCreateResult> {
    let path = path.as_ref().to_path_buf();
    let bytes = bytes.as_ref().to_vec();
    wait_for_fs(move || async move { atomic_create_new_async(path, bytes).await })
}

pub async fn read_to_string_async(path: impl AsRef<Path>) -> io::Result<String> {
    let bytes = read_async(path).await?;
    String::from_utf8(bytes).map_err(|err| io::Error::new(ErrorKind::InvalidData, err.utf8_error()))
}

pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { read_to_string_async(path).await })
}

pub async fn write_async(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.write(&resolved, data.as_ref()).await
}

pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
    let path = path.as_ref().to_path_buf();
    let data = data.as_ref().to_vec();
    wait_for_fs(move || async move { write_async(path, data).await })
}

pub async fn remove_file_async(path: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.remove_file(&resolved).await
}

pub fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { remove_file_async(path).await })
}

/// Remove through a retained provider without consulting process-global
/// provider state. Safe to call from a synchronous destructor inside an
/// already-running executor on native hosts: `wait_for_fs` owns the worker.
#[cfg(not(target_arch = "wasm32"))]
pub fn remove_file_in_provider(
    provider: Arc<dyn FsProvider>,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { provider.remove_file(&path).await })
}

pub async fn metadata_async(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.metadata(&resolved).await
}

pub fn metadata(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { metadata_async(path).await })
}

pub async fn symlink_metadata_async(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.symlink_metadata(&resolved).await
}

pub fn symlink_metadata(path: impl AsRef<Path>) -> io::Result<FsMetadata> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { symlink_metadata_async(path).await })
}

pub async fn read_dir_async(path: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.read_dir(&resolved).await
}

pub fn read_dir(path: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { read_dir_async(path).await })
}

pub async fn canonicalize_async(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.canonicalize(&resolved).await
}

pub async fn create_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.create_dir(&resolved).await
}

pub async fn create_dir_all_async(path: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.create_dir_all(&resolved).await
}

pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref().to_path_buf();
    wait_for_fs(move || async move { create_dir_all_async(path).await })
}

pub async fn remove_dir_async(path: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.remove_dir(&resolved).await
}

pub async fn remove_dir_all_async(path: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.remove_dir_all(&resolved).await
}

pub async fn rename_async(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved_from, resolved_to) = capture_provider_pair(from.as_ref(), to.as_ref())?;
    provider.rename(&resolved_from, &resolved_to).await
}

pub async fn rename_replace_async(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let (provider, resolved_from, resolved_to) = capture_provider_pair(from.as_ref(), to.as_ref())?;
    provider.rename_replace(&resolved_from, &resolved_to).await
}

pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let from = from.as_ref().to_path_buf();
    let to = to.as_ref().to_path_buf();
    wait_for_fs(move || async move { rename_async(from, to).await })
}

pub async fn set_readonly_async(path: impl AsRef<Path>, readonly: bool) -> io::Result<()> {
    let (provider, resolved) = capture_provider_path(path.as_ref())?;
    provider.set_readonly(&resolved, readonly).await
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
    let (provider, from, to) = capture_provider_pair(from.as_ref(), to.as_ref())?;
    let mut reader = OpenOptions::new()
        .read(true)
        .open_on(provider.as_ref(), &from)?;
    let mut writer = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open_on(provider.as_ref(), &to)?;
    io::copy(&mut reader, &mut writer)
}

/// Copy between absolute paths through one retained provider. Unlike
/// `copy_file`, this never consults the process-wide provider or CWD after
/// capture. It preserves the existing synchronous stream-copy semantics.
pub fn copy_file_on(provider: &dyn FsProvider, from: &Path, to: &Path) -> io::Result<u64> {
    if !provider.path_namespace().is_absolute(from) || !provider.path_namespace().is_absolute(to) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "copy_file_on requires captured absolute paths",
        ));
    }
    let mut reader = File::from_handle(provider.open(
        from,
        &OpenFlags {
            read: true,
            ..OpenFlags::default()
        },
    )?);
    let mut writer = File::from_handle(provider.open(
        to,
        &OpenFlags {
            write: true,
            create: true,
            truncate: true,
            ..OpenFlags::default()
        },
    )?);
    io::copy(&mut reader, &mut writer)
}

#[path = "copy_stream.rs"]
mod copy_stream;
pub use copy_stream::{copy_file_on_async, FsCopyCheckpoint, FsCopyError};

#[cfg(test)]
#[path = "copy_tests.rs"]
mod copy_tests;

#[cfg(not(target_arch = "wasm32"))]
fn wait_for_fs<T, F, Fut>(factory: F) -> io::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = io::Result<T>> + 'static,
{
    std::thread::spawn(move || futures::executor::block_on(factory()))
        .join()
        .map_err(|_| io::Error::other("filesystem worker thread panicked"))?
}

#[cfg(target_arch = "wasm32")]
fn wait_for_fs<T, F, Fut>(factory: F) -> io::Result<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = io::Result<T>>,
{
    futures::executor::block_on(factory())
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
    use std::collections::{HashMap, HashSet};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        provider_override_lock()
    }

    fn comparable_path(path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        #[cfg(windows)]
        {
            let text = canonical.to_string_lossy();
            if let Some(stripped) = text.strip_prefix(r"\\?\UNC\") {
                return PathBuf::from(format!(r"\\{stripped}"));
            }
            if let Some(stripped) = text.strip_prefix(r"\\?\") {
                return PathBuf::from(stripped);
            }
            PathBuf::from(text.as_ref())
        }
        #[cfg(not(windows))]
        {
            canonical
        }
    }

    fn assert_same_path(left: impl AsRef<Path>, right: impl AsRef<Path>) {
        assert_eq!(comparable_path(left), comparable_path(right));
    }

    struct UnsupportedProvider;

    #[test]
    fn unqualified_provider_cannot_claim_a_staged_replacement() {
        assert!(UnsupportedProvider
            .begin_staged_replace(Path::new("/report.csv"))
            .expect("capability query")
            .is_none());
    }

    #[test]
    fn retained_stage_uses_issuing_provider_after_ambient_swap() {
        let _lock = test_lock();
        let original = Arc::new(MemoryFsProvider::with_current_dir("/reports"));
        original
            .write_project_path("/reports/data.csv", b"original")
            .expect("original file");
        let replacement = Arc::new(MemoryFsProvider::with_current_dir("/reports"));
        replacement
            .write_project_path("/reports/data.csv", b"unrelated")
            .expect("replacement file");
        let _original = replace_provider(original.clone());
        let mut stage = original
            .begin_staged_replace(Path::new("/reports/data.csv"))
            .expect("stage acquisition")
            .expect("memory stage");
        let _replacement = replace_provider(replacement.clone());
        assert_eq!(
            futures::executor::block_on(stage.write_chunk(b"published")).unwrap(),
            9
        );
        futures::executor::block_on(stage.publish()).expect("publish on original provider");
        assert_eq!(
            original.read_project_path("/reports/data.csv").unwrap(),
            b"published"
        );
        assert_eq!(
            replacement.read_project_path("/reports/data.csv").unwrap(),
            b"unrelated"
        );
    }

    #[test]
    fn read_capability_metadata_is_queried_outside_the_provider_lock() {
        let _guard = test_lock();
        let _provider = replace_provider(Arc::new(UnsupportedProvider));
        assert!(!current_provider_supports_read());
    }

    #[test]
    fn unimplemented_replacement_fails_before_any_provider_operation() {
        let result = futures::executor::block_on(
            UnsupportedProvider.rename_replace(Path::new("/source"), Path::new("/target")),
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Unsupported);
    }

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
        fn supports_read(&self) -> bool {
            // A metadata callback may inspect the provider registry. A write
            // lock must be available here, without recursively blocking.
            assert!(provider_state_lock().try_write().is_ok());
            false
        }

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
        fn path_namespace(&self) -> FsPathNamespace {
            FsPathNamespace::VirtualRooted
        }

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
        let _guard = test_lock();
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
        let _guard = test_lock();
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
    fn sync_helpers_work_inside_async_executor() {
        let _guard = test_lock();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("file.txt");
        let parent = path.parent().unwrap().to_path_buf();

        futures::executor::block_on(async {
            create_dir_all(&parent).expect("create dir");
            write(&path, b"hello").expect("write");
            assert_eq!(read(&path).expect("read"), b"hello");
            assert_eq!(read_to_string(&path).expect("read string"), "hello");
            assert!(metadata(&path).expect("metadata").is_file());
            assert_eq!(read_dir(&parent).expect("read dir").len(), 1);
            remove_file(&path).expect("remove");
        });
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn explicit_provider_cleanup_inside_executor_does_not_switch_ambient_provider() {
        let _guard = test_lock();
        let ambient = current_provider();
        let provider = Arc::new(MemoryFsProvider::new());
        provider.write_project_path("/partial", b"discard").unwrap();
        futures::executor::block_on(async {
            remove_file_in_provider(provider.clone(), "/partial").expect("nested cleanup");
        });
        assert_eq!(
            provider.read_project_path("/partial").unwrap_err().kind(),
            ErrorKind::NotFound
        );
        assert!(Arc::ptr_eq(&ambient, &current_provider()));
    }

    #[test]
    fn replace_provider_restores_previous() {
        let _guard = test_lock();
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
        let _guard = test_lock();
        let temp = tempdir().expect("tempdir");
        let previous = std::env::current_dir().expect("current dir");
        let _cwd_guard = ProcessCwdGuard { previous };
        std::env::set_current_dir(temp.path()).expect("set temp cwd");

        let _provider_guard = replace_provider(Arc::new(NativeFsProvider));
        let current = current_dir().expect("vfs current dir");
        let expected = std::fs::canonicalize(temp.path()).expect("canonical temp");
        assert_same_path(&current, &expected);

        futures::executor::block_on(write_async("native-relative.txt", b"native"))
            .expect("write relative path");
        assert_eq!(
            std::fs::read_to_string(temp.path().join("native-relative.txt")).expect("read file"),
            "native"
        );

        std::fs::create_dir(temp.path().join("child")).expect("create child");
        set_current_dir("child").expect("set child cwd");
        assert_same_path(
            std::env::current_dir().expect("process cwd"),
            expected.join("child"),
        );
    }

    #[test]
    fn set_provider_initializes_virtual_cwd_from_provider_default() {
        let _guard = test_lock();
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
        let _guard = test_lock();
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
        let _guard = test_lock();
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
    fn virtual_root_paths_do_not_resolve_relative_to_virtual_cwd() {
        let _guard = test_lock();
        let provider = Arc::new(VirtualFsProvider::new("/", &["/workspace"]));
        let _provider_guard = replace_provider(provider.clone());
        set_current_dir("/workspace").expect("set virtual cwd");

        futures::executor::block_on(write_async("/root.txt", b"root")).expect("write absolute");

        assert_eq!(
            provider.file_bytes("/root.txt").as_deref(),
            Some(&b"root"[..])
        );
        assert_eq!(provider.file_bytes("/workspace/root.txt"), None);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn native_provider_replacement_clears_virtual_cwd_override() {
        let _guard = test_lock();
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
            assert_same_path(current_dir().expect("native cwd"), &expected);

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
        let _guard = test_lock();
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
    fn dialog_requests_resolve_default_paths_through_the_selected_storage_provider() {
        let _guard = test_lock();
        let provider = Arc::new(VirtualFsProvider::new("/workspace", &["/workspace/data"]));
        let _provider_guard = replace_provider(provider);
        let open = OpenFileDialogRequest {
            title: Some("Open".to_string()),
            default_path: Some(PathBuf::from("data")),
            filters: vec![OpenFileDialogFilter {
                patterns: vec!["*.csv".to_string()],
                description: Some("CSV files".to_string()),
            }],
            multiselect: false,
        };
        let save = SaveFileDialogRequest {
            title: Some("Save".to_string()),
            default_path: Some(PathBuf::from("data/results.mat")),
            filters: Vec::new(),
        };
        let directory = DirectoryDialogRequest {
            title: Some("Directory".to_string()),
            default_path: Some(PathBuf::from("data")),
        };

        assert_eq!(
            resolve_open_file_dialog_request(&open)
                .unwrap()
                .default_path,
            Some(PathBuf::from("/workspace/data"))
        );
        assert_eq!(
            resolve_save_file_dialog_request(&save)
                .unwrap()
                .default_path,
            Some(PathBuf::from("/workspace/data/results.mat"))
        );
        assert_eq!(
            resolve_directory_dialog_request(&directory)
                .unwrap()
                .default_path,
            Some(PathBuf::from("/workspace/data"))
        );
    }

    #[test]
    fn with_provider_restores_even_on_panic() {
        let _guard = test_lock();
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

    #[test]
    fn atomic_publication_uses_only_the_installed_provider() {
        let _guard = test_lock();
        // A relative name is valid in both namespaces; an OS drive prefix is
        // deliberately not a portable memory-filesystem path on Windows.
        let host = tempfile::NamedTempFile::new_in(std::env::current_dir().unwrap()).unwrap();
        let logical_path = Path::new(host.path().file_name().unwrap());
        std::fs::write(host.path(), b"host").unwrap();
        let memory = Arc::new(MemoryFsProvider::new());
        let provider: Arc<dyn FsProvider> = memory.clone();
        let _provider_guard = replace_provider(provider);

        let outcome = atomic_create_new(logical_path, b"provider").unwrap();

        assert!(matches!(outcome, AtomicCreateResult::Published { .. }));
        assert_eq!(memory.read_project_path(logical_path).unwrap(), b"provider");
        assert_eq!(std::fs::read(host.path()).unwrap(), b"host");
    }

    #[test]
    fn unsupported_provider_does_not_emulate_atomic_publication() {
        let error = futures::executor::block_on(
            UnsupportedProvider.atomic_create_new(Path::new("/artifact"), b"payload"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
