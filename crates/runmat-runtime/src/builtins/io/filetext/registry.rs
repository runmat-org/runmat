use once_cell::sync::Lazy;
use runmat_filesystem::File;
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Mutex as StdMutex};

pub(crate) type SharedFileHandle = Arc<StdMutex<Option<File>>>;

mod operation;
#[cfg(test)]
pub(crate) use operation::acquire_file;
pub(crate) use operation::{acquire_file_in, FileOperation, FileOperationError};
mod retirement;
use retirement::retire_session;
mod authority;
use crate::context::{RuntimeContext, RuntimeContextLocalId};
pub(crate) use authority::{FileFilesystem, FileIssuer};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FileAccess {
    pub read: bool,
    pub write: bool,
}

impl FileAccess {
    pub const READ: Self = Self {
        read: true,
        write: false,
    };
    pub const WRITE: Self = Self {
        read: false,
        write: true,
    };
}

#[derive(Clone)]
pub(crate) enum StandardStream {
    Stdin,
    Stdout,
    Stderr,
}

#[derive(Clone)]
enum Resource {
    Standard,
    File(SharedFileHandle),
}

struct CloseEntry {
    fid: i32,
    info: FileInfo,
    entry: Option<Entry>,
    file: Option<File>,
    committed: bool,
}

/// Owns the checked-out identifier until the invocation has completed its
/// cancellation checkpoint. Dropping this value restores the same handle.
pub(crate) struct PreparedClose(CloseEntry);

impl PreparedClose {
    pub(crate) fn commit(mut self, runtime: Option<&RuntimeContext>) -> io::Result<FileInfo> {
        if let Some(registered) = self.0.entry.as_ref() {
            if let Some(filesystem) = &registered.filesystem {
                filesystem.check()?;
            }
            if !registered.issuer.admits(runtime) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "file identifier session authority expired",
                ));
            }
        }
        self.0.committed = true;
        Ok(self.0.info.clone())
    }
}

impl Drop for CloseEntry {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Dropping a pending flush future must restore both the identifier
        // and its exact handle. Keep the registry -> handle lock order used
        // by close_async; no lock is held across the flush await.
        let mut registry = REGISTRY.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(entry) = self.entry.take() {
            if !entry.issuer.live() {
                drop(registry);
                drop(entry);
                return;
            }
            if let Some(handle) = entry.file_handle() {
                let mut slot = handle.lock().unwrap_or_else(|poison| poison.into_inner());
                if slot.is_none() {
                    *slot = self.file.take();
                }
            }
            registry.entries.insert(self.fid, entry);
        }
    }
}

struct Entry {
    id: i32,
    name: String,
    path: Option<PathBuf>,
    permission: String,
    access: FileAccess,
    machinefmt: String,
    encoding: String,
    resource: Resource,
    eof_encountered: bool,
    text_input: Option<super::encoding::TextInputBox>,
    restore_failure: Option<Arc<io::Error>>,
    issuer: FileIssuer,
    filesystem: Option<Arc<FileFilesystem>>,
    retired: bool,
}

impl Entry {
    fn standard(id: i32, stream: StandardStream) -> Self {
        let name = match stream {
            StandardStream::Stdin => "stdin".to_string(),
            StandardStream::Stdout => "stdout".to_string(),
            StandardStream::Stderr => "stderr".to_string(),
        };
        Self {
            id,
            name,
            path: None,
            permission: match stream {
                StandardStream::Stdin => "r".to_string(),
                StandardStream::Stdout | StandardStream::Stderr => "w".to_string(),
            },
            access: match stream {
                StandardStream::Stdin => FileAccess::READ,
                StandardStream::Stdout | StandardStream::Stderr => FileAccess::WRITE,
            },
            machinefmt: "native".to_string(),
            encoding: "UTF-8".to_string(),
            resource: Resource::Standard,
            eof_encountered: false,
            text_input: None,
            restore_failure: None,
            issuer: FileIssuer::Process,
            filesystem: None,
            retired: false,
        }
    }

    fn info(&self) -> FileInfo {
        FileInfo {
            id: self.id,
            name: self.name.clone(),
            path: self.path.clone(),
            permission: self.permission.clone(),
            access: self.access,
            machinefmt: self.machinefmt.clone(),
            encoding: self.encoding.clone(),
        }
    }

    fn file_handle(&self) -> Option<SharedFileHandle> {
        match &self.resource {
            Resource::File(handle) => Some(handle.clone()),
            Resource::Standard => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FileInfo {
    pub id: i32,
    pub name: String,
    pub path: Option<PathBuf>,
    pub permission: String,
    pub access: FileAccess,
    pub machinefmt: String,
    pub encoding: String,
}

pub(crate) struct RegisteredFile {
    pub path: PathBuf,
    pub permission: String,
    pub access: FileAccess,
    pub machinefmt: String,
    pub encoding: String,
    pub handle: SharedFileHandle,
}

impl RegisteredFile {
    fn into_entry(
        self,
        id: i32,
        issuer: FileIssuer,
        filesystem: Option<Arc<FileFilesystem>>,
    ) -> Entry {
        let display_name = self.path.to_string_lossy().to_string();
        Entry {
            id,
            name: display_name,
            path: Some(self.path),
            permission: self.permission,
            access: self.access,
            machinefmt: self.machinefmt,
            encoding: self.encoding,
            resource: Resource::File(self.handle),
            eof_encountered: false,
            text_input: None,
            restore_failure: None,
            issuer,
            filesystem,
            retired: false,
        }
    }
}

struct FileRegistry {
    next_id: i32,
    entries: HashMap<i32, Entry>,
    cleanup_registered: HashSet<RuntimeContextLocalId>,
}

impl FileRegistry {
    fn new() -> Self {
        let mut entries = HashMap::new();
        entries.insert(0, Entry::standard(0, StandardStream::Stdin));
        entries.insert(1, Entry::standard(1, StandardStream::Stdout));
        entries.insert(2, Entry::standard(2, StandardStream::Stderr));
        Self {
            next_id: 3,
            entries,
            cleanup_registered: HashSet::new(),
        }
    }

    fn allocate_id(&mut self) -> i32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

static REGISTRY: Lazy<Mutex<FileRegistry>> = Lazy::new(|| Mutex::new(FileRegistry::new()));

#[cfg(test)]
static TEST_REGISTRY_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(test)]
pub(crate) fn register_file(file: RegisteredFile) -> i32 {
    register_file_in(file, None, None)
}

pub(crate) fn register_file_in(
    file: RegisteredFile,
    runtime: Option<&RuntimeContext>,
    filesystem: Option<Arc<FileFilesystem>>,
) -> i32 {
    let issuer = FileIssuer::from_runtime(runtime);
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let id = guard.allocate_id();
    let entry = file.into_entry(id, issuer, filesystem);
    guard.entries.insert(id, entry);
    let register_cleanup =
        runtime.is_some_and(|runtime| guard.cleanup_registered.insert(runtime.local_identity()));
    drop(guard);
    if register_cleanup {
        let runtime = runtime.expect("session cleanup has a runtime");
        let identity = runtime.local_identity();
        runtime.register_session_resource_cleanup(move || retire_session(identity));
    }
    id
}

#[cfg(test)]
pub(crate) fn info_for(fid: i32) -> Option<FileInfo> {
    info_for_in(fid, None)
}

pub(crate) fn info_for_in(fid: i32, runtime: Option<&RuntimeContext>) -> Option<FileInfo> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard
        .entries
        .get(&fid)
        .filter(|entry| entry_visible(entry, runtime))
        .map(|entry| entry.info())
}

#[cfg(test)]
pub(crate) fn list_infos() -> Vec<FileInfo> {
    list_infos_in(None)
}

pub(crate) fn list_infos_in(runtime: Option<&RuntimeContext>) -> Vec<FileInfo> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    let mut infos: Vec<FileInfo> = guard
        .entries
        .values()
        .filter(|entry| entry_visible(entry, runtime))
        .map(|entry| entry.info())
        .collect();
    infos.sort_by_key(|info| info.id);
    infos
}

#[cfg(test)]
pub(crate) fn shared_handle(fid: i32) -> Option<SharedFileHandle> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard
        .entries
        .get(&fid)
        .and_then(|entry| entry.file_handle())
}

/// Metadata and handle from one registered entry. The handle's slot remains
/// authoritative for close state after the registry lock has been released.
#[cfg(test)]
pub(crate) struct RegisteredFileSnapshot {
    pub info: FileInfo,
    pub handle: SharedFileHandle,
}

#[cfg(test)]
pub(crate) fn snapshot_file(fid: i32) -> Option<RegisteredFileSnapshot> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    let entry = guard.entries.get(&fid)?;
    Some(RegisteredFileSnapshot {
        info: entry.info(),
        handle: entry.file_handle()?,
    })
}

pub(crate) fn eof_encountered_in(fid: i32, runtime: Option<&RuntimeContext>) -> Option<bool> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard
        .entries
        .get(&fid)
        .filter(|entry| entry_visible(entry, runtime))
        .map(|entry| entry.eof_encountered)
}

pub(crate) fn mark_eof_encountered_in(fid: i32, runtime: Option<&RuntimeContext>) -> bool {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let Some(entry) = guard.entries.get_mut(&fid) else {
        return false;
    };
    if !entry_visible(entry, runtime) {
        return false;
    }
    entry.eof_encountered = true;
    true
}

pub(crate) fn clear_eof_encountered_in(fid: i32, runtime: Option<&RuntimeContext>) -> bool {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let Some(entry) = guard.entries.get_mut(&fid) else {
        return false;
    };
    if !entry_visible(entry, runtime) {
        return false;
    }
    entry.eof_encountered = false;
    true
}

#[cfg(test)]
pub(crate) fn close(fid: i32) -> Option<FileInfo> {
    if fid < 3 {
        return None;
    }
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.remove(&fid).map(|entry| entry.info())
}

#[cfg(test)]
pub(crate) async fn close_async(fid: i32) -> std::io::Result<Option<FileInfo>> {
    close_async_in(fid, None).await
}

#[cfg(test)]
pub(crate) async fn close_async_in(
    fid: i32,
    runtime: Option<&RuntimeContext>,
) -> std::io::Result<Option<FileInfo>> {
    prepare_close_async_in(fid, runtime)
        .await?
        .map(|prepared| prepared.commit(runtime))
        .transpose()
}

pub(crate) async fn prepare_close_async_in(
    fid: i32,
    runtime: Option<&RuntimeContext>,
) -> std::io::Result<Option<PreparedClose>> {
    let entry = {
        if fid < 3 {
            return Ok(None);
        }
        let mut guard = REGISTRY.lock().expect("file registry poisoned");
        let Some(registered) = guard.entries.get(&fid) else {
            return Ok(None);
        };
        if !entry_visible(registered, runtime) {
            return Ok(None);
        }
        let Some(handle) = registered.file_handle() else {
            return Ok(None);
        };
        let mut slot = handle.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => io::Error::new(
                io::ErrorKind::WouldBlock,
                "file identifier is in another operation",
            ),
            std::sync::TryLockError::Poisoned(_) => {
                io::Error::other("registered file handle is poisoned")
            }
        })?;
        if slot.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "file identifier is in another operation",
            ));
        }
        let file = slot.take();
        drop(slot);
        guard.entries.remove(&fid).map(|entry| CloseEntry {
            fid,
            info: entry.info(),
            entry: Some(entry),
            file,
            committed: false,
        })
    };

    let Some(mut entry) = entry else {
        return Ok(None);
    };

    if let Some(filesystem) = entry
        .entry
        .as_ref()
        .and_then(|entry| entry.filesystem.as_ref())
    {
        filesystem.check()?;
    }

    if let Some(file) = entry.file.as_mut() {
        file.flush_async().await?;
    }
    if let Some(registered) = entry.entry.as_ref() {
        if let Some(filesystem) = &registered.filesystem {
            filesystem.check()?;
        }
        if !registered.issuer.admits(runtime) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "file identifier session authority expired",
            ));
        }
    }
    Ok(Some(PreparedClose(entry)))
}

fn entry_visible(entry: &Entry, runtime: Option<&RuntimeContext>) -> bool {
    !entry.retired && (entry.id < 3 || entry.issuer.admits(runtime))
}

#[cfg(test)]
pub(crate) fn reset_for_tests() {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.retain(|&id, _| id < 3);
    guard.cleanup_registered.clear();
    guard.next_id = 3;
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type RegistryTestGuard = std::sync::MutexGuard<'static, ()>;

#[cfg(all(test, target_arch = "wasm32"))]
// The locked wasm-bindgen-test runner schedules one test at a time. Native
// tests retain their original locking below. Do not acquire the filesystem
// override lock here: provider-specific fixtures acquire it after this guard.
// A concurrent browser harness must replace this ambient fixture strategy.
pub(crate) struct RegistryTestGuard {
    _filesystem: crate::builtins::common::test_support::filesystem::TestDirectory,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
pub(crate) fn test_guard() -> RegistryTestGuard {
    let lock = TEST_REGISTRY_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    #[cfg(not(target_arch = "wasm32"))]
    {
        lock
    }
    #[cfg(target_arch = "wasm32")]
    {
        RegistryTestGuard {
            _filesystem: crate::builtins::common::test_support::filesystem::tempdir()
                .expect("isolated browser file registry fixture"),
            _lock: lock,
        }
    }
}

#[cfg(test)]
#[path = "registry/await_tests.rs"]
mod await_tests;
#[cfg(test)]
#[path = "registry/session_tests.rs"]
mod session_tests;
