use once_cell::sync::Lazy;
use runmat_filesystem::File;
use std::collections::HashMap;
use std::io::{self, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Mutex as StdMutex};

pub(crate) type SharedFileHandle = Arc<StdMutex<Option<File>>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FileOperationError {
    InvalidIdentifier,
    Busy,
    Poisoned,
}

impl FileOperationError {
    pub(crate) fn into_io_error(self) -> io::Error {
        match self {
            Self::InvalidIdentifier => {
                io::Error::new(io::ErrorKind::NotFound, "file identifier is not registered")
            }
            Self::Busy => io::Error::new(
                io::ErrorKind::WouldBlock,
                "file identifier is in another operation",
            ),
            Self::Poisoned => io::Error::other("registered file handle is poisoned"),
        }
    }
}

/// Exact registered file checked out for one operation. No registry or handle
/// mutex is held during filesystem callbacks, suspension, or CPU parsing.
pub(crate) struct FileOperation {
    fid: i32,
    handle: SharedFileHandle,
    file: Option<File>,
    rollback_cursor_to: Option<u64>,
    pub(crate) info: FileInfo,
}

impl FileOperation {
    pub(crate) fn file_mut(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("a live operation retains its file")
    }

    pub(crate) fn restore_cursor_on_drop(&mut self, position: u64) {
        self.rollback_cursor_to = Some(position);
    }

    pub(crate) fn commit_cursor(&mut self) {
        self.rollback_cursor_to = None;
    }
}

impl Drop for FileOperation {
    fn drop(&mut self) {
        if let (Some(position), Some(file)) = (self.rollback_cursor_to, self.file.as_mut()) {
            let _ = file.seek(SeekFrom::Start(position));
        }
        let registry = REGISTRY.lock().unwrap_or_else(|poison| poison.into_inner());
        let is_current = registry
            .entries
            .get(&self.fid)
            .and_then(Entry::file_handle)
            .is_some_and(|handle| Arc::ptr_eq(&handle, &self.handle));
        if is_current {
            let mut slot = self
                .handle
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if slot.is_none() {
                *slot = self.file.take();
            }
        }
    }
}

pub(crate) fn acquire_file(fid: i32) -> Result<FileOperation, FileOperationError> {
    let registry = REGISTRY.lock().map_err(|_| FileOperationError::Poisoned)?;
    let entry = registry
        .entries
        .get(&fid)
        .ok_or(FileOperationError::InvalidIdentifier)?;
    let handle = entry
        .file_handle()
        .ok_or(FileOperationError::InvalidIdentifier)?;
    let mut slot = handle.try_lock().map_err(|error| match error {
        std::sync::TryLockError::WouldBlock => FileOperationError::Busy,
        std::sync::TryLockError::Poisoned(_) => FileOperationError::Poisoned,
    })?;
    let file = slot.take().ok_or(FileOperationError::Busy)?;
    let info = entry.info();
    drop(slot);
    drop(registry);
    Ok(FileOperation {
        fid,
        handle,
        file: Some(file),
        rollback_cursor_to: None,
        info,
    })
}

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

#[derive(Clone)]
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
    fn into_entry(self, id: i32) -> Entry {
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
        }
    }
}

struct FileRegistry {
    next_id: i32,
    entries: HashMap<i32, Entry>,
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

pub(crate) fn register_file(file: RegisteredFile) -> i32 {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let id = guard.allocate_id();
    let entry = file.into_entry(id);
    guard.entries.insert(id, entry);
    id
}

pub(crate) fn info_for(fid: i32) -> Option<FileInfo> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.get(&fid).map(|entry| entry.info())
}

pub(crate) fn list_infos() -> Vec<FileInfo> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    let mut infos: Vec<FileInfo> = guard.entries.values().map(|entry| entry.info()).collect();
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
pub(crate) struct RegisteredFileSnapshot {
    pub info: FileInfo,
    pub handle: SharedFileHandle,
}

pub(crate) fn snapshot_file(fid: i32) -> Option<RegisteredFileSnapshot> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    let entry = guard.entries.get(&fid)?;
    Some(RegisteredFileSnapshot {
        info: entry.info(),
        handle: entry.file_handle()?,
    })
}

pub(crate) fn eof_encountered(fid: i32) -> Option<bool> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.get(&fid).map(|entry| entry.eof_encountered)
}

pub(crate) fn mark_eof_encountered(fid: i32) -> bool {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let Some(entry) = guard.entries.get_mut(&fid) else {
        return false;
    };
    entry.eof_encountered = true;
    true
}

pub(crate) fn clear_eof_encountered(fid: i32) -> bool {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    let Some(entry) = guard.entries.get_mut(&fid) else {
        return false;
    };
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

pub(crate) async fn close_async(fid: i32) -> std::io::Result<Option<FileInfo>> {
    let entry = {
        if fid < 3 {
            return Ok(None);
        }
        let mut guard = REGISTRY.lock().expect("file registry poisoned");
        let Some(registered) = guard.entries.get(&fid) else {
            return Ok(None);
        };
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

    if let Some(file) = entry.file.as_mut() {
        file.flush_async().await?;
    }
    entry.committed = true;
    Ok(Some(entry.info.clone()))
}

#[cfg(test)]
pub(crate) fn reset_for_tests() {
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.retain(|&id, _| id < 3);
    guard.next_id = 3;
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_REGISTRY_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}
