use once_cell::sync::Lazy;
use runmat_filesystem::File;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Mutex as StdMutex};

#[derive(Clone)]
pub(crate) enum StandardStream {
    Stdin,
    Stdout,
    Stderr,
}

#[derive(Clone)]
enum Resource {
    Standard,
    File(Arc<StdMutex<File>>),
}

struct Entry {
    id: i32,
    name: String,
    path: Option<PathBuf>,
    permission: String,
    machinefmt: String,
    encoding: String,
    resource: Resource,
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
            machinefmt: "native".to_string(),
            encoding: "UTF-8".to_string(),
            resource: Resource::Standard,
        }
    }

    fn info(&self) -> FileInfo {
        FileInfo {
            id: self.id,
            name: self.name.clone(),
            path: self.path.clone(),
            permission: self.permission.clone(),
            machinefmt: self.machinefmt.clone(),
            encoding: self.encoding.clone(),
        }
    }

    fn file_handle(&self) -> Option<Arc<StdMutex<File>>> {
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
    pub machinefmt: String,
    pub encoding: String,
}

pub(crate) struct RegisteredFile {
    pub path: PathBuf,
    pub permission: String,
    pub machinefmt: String,
    pub encoding: String,
    pub handle: Arc<StdMutex<File>>,
}

impl RegisteredFile {
    fn into_entry(self, id: i32) -> Entry {
        let display_name = self.path.to_string_lossy().to_string();
        Entry {
            id,
            name: display_name,
            path: Some(self.path),
            permission: self.permission,
            machinefmt: self.machinefmt,
            encoding: self.encoding,
            resource: Resource::File(self.handle),
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

#[allow(dead_code)]
pub(crate) fn take_handle(fid: i32) -> Option<Arc<StdMutex<File>>> {
    let guard = REGISTRY.lock().expect("file registry poisoned");
    guard
        .entries
        .get(&fid)
        .and_then(|entry| entry.file_handle())
}

#[allow(dead_code)]
pub(crate) fn close(fid: i32) -> Option<FileInfo> {
    if fid < 3 {
        return None;
    }
    let mut guard = REGISTRY.lock().expect("file registry poisoned");
    guard.entries.remove(&fid).map(|entry| entry.info())
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
