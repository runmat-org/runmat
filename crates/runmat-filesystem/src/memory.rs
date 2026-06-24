use crate::{DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use async_trait::async_trait;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::{self, Cursor, ErrorKind, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
pub struct MemoryFsProvider {
    default_current_dir: PathBuf,
    inner: Arc<Mutex<MemoryTree>>,
}

#[derive(Clone, Debug)]
enum MemoryEntry {
    Directory,
    File { bytes: Vec<u8>, readonly: bool },
}

#[derive(Debug)]
struct MemoryTree {
    entries: BTreeMap<PathBuf, MemoryEntry>,
}

impl Default for MemoryFsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryFsProvider {
    pub fn new() -> Self {
        Self::with_current_dir(PathBuf::from("/"))
    }

    pub fn with_current_dir(default_current_dir: impl Into<PathBuf>) -> Self {
        let default_current_dir =
            normalize_path(&default_current_dir.into()).unwrap_or_else(|_| PathBuf::from("/"));
        let mut entries = BTreeMap::new();
        entries.insert(PathBuf::from("/"), MemoryEntry::Directory);
        let provider = Self {
            default_current_dir,
            inner: Arc::new(Mutex::new(MemoryTree { entries })),
        };
        let _ = provider.create_dir_all_sync(&provider.default_current_dir);
        provider
    }

    pub fn reset(&self) {
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        guard.entries.clear();
        guard
            .entries
            .insert(PathBuf::from("/"), MemoryEntry::Directory);
        ensure_dirs(&mut guard, &self.default_current_dir).expect("default cwd must be valid");
    }

    pub fn read_project_path(&self, path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        let path = normalize_path(path.as_ref())?;
        let guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::File { bytes, .. }) => Ok(bytes.clone()),
            _ => Err(not_found(&path)),
        }
    }

    pub fn write_project_path(&self, path: impl AsRef<Path>, data: &[u8]) -> io::Result<()> {
        let path = normalize_path(path.as_ref())?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::Directory) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("not a file: {}", path.display()),
                ));
            }
            Some(MemoryEntry::File { readonly: true, .. }) => {
                return Err(io::Error::new(
                    ErrorKind::PermissionDenied,
                    format!("file is readonly: {}", path.display()),
                ));
            }
            Some(MemoryEntry::File { .. }) | None => {}
        }
        ensure_parent_dirs(&mut guard, &path)?;
        guard.entries.insert(
            path,
            MemoryEntry::File {
                bytes: data.to_vec(),
                readonly: false,
            },
        );
        Ok(())
    }

    pub fn list_project_path(&self, path: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
        let path = normalize_path(path.as_ref())?;
        let guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::Directory) => {}
            Some(MemoryEntry::File { .. }) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("not a directory: {}", path.display()),
                ));
            }
            None => return Err(not_found(&path)),
        }
        let mut entries = Vec::new();
        for (candidate, entry) in guard.entries.iter() {
            if candidate == &path || !is_direct_child(&path, candidate) {
                continue;
            }
            entries.push(DirEntry::new(
                candidate.clone(),
                candidate
                    .file_name()
                    .map(OsString::from)
                    .unwrap_or_else(|| OsString::from("")),
                entry_type(entry),
            ));
        }
        Ok(entries)
    }

    pub fn metadata_project_path(&self, path: impl AsRef<Path>) -> io::Result<FsMetadata> {
        let path = normalize_path(path.as_ref())?;
        let guard = self.inner.lock().expect("memory filesystem lock poisoned");
        metadata_for(&path, guard.entries.get(&path))
    }

    pub fn create_dir_project_path(
        &self,
        path: impl AsRef<Path>,
        recursive: bool,
    ) -> io::Result<()> {
        let path = normalize_path(path.as_ref())?;
        if recursive {
            self.create_dir_all_sync(&path)
        } else {
            let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
            ensure_parent_dir(&guard, &path)?;
            match guard.entries.get(&path) {
                Some(MemoryEntry::Directory) | Some(MemoryEntry::File { .. }) => {
                    Err(io::Error::new(
                        ErrorKind::AlreadyExists,
                        format!("entry already exists: {}", path.display()),
                    ))
                }
                None => {
                    guard.entries.insert(path, MemoryEntry::Directory);
                    Ok(())
                }
            }
        }
    }

    pub fn remove_project_path(&self, path: impl AsRef<Path>, recursive: bool) -> io::Result<()> {
        let path = normalize_path(path.as_ref())?;
        if path == Path::new("/") {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "cannot remove memory filesystem root",
            ));
        }
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::File { .. }) => {
                guard.entries.remove(&path);
            }
            Some(MemoryEntry::Directory) => {
                let children = guard
                    .entries
                    .keys()
                    .filter(|candidate| is_descendant(&path, candidate))
                    .cloned()
                    .collect::<Vec<_>>();
                if !children.is_empty() && !recursive {
                    return Err(io::Error::new(
                        ErrorKind::DirectoryNotEmpty,
                        format!("directory is not empty: {}", path.display()),
                    ));
                }
                for child in children {
                    guard.entries.remove(&child);
                }
                guard.entries.remove(&path);
            }
            None => return Err(not_found(&path)),
        }
        Ok(())
    }

    pub fn rename_project_path(
        &self,
        from: impl AsRef<Path>,
        to: impl AsRef<Path>,
    ) -> io::Result<()> {
        let from = normalize_path(from.as_ref())?;
        let to = normalize_path(to.as_ref())?;
        if from == Path::new("/") {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "cannot rename memory filesystem root",
            ));
        }
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        let entry = guard
            .entries
            .get(&from)
            .cloned()
            .ok_or_else(|| not_found(&from))?;
        ensure_parent_dirs(&mut guard, &to)?;
        if matches!(entry, MemoryEntry::Directory) {
            let descendants = guard
                .entries
                .iter()
                .filter(|(candidate, _)| is_descendant(&from, candidate))
                .map(|(candidate, entry)| (candidate.clone(), entry.clone()))
                .collect::<Vec<_>>();
            guard.entries.insert(to.clone(), entry);
            for (candidate, child) in descendants.iter() {
                let suffix = candidate.strip_prefix(&from).unwrap_or(Path::new(""));
                guard.entries.insert(to.join(suffix), child.clone());
            }
            for (candidate, _) in descendants {
                guard.entries.remove(&candidate);
            }
            guard.entries.remove(&from);
        } else {
            guard.entries.insert(to, entry);
            guard.entries.remove(&from);
        }
        Ok(())
    }

    fn create_dir_all_sync(&self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        ensure_dirs(&mut guard, &path)
    }
}

#[async_trait(?Send)]
impl FsProvider for MemoryFsProvider {
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.default_current_dir.clone())
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let path = normalize_path(path)?;
        let existing_entry = {
            let guard = self.inner.lock().expect("memory filesystem lock poisoned");
            guard.entries.get(&path).cloned()
        };
        if flags.create_new && existing_entry.is_some() {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!("entry already exists: {}", path.display()),
            ));
        }
        let existing = match existing_entry {
            Some(MemoryEntry::Directory) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("not a file: {}", path.display()),
                ));
            }
            Some(MemoryEntry::File { readonly: true, .. })
                if flags.write || flags.append || flags.truncate =>
            {
                return Err(io::Error::new(
                    ErrorKind::PermissionDenied,
                    format!("file is readonly: {}", path.display()),
                ));
            }
            Some(MemoryEntry::File { bytes, .. }) => Some(bytes),
            None => None,
        };
        if existing.is_none() && !flags.create && !flags.create_new {
            return Err(not_found(&path));
        }
        let bytes = if flags.truncate {
            Vec::new()
        } else {
            existing.unwrap_or_default()
        };
        let mut cursor = Cursor::new(bytes);
        if flags.append {
            cursor.seek(SeekFrom::End(0))?;
        }
        Ok(Box::new(MemoryFileHandle {
            provider: self.clone(),
            path,
            cursor,
            writable: flags.write || flags.append || flags.truncate,
            dirty: flags.truncate,
            flushed: false,
        }))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.read_project_path(path)
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.write_project_path(path, data)
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::File { .. }) => {
                guard.entries.remove(&path);
                Ok(())
            }
            Some(MemoryEntry::Directory) => Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("not a file: {}", path.display()),
            )),
            None => Err(not_found(&path)),
        }
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata_project_path(path)
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata_project_path(path)
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.list_project_path(path)
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let normalized = normalize_path(path)?;
        self.metadata_project_path(&normalized)?;
        Ok(normalized)
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.create_dir_project_path(path, false)
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.create_dir_project_path(path, true)
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        {
            let guard = self.inner.lock().expect("memory filesystem lock poisoned");
            match guard.entries.get(&path) {
                Some(MemoryEntry::Directory) => {}
                Some(MemoryEntry::File { .. }) => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidInput,
                        format!("not a directory: {}", path.display()),
                    ));
                }
                None => return Err(not_found(&path)),
            }
        }
        self.remove_project_path(path, false)
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.remove_project_path(path, true)
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.rename_project_path(from, to)
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let path = normalize_path(path)?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get_mut(&path) {
            Some(MemoryEntry::File {
                readonly: value, ..
            }) => {
                *value = readonly;
                Ok(())
            }
            Some(MemoryEntry::Directory) => Ok(()),
            None => Err(not_found(&path)),
        }
    }
}

struct MemoryFileHandle {
    provider: MemoryFsProvider,
    path: PathBuf,
    cursor: Cursor<Vec<u8>>,
    writable: bool,
    dirty: bool,
    flushed: bool,
}

impl MemoryFileHandle {
    fn flush_to_provider(&mut self) -> io::Result<()> {
        if !self.writable || !self.dirty || self.flushed {
            return Ok(());
        }
        self.provider
            .write_project_path(&self.path, self.cursor.get_ref())?;
        self.flushed = true;
        Ok(())
    }
}

impl Read for MemoryFileHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Write for MemoryFileHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if !self.writable {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "file is not open for writing",
            ));
        }
        self.dirty = true;
        self.flushed = false;
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_to_provider()
    }
}

impl Seek for MemoryFileHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl Drop for MemoryFileHandle {
    fn drop(&mut self) {
        let _ = self.flush_to_provider();
    }
}

#[async_trait(?Send)]
impl FileHandle for MemoryFileHandle {
    async fn flush_async(&mut self) -> io::Result<()> {
        self.flush_to_provider()
    }

    async fn sync_all_async(&mut self) -> io::Result<()> {
        self.flush_to_provider()
    }
}

fn normalize_path(path: &Path) -> io::Result<PathBuf> {
    let mut parts = Vec::<OsString>::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "path prefixes are not supported by memory filesystem",
                ));
            }
            Component::RootDir => parts.clear(),
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.pop().is_none() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidInput,
                        "parent directory traversal is not allowed",
                    ));
                }
            }
            Component::Normal(part) => parts.push(part.to_os_string()),
        }
    }
    let mut normalized = PathBuf::from("/");
    for part in parts {
        normalized.push(part);
    }
    Ok(normalized)
}

fn ensure_parent_dirs(tree: &mut MemoryTree, path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("/"));
    ensure_dirs(tree, parent)
}

fn ensure_parent_dir(tree: &MemoryTree, path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("/"));
    match tree.entries.get(parent) {
        Some(MemoryEntry::Directory) => Ok(()),
        Some(MemoryEntry::File { .. }) => Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("not a directory: {}", parent.display()),
        )),
        None => Err(not_found(parent)),
    }
}

fn ensure_dirs(tree: &mut MemoryTree, path: &Path) -> io::Result<()> {
    let normalized = normalize_path(path)?;
    let mut current = PathBuf::from("/");
    for part in normalized
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_os_string()),
            _ => None,
        })
    {
        current.push(part);
        match tree.entries.get(&current) {
            Some(MemoryEntry::Directory) => {}
            Some(MemoryEntry::File { .. }) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("not a directory: {}", current.display()),
                ));
            }
            None => {
                tree.entries.insert(current.clone(), MemoryEntry::Directory);
            }
        }
    }
    Ok(())
}

fn is_direct_child(parent: &Path, candidate: &Path) -> bool {
    if !is_descendant(parent, candidate) {
        return false;
    }
    candidate
        .strip_prefix(parent)
        .ok()
        .map(|relative| relative.components().count() == 1)
        .unwrap_or(false)
}

fn is_descendant(parent: &Path, candidate: &Path) -> bool {
    candidate != parent && candidate.starts_with(parent)
}

fn entry_type(entry: &MemoryEntry) -> FsFileType {
    match entry {
        MemoryEntry::Directory => FsFileType::Directory,
        MemoryEntry::File { .. } => FsFileType::File,
    }
}

fn metadata_for(path: &Path, entry: Option<&MemoryEntry>) -> io::Result<FsMetadata> {
    match entry {
        Some(MemoryEntry::Directory) => Ok(FsMetadata::new(FsFileType::Directory, 0, None, false)),
        Some(MemoryEntry::File { bytes, readonly }) => Ok(FsMetadata::new(
            FsFileType::File,
            bytes.len() as u64,
            None,
            *readonly,
        )),
        None => Err(not_found(path)),
    }
}

fn not_found(path: &Path) -> io::Error {
    io::Error::new(ErrorKind::NotFound, path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_provider_supports_crud_and_dirs() {
        let provider = MemoryFsProvider::new();
        provider
            .write(Path::new("/src/main.m"), b"disp('hi')")
            .await
            .unwrap();
        assert_eq!(
            provider.read(Path::new("src/main.m")).await.unwrap(),
            b"disp('hi')"
        );
        let entries = provider.read_dir(Path::new("/src")).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path(), Path::new("/src/main.m"));
        provider
            .rename(Path::new("/src/main.m"), Path::new("/src/renamed.m"))
            .await
            .unwrap();
        assert!(provider.read(Path::new("/src/main.m")).await.is_err());
        assert_eq!(
            provider.read(Path::new("/src/renamed.m")).await.unwrap(),
            b"disp('hi')"
        );
        provider.remove_dir_all(Path::new("/src")).await.unwrap();
        assert!(provider.metadata(Path::new("/src")).await.is_err());
    }

    #[test]
    fn memory_provider_file_handle_flushes_to_store() {
        let provider = MemoryFsProvider::new();
        let flags = OpenFlags {
            write: true,
            create: true,
            ..Default::default()
        };
        let mut file = provider.open(Path::new("/out.txt"), &flags).unwrap();
        file.write_all(b"hello").unwrap();
        file.flush().unwrap();
        assert_eq!(provider.read_project_path("/out.txt").unwrap(), b"hello");
    }

    #[tokio::test]
    async fn memory_provider_keeps_file_and_directory_operations_distinct() {
        let provider = MemoryFsProvider::new();
        provider.create_dir(Path::new("/src")).await.unwrap();
        assert!(provider.remove_file(Path::new("/src")).await.is_err());
        assert!(provider.create_dir(Path::new("/src")).await.is_err());

        provider
            .write(Path::new("/src/main.m"), b"x = 1")
            .await
            .unwrap();
        provider
            .set_readonly(Path::new("/src/main.m"), true)
            .await
            .unwrap();
        assert!(matches!(
            provider.write(Path::new("/src/main.m"), b"x = 2").await,
            Err(error) if error.kind() == ErrorKind::PermissionDenied
        ));
        assert!(provider.remove_dir(Path::new("/src/main.m")).await.is_err());
    }

    #[test]
    fn memory_provider_rejects_parent_traversal() {
        let provider = MemoryFsProvider::new();
        assert!(provider.write_project_path("../secret.txt", b"no").is_err());
    }
}
