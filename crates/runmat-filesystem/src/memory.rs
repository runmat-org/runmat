use crate::{
    AtomicCreateResult, DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags,
    PublicationDurability,
};
use async_trait::async_trait;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::{self, Cursor, ErrorKind, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

mod bounded;
mod handle_commit;
mod mapped;
mod rename;
pub(crate) mod scope;
mod staged;

#[derive(Clone, Debug)]
pub struct MemoryFsProvider {
    // Only an explicitly captured transaction can supply synchronous workspace
    // scopes. Ordinary MemoryFS retains its existing asynchronous authority.
    workspace_root: Option<PathBuf>,
    default_current_dir: PathBuf,
    inner: Arc<Mutex<MemoryTree>>,
}

#[derive(Clone, Debug)]
enum MemoryEntry {
    Directory,
    File {
        bytes: Arc<Vec<u8>>,
        readonly: bool,
        revision: String,
        identity: Arc<AtomicU64>,
    },
}

/// Identity for a whole-file replacement: an existing file keeps its identity
/// at the next version; a new file starts a fresh identity.
fn successor_identity(existing: Option<&MemoryEntry>) -> io::Result<Arc<AtomicU64>> {
    match existing {
        Some(MemoryEntry::File { identity, .. }) => {
            bump_identity(identity)?;
            Ok(identity.clone())
        }
        _ => Ok(Arc::new(AtomicU64::new(0))),
    }
}

fn bump_identity(identity: &AtomicU64) -> io::Result<u64> {
    let next = identity
        .load(Ordering::SeqCst)
        .checked_add(1)
        .ok_or_else(|| io::Error::other("memory file revision exhausted"))?;
    identity.store(next, Ordering::SeqCst);
    Ok(next)
}

mod workspace;

#[derive(Debug, Clone)]
struct MemoryTree {
    entries: BTreeMap<PathBuf, MemoryEntry>,
    publications: BTreeMap<crate::workspace::WorkspacePublicationId, workspace::Published>,
    executions: Arc<BTreeMap<String, crate::execution::ExecutionLeaseState>>,
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
            workspace_root: None,
            default_current_dir,
            inner: Arc::new(Mutex::new(MemoryTree {
                entries,
                publications: BTreeMap::new(),
                executions: Arc::new(BTreeMap::new()),
            })),
        };
        let _ = provider.create_dir_all_sync(&provider.default_current_dir);
        provider
    }

    pub fn reset(&self) {
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        guard.entries.clear();
        guard.publications.clear();
        guard
            .entries
            .insert(PathBuf::from("/"), MemoryEntry::Directory);
        ensure_dirs(&mut guard, &self.default_current_dir).expect("default cwd must be valid");
    }

    fn atomic_create_project_path(
        &self,
        path: &Path,
        bytes: &[u8],
    ) -> io::Result<AtomicCreateResult> {
        let path = normalize_path(path)?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::File { bytes, .. }) => {
                return Ok(AtomicCreateResult::Existing(bytes.as_ref().clone()));
            }
            Some(MemoryEntry::Directory) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("not a file: {}", path.display()),
                ));
            }
            None => {}
        }
        ensure_parent_dirs(&mut guard, &path)?;
        let identity = successor_identity(guard.entries.get(&path))?;
        guard.entries.insert(
            path,
            MemoryEntry::File {
                bytes: Arc::new(bytes.to_vec()),
                readonly: false,
                revision: uuid::Uuid::new_v4().to_string(),
                identity,
            },
        );
        Ok(AtomicCreateResult::Published {
            durability: PublicationDurability::AtomicVisibility,
        })
    }

    pub fn read_project_path(&self, path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        let path = normalize_path(path.as_ref())?;
        let guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get(&path) {
            Some(MemoryEntry::File { bytes, .. }) => Ok(bytes.as_ref().clone()),
            _ => Err(not_found(&path)),
        }
    }

    pub fn write_project_path(&self, path: impl AsRef<Path>, data: &[u8]) -> io::Result<()> {
        let path = normalize_path(path.as_ref())?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        let identity = match guard.entries.get(&path) {
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
            Some(MemoryEntry::File { identity, .. }) => identity.clone(),
            None => Arc::new(AtomicU64::new(0)),
        };
        ensure_parent_dirs(&mut guard, &path)?;
        let replacement = data.to_vec();
        if guard.entries.contains_key(&path) {
            bump_identity(&identity)?;
        }
        guard.entries.insert(
            path,
            MemoryEntry::File {
                bytes: Arc::new(replacement),
                readonly: false,
                revision: uuid::Uuid::new_v4().to_string(),
                identity,
            },
        );
        Ok(())
    }

    /// Frozen invocation-tree construction; ordinary writes still respect
    /// the file's readonly bit after this initialization step.
    pub fn set_readonly_project_path(
        &self,
        path: impl AsRef<Path>,
        readonly: bool,
    ) -> io::Result<()> {
        let path = normalize_path(path.as_ref())?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        match guard.entries.get_mut(&path) {
            Some(MemoryEntry::File {
                readonly: value, ..
            }) => {
                *value = readonly;
                Ok(())
            }
            Some(MemoryEntry::Directory) => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "selected readonly entry is a directory",
            )),
            None => Err(not_found(&path)),
        }
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
        rename::execute(&mut guard, &from, &to)
    }

    fn create_dir_all_sync(&self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        let mut guard = self.inner.lock().expect("memory filesystem lock poisoned");
        ensure_dirs(&mut guard, &path)
    }
}

#[async_trait(?Send)]
impl FsProvider for MemoryFsProvider {
    fn supports_workspace_transactions(&self) -> bool {
        true
    }
    fn workspace_access(
        &self,
        root: &Path,
        writable: bool,
        maximum: u64,
    ) -> io::Result<Box<dyn crate::workspace_access::WorkspaceAccess>> {
        scope::open(self, root, writable, maximum)
    }
    async fn transition_execution(
        &self,
        work: &crate::execution::ExecutionWorkId,
        command: crate::execution::ExecutionLeaseCommand,
    ) -> io::Result<crate::execution::ExecutionLeaseOutcome> {
        let key = crate::execution::work_key(work)?;
        let mut tree = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("memory filesystem lock poisoned"))?;
        let (next, outcome) = crate::execution::transition_execution_lease(
            work,
            tree.executions.get(&key),
            &command,
            crate::execution::authority_time_ms()?,
        )
        .map_err(io::Error::other)?;
        if let Some(next) = next {
            if !tree.executions.contains_key(&key)
                && tree.executions.len() >= crate::execution::MAX_EXECUTION_RECORDS
            {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "execution ownership record budget exhausted",
                ));
            }
            Arc::make_mut(&mut tree.executions).insert(key, next);
        }
        Ok(outcome)
    }
    async fn lookup_workspace_publication(
        &self,
        request: &crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<crate::workspace::WorkspacePublicationOutcome>> {
        workspace::lookup(self, request)
    }
    async fn capture_workspace(
        &self,
        request: crate::workspace::WorkspaceCaptureRequest,
    ) -> io::Result<crate::workspace::WorkspaceSnapshot> {
        workspace::capture(self, request)
    }

    async fn publish_workspace(
        &self,
        request: crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<crate::workspace::WorkspacePublicationOutcome> {
        workspace::publish(self, request)
    }
    async fn publish_workspace_fenced(
        &self,
        request: crate::workspace::WorkspacePublicationRequest,
        lease: &crate::execution::ExecutionLease,
    ) -> io::Result<crate::execution::FencedWorkspacePublicationOutcome> {
        crate::execution::fenced_outcome(workspace::publish_with_lease(self, request, Some(lease)))
    }
    fn path_namespace(&self) -> crate::FsPathNamespace {
        crate::FsPathNamespace::VirtualRooted
    }

    async fn open_bounded_input(&self, path: &Path) -> io::Result<Option<crate::FsBoundedInput>> {
        bounded::open(self, path).map(Some)
    }

    fn managed_read_contract(&self) -> Option<crate::FsManagedReadContract> {
        Some(bounded::managed_read_contract())
    }

    fn resource_lease(&self) -> Option<Arc<dyn crate::FsResourceLease>> {
        None
    }

    fn resource_lease_handle(&self) -> Option<crate::FsResourceLeaseHandle> {
        Some(bounded::resource_lease_handle())
    }

    async fn open_mapped_backing(
        &self,
        path: &Path,
        writable: bool,
    ) -> io::Result<Option<Box<dyn crate::FsMappedBacking>>> {
        mapped::open(self.clone(), path, writable).map(Some)
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn crate::FsStagedOutput>>> {
        staged::begin(self.clone(), destination).map(Some)
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.default_current_dir.clone())
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let path = normalize_path(path)?;
        let (existing_entry, existing_version) = {
            let guard = self.inner.lock().expect("memory filesystem lock poisoned");
            let entry = guard.entries.get(&path);
            let version = match entry {
                Some(MemoryEntry::File { identity, .. }) => Some(identity.load(Ordering::SeqCst)),
                Some(MemoryEntry::Directory) | None => None,
            };
            (entry.cloned(), version)
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
            Some(MemoryEntry::File {
                bytes, identity, ..
            }) => Some((bytes, identity)),
            None => None,
        };
        if existing.is_none() && !flags.create && !flags.create_new {
            return Err(not_found(&path));
        }
        let originally_present = existing.is_some();
        let identity = existing.as_ref().map_or_else(
            || Arc::new(AtomicU64::new(0)),
            |(_, identity)| identity.clone(),
        );
        let opened_version = existing_version.unwrap_or(0);
        let bytes = if flags.truncate {
            Vec::new()
        } else {
            existing.map_or_else(Vec::new, |(bytes, _)| bytes.as_ref().clone())
        };
        let mut cursor = Cursor::new(bytes);
        if flags.append {
            cursor.seek(SeekFrom::End(0))?;
        }
        Ok(Box::new(MemoryFileHandle {
            provider: self.clone(),
            path,
            identity,
            opened_version,
            originally_present,
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

    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String> {
        let path = normalize_path(path)?;
        let mut tree = self.inner.lock().expect("memory filesystem lock poisoned");
        match (tree.entries.get(&path), expected) {
            (None, None) => (),
            (
                Some(MemoryEntry::File {
                    revision,
                    readonly: false,
                    ..
                }),
                Some(expected),
            ) if revision == expected => (),
            _ => {
                return Err(io::Error::new(
                    ErrorKind::PermissionDenied,
                    "output version changed",
                ))
            }
        }
        ensure_parent_dirs(&mut tree, &path)?;
        let revision = uuid::Uuid::new_v4().to_string();
        let identity = successor_identity(tree.entries.get(&path))?;
        tree.entries.insert(
            path,
            MemoryEntry::File {
                bytes: Arc::new(data.to_vec()),
                readonly: false,
                revision: revision.clone(),
                identity,
            },
        );
        Ok(revision)
    }

    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()> {
        let path = normalize_path(path)?;
        let mut tree = self.inner.lock().expect("memory filesystem lock poisoned");
        match tree.entries.get(&path) {
            Some(MemoryEntry::File {
                revision,
                readonly: false,
                ..
            }) if revision == expected => (),
            _ => {
                return Err(io::Error::new(
                    ErrorKind::PermissionDenied,
                    "output version changed",
                ))
            }
        }
        tree.entries.remove(&path);
        Ok(())
    }

    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        size: u64,
        _snapshot: &str,
    ) -> io::Result<Vec<u8>> {
        let path = normalize_path(path)?;
        let tree = self.inner.lock().expect("memory filesystem lock poisoned");
        match tree.entries.get(&path) {
            Some(MemoryEntry::File {
                revision, bytes, ..
            }) if revision == version && bytes.len() as u64 == size => Ok(bytes.as_ref().clone()),
            _ => Err(io::Error::new(
                ErrorKind::NotFound,
                "memory file version is unavailable",
            )),
        }
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

    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
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

    async fn atomic_create_new(&self, path: &Path, bytes: &[u8]) -> io::Result<AtomicCreateResult> {
        self.atomic_create_project_path(path, bytes)
    }
}

struct MemoryFileHandle {
    provider: MemoryFsProvider,
    path: PathBuf,
    identity: Arc<AtomicU64>,
    opened_version: u64,
    originally_present: bool,
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
        self.opened_version = handle_commit::commit(
            &self.provider,
            &self.path,
            &self.identity,
            self.opened_version,
            self.originally_present,
            self.cursor.get_ref(),
        )?;
        self.originally_present = true;
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
        Some(MemoryEntry::File {
            bytes, readonly, ..
        }) => Ok(FsMetadata::new(
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
    use futures::executor;
    use std::sync::{Arc, Barrier};

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

    #[test]
    fn concurrent_memory_publishers_choose_exactly_one_complete_payload() {
        let provider = Arc::new(MemoryFsProvider::new());
        let barrier = Arc::new(Barrier::new(2));
        let handles = [b"first".to_vec(), b"second".to_vec()].map(|payload| {
            let provider = Arc::clone(&provider);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                executor::block_on(provider.atomic_create_new(Path::new("/artifact"), &payload))
            })
        });
        let outcomes = handles.map(|handle| handle.join().unwrap().unwrap());
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, AtomicCreateResult::Published { .. }))
                .count(),
            1
        );
        let winning = provider.read_project_path("/artifact").unwrap();
        assert!(winning == b"first" || winning == b"second");
        assert!(outcomes.iter().any(
            |outcome| matches!(outcome, AtomicCreateResult::Existing(bytes) if bytes == &winning)
        ));
    }
}
