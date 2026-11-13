//! Synchronous algorithms operate only on an explicitly captured private transaction.
//! Persistence and durability belong to the original provider's asynchronous commit.
use super::*;
use crate::workspace::*;
use crate::workspace_access::WorkspaceAccess;

pub(crate) fn from_snapshot(snapshot: WorkspaceSnapshot) -> io::Result<MemoryFsProvider> {
    let root = normalize_path(&snapshot.root)?;
    let mut provider = MemoryFsProvider::new();
    provider.workspace_root = Some(root.clone());
    let mut tree = provider
        .inner
        .lock()
        .map_err(|_| io::Error::other("snapshot lock poisoned"))?;
    ensure_dirs(&mut tree, &root)?;
    for file in snapshot.files {
        if file.path.as_os_str().is_empty()
            || file
                .path
                .components()
                .any(|p| !matches!(p, Component::Normal(_)))
        {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "invalid captured workspace path",
            ));
        }
        let path = root.join(file.path);
        if tree.entries.contains_key(&path) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "duplicate captured workspace path",
            ));
        }
        ensure_parent_dirs(&mut tree, &path)?;
        tree.entries.insert(
            path,
            MemoryEntry::File {
                bytes: Arc::new(file.bytes),
                readonly: file.readonly,
                revision: file.version,
                identity: Arc::new(AtomicU64::new(0)),
            },
        );
    }
    drop(tree);
    Ok(provider)
}

pub(super) fn open(
    provider: &MemoryFsProvider,
    root: &Path,
    writable: bool,
    maximum: u64,
) -> io::Result<Box<dyn WorkspaceAccess>> {
    let admitted = provider.workspace_root.as_ref().ok_or_else(|| {
        io::Error::new(
            ErrorKind::Unsupported,
            "provider coherent workspace access is unsupported",
        )
    })?;
    let root = normalize_path(root)?;
    if maximum == 0 || !root.starts_with(admitted) {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "workspace scope escapes its captured transaction",
        ));
    }
    Ok(Box::new(Access {
        provider: provider.clone(),
        root,
        writable,
        maximum,
    }))
}

struct Access {
    provider: MemoryFsProvider,
    root: PathBuf,
    writable: bool,
    maximum: u64,
}
impl Access {
    fn path(&self, path: &Path) -> io::Result<PathBuf> {
        if path
            .components()
            .any(|p| !matches!(p, Component::Normal(_) | Component::CurDir))
        {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "workspace access escapes captured root",
            ));
        }
        Ok(self.root.join(path))
    }
    fn writing(&self) -> io::Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "workspace access is read-only",
            ))
        }
    }
    fn bound(&self, bytes: u64, maximum: u64) -> io::Result<()> {
        let maximum = maximum.min(self.maximum);
        if bytes > maximum {
            Err(io::Error::new(
                ErrorKind::InvalidInput,
                crate::workspace_access::WorkspaceResourceLimit {
                    required_bytes: bytes,
                    maximum_bytes: maximum,
                },
            ))
        } else {
            Ok(())
        }
    }
}
impl WorkspaceAccess for Access {
    fn read(&self, path: &Path, maximum: u64) -> io::Result<Vec<u8>> {
        let path = self.path(path)?;
        self.bound(self.provider.metadata_project_path(&path)?.len, maximum)?;
        self.provider.read_project_path(path)
    }
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn FileHandle>> {
        let path = self.path(path)?;
        self.bound(
            self.provider.metadata_project_path(&path)?.len,
            self.maximum,
        )?;
        self.provider.open(
            &path,
            &OpenFlags {
                read: true,
                ..Default::default()
            },
        )
    }
    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.provider.metadata_project_path(self.path(path)?)
    }
    fn read_dir(&self, path: &Path, prefix: &Path, maximum: u64) -> io::Result<Vec<DirEntry>> {
        let path = self.path(path)?;
        // Admit listing allocation before constructing its vector.
        let bytes = {
            let tree = self
                .provider
                .inner
                .lock()
                .map_err(|_| io::Error::other("snapshot lock poisoned"))?;
            tree.entries
                .keys()
                .filter(|p| is_direct_child(&path, p))
                .try_fold(0u64, |total, p| {
                    total
                        .checked_add(
                            (std::mem::size_of::<DirEntry>()
                                + p.as_os_str().len() * 4
                                + prefix.as_os_str().len() * 2) as u64,
                        )
                        .ok_or_else(|| io::Error::other("directory inventory overflow"))
                })?
        };
        self.bound(bytes, maximum)?;
        self.provider
            .list_project_path(&path)?
            .into_iter()
            .map(|entry| {
                let relative = entry
                    .path()
                    .strip_prefix(&self.root)
                    .map_err(|_| io::Error::other("invalid captured directory"))?;
                Ok(DirEntry::new(
                    prefix.join(relative),
                    entry.file_name().to_owned(),
                    entry.file_type(),
                ))
            })
            .collect()
    }
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.writing()?;
        self.provider
            .create_dir_project_path(self.path(path)?, true)
    }
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.writing()?;
        let path = self.path(path)?;
        if !self.provider.metadata_project_path(&path)?.is_file() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a captured file",
            ));
        }
        self.provider.remove_project_path(path, false)
    }
    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.writing()?;
        let path = self.path(path)?;
        if !self.provider.metadata_project_path(&path)?.is_dir() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a captured directory",
            ));
        }
        self.provider.remove_project_path(path, false)
    }
    fn atomic_create_new(
        &self,
        path: &Path,
        bytes: &[u8],
        maximum: u64,
    ) -> io::Result<AtomicCreateResult> {
        self.writing()?;
        self.bound(bytes.len() as u64, maximum)?;
        let path = self.path(path)?;
        if let Ok(metadata) = self.provider.metadata_project_path(&path) {
            self.bound(metadata.len, maximum)?;
        }
        self.provider.atomic_create_project_path(&path, bytes)
    }
    fn synchronize_immutable(&self, path: &Path) -> io::Result<()> {
        // This is private staging, not a persistence receipt. The enclosing
        // async transaction must commit before exposing its operation result.
        self.provider.metadata_project_path(self.path(path)?)?;
        Ok(())
    }
    fn capture(
        &self,
        mut request: WorkspaceCaptureRequest,
        maximum: u64,
    ) -> io::Result<WorkspaceSnapshot> {
        if request.root != Path::new("/") && request.root != Path::new(".") {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "capture root must be scope-relative",
            ));
        }
        request.root = self.root.clone();
        request.max_bytes = request.max_bytes.min(maximum).min(self.maximum);
        workspace::capture(&self.provider, request)
    }
    fn publish(
        &self,
        mut request: WorkspacePublicationRequest,
        maximum: u64,
    ) -> io::Result<WorkspacePublicationOutcome> {
        self.writing()?;
        if request.root != Path::new("/") && request.root != Path::new(".") {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "publication root must be scope-relative",
            ));
        }
        let bytes = request.edits.iter().try_fold(0u64, |total, edit| {
            total
                .checked_add(match edit {
                    WorkspaceEdit::Write { bytes, .. } => bytes.len() as u64,
                    _ => 0,
                })
                .ok_or_else(|| io::Error::other("publication byte overflow"))
        })?;
        self.bound(bytes, maximum)?;
        request.root = self.root.clone();
        workspace::publish(&self.provider, request)
    }
}
