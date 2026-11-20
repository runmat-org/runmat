use crate::{
    normalize_virtual_path, DirEntry, FileHandle, FsMetadata, FsPathNamespace, FsProvider,
    MemoryFsProvider, OpenFlags,
};
use async_trait::async_trait;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Immutable editor text over one captured, read-attenuated virtual provider.
pub(crate) struct SourceOverlayProvider {
    base: Arc<dyn FsProvider>,
    sources: MemoryFsProvider,
}

impl SourceOverlayProvider {
    pub(crate) fn new(
        base: Arc<dyn FsProvider>,
        sources: BTreeMap<PathBuf, String>,
    ) -> io::Result<Self> {
        let memory = MemoryFsProvider::new();
        let mut paths = BTreeSet::new();
        for (path, text) in sources {
            if !FsPathNamespace::VirtualRooted.is_absolute(&path) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "editor source paths must be absolute",
                ));
            }
            let path = normalize_virtual_path(&path)?;
            if !paths.insert(path.clone()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate normalized editor source path",
                ));
            }
            memory.write_project_path(&path, text.as_bytes())?;
            memory.set_readonly_project_path(&path, true)?;
        }
        Ok(Self {
            base,
            sources: memory,
        })
    }

    fn is_source(&self, path: &Path) -> bool {
        self.sources
            .metadata_project_path(path)
            .is_ok_and(|metadata| metadata.is_file())
    }

    fn require_read_flags(flags: &OpenFlags) -> io::Result<()> {
        if flags.write || flags.append || flags.create || flags.create_new || flags.truncate {
            Err(readonly())
        } else {
            Ok(())
        }
    }
}

fn readonly() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "editor source snapshots are read-only",
    )
}

#[async_trait(?Send)]
impl FsProvider for SourceOverlayProvider {
    fn path_namespace(&self) -> FsPathNamespace {
        FsPathNamespace::VirtualRooted
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Self::require_read_flags(flags)?;
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            self.sources.open(&path, flags)
        } else {
            self.base.open(&path, flags)
        }
    }

    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Self::require_read_flags(flags)?;
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            self.sources.open_async(&path, flags).await
        } else {
            self.base.open_async(&path, flags).await
        }
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            self.sources.read(&path).await
        } else {
            self.base.read(&path).await
        }
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            return self.sources.metadata(&path).await;
        }
        match self.base.metadata(&path).await {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.sources.metadata(&path).await
            }
            result => result,
        }
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            return self.sources.metadata(&path).await;
        }
        match self.base.symlink_metadata(&path).await {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.sources.metadata(&path).await
            }
            result => result,
        }
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let path = normalize_virtual_path(path)?;
        if self.is_source(&path) {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "editor source is a file",
            ));
        }
        let overlay = match self.sources.list_project_path(&path) {
            Ok(entries) => Some(entries),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        let base = match self.base.read_dir(&path).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound && overlay.is_some() => {
                Vec::new()
            }
            Err(error) => return Err(error),
        };
        let mut entries = BTreeMap::new();
        for entry in base.into_iter().chain(overlay.into_iter().flatten()) {
            let path = normalize_virtual_path(entry.path())?;
            entries.insert(
                path.clone(),
                DirEntry::new(path, entry.file_name().clone(), entry.file_type()),
            );
        }
        Ok(entries.into_values().collect())
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let path = normalize_virtual_path(path)?;
        match self.base.canonicalize(&path).await {
            Ok(path) => normalize_virtual_path(&path),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.sources.metadata_project_path(&path)?;
                Ok(path)
            }
            Err(error) => Err(error),
        }
    }

    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(readonly())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(readonly())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(readonly())
    }
}
