use super::{normalize, InputManifest, PinnedContent, PinnedEntry};
use crate::{
    AtomicCreateResult, DataManifestDescriptor, DataManifestRequest, DirEntry, FileHandle,
    FsFileType, FsMetadata, FsProvider, OpenFlags,
};
use async_trait::async_trait;
use std::{
    collections::BTreeMap,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub struct PinnedFsProvider {
    base: Arc<dyn FsProvider>,
    snapshot: String,
    entries: BTreeMap<PathBuf, PinnedEntry>,
    outputs: super::output::OutputRegistry,
    mounts: super::mounts::ReadonlyMounts,
}
impl PinnedFsProvider {
    /// Internal identity check for a local captured execution scope. Never
    /// expose or substitute the original live provider through this port.
    pub(crate) fn base_matches(&self, provider: &Arc<dyn FsProvider>) -> bool {
        Arc::ptr_eq(&self.base, provider)
    }
    pub fn new(base: Arc<dyn FsProvider>, manifest: InputManifest) -> io::Result<Self> {
        Ok(Self {
            base,
            snapshot: manifest.snapshot_id.clone(),
            entries: manifest.index()?,
            outputs: Arc::new(Mutex::new(BTreeMap::new())),
            mounts: Default::default(),
        })
    }
    /// Register only an acquired, verified, immutable dependency tree. Hosts
    /// retain its cache lease for this provider's lifetime. Workspace paths are
    /// never read through this port and overlapping mounts are rejected.
    pub fn mount_readonly_dependency(
        &self,
        root: &Path,
        provider: Arc<dyn FsProvider>,
    ) -> io::Result<()> {
        self.mounts
            .insert(root, provider, &self.entries, &self.outputs)
    }
    fn owns(&self, path: &Path) -> bool {
        self.outputs
            .lock()
            .expect("output ownership lock")
            .contains_key(path)
    }
    fn mutable(&self, path: &Path) -> io::Result<()> {
        if self.entries.contains_key(path) || self.mounts.intersects(path) {
            return Err(denied("execution inputs are immutable"));
        }
        Ok(())
    }
}
#[async_trait(?Send)]
impl FsProvider for PinnedFsProvider {
    async fn data_manifest_descriptor(
        &self,
        request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        #[cfg(windows)]
        {
            let requested = if request.path.ends_with(".json") {
                PathBuf::from(&request.path)
            } else {
                PathBuf::from(&request.path).join("manifest.json")
            };
            if let Some((path, provider)) = self.mounts.native_access(&requested)? {
                if request.version.is_some() {
                    return Err(missing());
                }
                let bytes = provider.read(&path).await?;
                return serde_json::from_slice(&bytes)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error));
            }
        }
        let path = normalize(&if request.path.ends_with(".json") {
            PathBuf::from(&request.path)
        } else {
            PathBuf::from(&request.path).join("manifest.json")
        })?;
        if let Some(requested) = &request.version {
            let Some(PinnedEntry {
                content: PinnedContent::File { version_id, .. },
                ..
            }) = self.entries.get(&path)
            else {
                return Err(missing());
            };
            if requested != version_id {
                return Err(denied("dataset version differs from the input snapshot"));
            }
        }
        // Parse the existing typed descriptor projection from immutable bytes.
        // Never call the base provider's live metadata endpoint.
        let bytes = self.read(&path).await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pinned inputs require asynchronous open",
        ))
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        #[cfg(windows)]
        if let Some((path, _)) = self.mounts.native_access(path)? {
            if flags.write || flags.append || flags.truncate || flags.create || flags.create_new {
                return Err(denied("execution inputs are immutable"));
            }
            return Ok(Box::new(InputHandle(Cursor::new(self.read(&path).await?))));
        }
        let path = normalize(path)?;
        if flags.write || flags.append || flags.truncate || flags.create || flags.create_new {
            self.mutable(&path)?;
            if flags.create_new && self.owns(&path) {
                return Err(io::ErrorKind::AlreadyExists.into());
            }
            if !self.owns(&path) {
                if !flags.create && !flags.create_new {
                    return Err(missing());
                }
                if flags.create_new {
                    match self.atomic_create_new(&path, &[]).await? {
                        AtomicCreateResult::Published { .. } => (),
                        AtomicCreateResult::Existing(_) => {
                            return Err(io::ErrorKind::AlreadyExists.into())
                        }
                    }
                } else {
                    self.write(&path, &[]).await?;
                }
            }
            if flags.truncate {
                self.write(&path, &[]).await?;
            }
            let output = self
                .outputs
                .lock()
                .expect("output ownership lock")
                .get(&path)
                .cloned()
                .ok_or_else(missing)?;
            return Ok(Box::new(
                super::output::OutputHandle::open(
                    self.base.clone(),
                    self.outputs.clone(),
                    path,
                    output,
                    flags,
                )
                .await?,
            ));
        }
        let bytes = self.read(&path).await?;
        Ok(Box::new(InputHandle(Cursor::new(bytes))))
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        #[cfg(windows)]
        if let Some((path, provider)) = self.mounts.native_access(path)? {
            return provider.read(&path).await;
        }
        let path = normalize(path)?;
        if let Some(provider) = self.mounts.provider(&path) {
            return provider.read(&path).await;
        }
        if let Some(entry) = self.entries.get(&path) {
            let PinnedContent::File { version_id, size } = &entry.content else {
                return Err(denied("input is a directory"));
            };
            let bytes = self
                .base
                .read_version(&path, version_id, *size, &self.snapshot)
                .await?;
            if bytes.len() as u64 != entry.size() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "immutable input length differs from manifest",
                ));
            }
            return Ok(bytes);
        }
        let output = self
            .outputs
            .lock()
            .expect("output ownership lock")
            .get(&path)
            .cloned()
            .ok_or_else(missing)?;
        output.read(self.base.as_ref(), &path).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let path = normalize(path)?;
        self.mutable(&path)?;
        let current = self
            .outputs
            .lock()
            .expect("output ownership lock")
            .get(&path)
            .cloned();
        let expected = current
            .as_ref()
            .map(super::output::Output::version)
            .transpose()?;
        let version = self.base.publish_output(&path, data, expected).await?;
        self.outputs.lock().expect("output ownership lock").insert(
            path,
            super::output::Output::File {
                version,
                size: data.len() as u64,
            },
        );
        Ok(())
    }

    async fn atomic_create_new(&self, path: &Path, data: &[u8]) -> io::Result<AtomicCreateResult> {
        let path = normalize(path)?;
        if self.entries.contains_key(&path)
            || self.owns(&path)
            || self.mounts.provider(&path).is_some()
        {
            return Ok(AtomicCreateResult::Existing(self.read(&path).await?));
        }
        self.mutable(&path)?;
        let version = self.base.publish_output(&path, data, None).await?;
        self.outputs.lock().expect("output ownership lock").insert(
            path,
            super::output::Output::File {
                version,
                size: data.len() as u64,
            },
        );
        Ok(AtomicCreateResult::Published {
            durability: crate::PublicationDurability::AtomicVisibility,
        })
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        #[cfg(windows)]
        if let Some((path, provider)) = self.mounts.native_access(path)? {
            let metadata = provider.metadata(&path).await?;
            return Ok(FsMetadata::new_with_hash(
                metadata.file_type(),
                metadata.len(),
                metadata.modified(),
                true,
                metadata.hash().map(str::to_owned),
            ));
        }
        let path = normalize(path)?;
        if let Some(provider) = self.mounts.provider(&path) {
            let metadata = provider.metadata(&path).await?;
            return Ok(FsMetadata::new_with_hash(
                metadata.file_type(),
                metadata.len(),
                metadata.modified(),
                true,
                metadata.hash().map(str::to_owned),
            ));
        }
        if let Some(entry) = self.entries.get(&path) {
            return Ok(FsMetadata::new(
                if entry.is_directory() {
                    FsFileType::Directory
                } else {
                    FsFileType::File
                },
                entry.size(),
                None,
                true,
            ));
        }
        self.outputs
            .lock()
            .expect("output ownership lock")
            .get(&path)
            .map(super::output::Output::metadata)
            .ok_or_else(missing)
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        #[cfg(windows)]
        if let Some((path, provider)) = self.mounts.native_access(path)? {
            return provider.read_dir(&path).await;
        }
        let path = normalize(path)?;
        if let Some(provider) = self.mounts.provider(&path) {
            return provider.read_dir(&path).await;
        }
        if !self.metadata(&path).await?.is_dir() {
            return Err(missing());
        }
        let mut entries = Vec::new();
        for (child, entry) in &self.entries {
            if child != &path && child.parent() == Some(path.as_path()) {
                entries.push(DirEntry::new(
                    child.clone(),
                    child.file_name().unwrap_or_default().into(),
                    if entry.is_directory() {
                        FsFileType::Directory
                    } else {
                        FsFileType::File
                    },
                ));
            }
        }
        let outputs: Vec<_> = self
            .outputs
            .lock()
            .expect("output ownership lock")
            .keys()
            .filter(|child| child.parent() == Some(path.as_path()))
            .cloned()
            .collect();
        for child in outputs {
            let metadata = self.metadata(&child).await?;
            entries.push(DirEntry::new(
                child.clone(),
                child.file_name().unwrap_or_default().into(),
                metadata.file_type(),
            ));
        }
        Ok(entries)
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        #[cfg(windows)]
        if let Some((path, _)) = self.mounts.native_access(path)? {
            self.metadata(&path).await?;
            return Ok(path);
        }
        let path = normalize(path)?;
        self.metadata(&path).await?;
        Ok(path)
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        let path = normalize(path)?;
        if self
            .entries
            .get(&path)
            .is_some_and(|entry| entry.is_directory())
        {
            return Ok(());
        }
        self.mutable(&path)?;
        self.base.create_dir(&path).await?;
        self.outputs
            .lock()
            .expect("output ownership lock")
            .insert(path, super::output::Output::Directory);
        Ok(())
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let path = normalize(path)?;
        let mut parts: Vec<_> = path.ancestors().map(Path::to_path_buf).collect();
        parts.reverse();
        for part in parts {
            if self.metadata(&part).await.is_err() {
                self.create_dir(&part).await?;
            }
        }
        Ok(())
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        let path = normalize(path)?;
        self.mutable(&path)?;
        let output = self
            .outputs
            .lock()
            .expect("output ownership lock")
            .get(&path)
            .cloned()
            .ok_or_else(missing)?;
        self.base.remove_output(&path, output.version()?).await?;
        self.outputs
            .lock()
            .expect("output ownership lock")
            .remove(&path);
        Ok(())
    }
    async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
        Err(denied("remove directories outside the pinned execution"))
    }
    async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(denied("remove directories outside the pinned execution"))
    }
    async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(denied("rename files outside the pinned execution"))
    }
    async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
        Err(denied("input permissions are immutable"))
    }
}
fn denied(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, message)
}
fn missing() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        "path is absent from the input snapshot",
    )
}
struct InputHandle(Cursor<Vec<u8>>);
impl Read for InputHandle {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.0.read(bytes)
    }
}
impl Seek for InputHandle {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.0.seek(from)
    }
}
impl Write for InputHandle {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(denied("execution inputs are immutable"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[async_trait(?Send)]
impl FileHandle for InputHandle {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        Ok(FsMetadata::new(
            FsFileType::File,
            self.0.get_ref().len() as u64,
            None,
            true,
        ))
    }
}
