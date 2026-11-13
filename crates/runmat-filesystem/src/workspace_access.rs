//! Scoped synchronous access shares the native workspace journal and recovery fence.
mod immutable_batch;
mod resource;
mod transaction;
use crate::workspace::{
    WorkspaceCaptureRequest, WorkspacePublicationOutcome, WorkspacePublicationRequest,
    WorkspaceSnapshot,
};
use crate::{AtomicCreateResult, DirEntry, FileHandle, FsMetadata};
pub use immutable_batch::{ImmutableBatch, ImmutableStage};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use resource::exceeded as resource_error;
pub use resource::{resource_limit, WorkspaceResourceLimit};
use std::{io, path::Path};
pub use transaction::WorkspaceTransaction;

/// A provider-owned access scope. Paths are relative to the admitted root.
/// Implementations must make all participating reads/writes coherent with journal recovery.
/// Unsupported providers reject; clients must not substitute native I/O or a private lock.
pub trait WorkspaceAccess: Send + Sync {
    fn read(&self, path: &Path, maximum_bytes: u64) -> io::Result<Vec<u8>>;
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn FileHandle>>;
    fn metadata(&self, path: &Path) -> io::Result<FsMetadata>;
    /// Inputs are scope-relative; returned paths use the caller's logical prefix.
    /// The requested listing inventory must fit before its entry vector is allocated.
    fn read_dir(
        &self,
        path: &Path,
        result_prefix: &Path,
        maximum_bytes: u64,
    ) -> io::Result<Vec<DirEntry>>;
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn remove_dir(&self, path: &Path) -> io::Result<()>;
    fn atomic_create_new(
        &self,
        path: &Path,
        bytes: &[u8],
        maximum_existing_bytes: u64,
    ) -> io::Result<AtomicCreateResult>;
    /// Confirm an immutable file and the directory entries that expose it. Exact retries
    /// require fresh confirmation; matching bytes alone do not establish durability.
    fn synchronize_immutable(&self, _path: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "provider cannot confirm immutable durability",
        ))
    }
    /// Group commit of immutable files: stage each, then make all of them durable and
    /// visible together. Providers without a native group commit publish and confirm each
    /// file on its own.
    fn immutable_batch(&self) -> io::Result<Box<dyn ImmutableBatch + '_>> {
        Ok(Box::new(immutable_batch::SequentialImmutableBatch {
            access: self,
            staged: Vec::new(),
        }))
    }
    /// Capture and conditional publication reuse the scope's recovery fence.
    /// The request root must be `/` or `.`, relative to this admitted scope.
    fn capture(
        &self,
        request: WorkspaceCaptureRequest,
        maximum_resident_bytes: u64,
    ) -> io::Result<WorkspaceSnapshot>;
    fn publish(
        &self,
        request: WorkspacePublicationRequest,
        maximum_resident_bytes: u64,
    ) -> io::Result<WorkspacePublicationOutcome>;
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct NativeAccess {
    root: std::path::PathBuf,
    writable: bool,
    maximum_resident_bytes: u64,
    present: bool,
    #[cfg(any(unix, windows))]
    root_handle: Option<std::fs::File>,
    operation: std::sync::Mutex<()>,
    _guard: Option<crate::workspace_native::Guard>,
    _mount_guard: Option<crate::workspace_native::Guard>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeAccess {
    pub(crate) fn open(
        root: std::path::PathBuf,
        writable: bool,
        mount_guard: Option<crate::workspace_native::Guard>,
        maximum_resident_bytes: u64,
    ) -> io::Result<Self> {
        if maximum_resident_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "workspace scope resident budget must be positive",
            ));
        }
        let present = match std::fs::symlink_metadata(&root) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => true,
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "workspace root must be a real directory",
                ))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(error),
        };
        if writable && !present {
            std::fs::create_dir_all(&root)?;
        }
        // An absent read scope stays absent even if another writer subsequently creates it.
        let present = present || writable;
        #[cfg(unix)]
        let root_handle = if present {
            Some(std::fs::File::open(&root)?)
        } else {
            None
        };
        #[cfg(windows)]
        let root_handle = if present {
            Some(crate::native::windows_identity::open(
                &root,
                crate::native::windows_identity::Kind::Directory,
            )?)
        } else {
            None
        };
        let guard = if present {
            crate::workspace_native::guard_admitted(&root, true, maximum_resident_bytes)?
        } else {
            None
        };
        Ok(Self {
            root,
            writable,
            maximum_resident_bytes,
            present,
            #[cfg(any(unix, windows))]
            root_handle,
            operation: std::sync::Mutex::new(()),
            _guard: guard,
            _mount_guard: mount_guard,
        })
    }
    fn access<T>(&self, writing: bool, action: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
        let _operation = self
            .operation
            .lock()
            .map_err(|_| io::Error::other("workspace scope lock poisoned"))?;
        if writing && !self.writable {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "workspace access is read-only",
            ));
        }
        if !self.present {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "workspace root is absent",
            ));
        }
        let metadata = std::fs::symlink_metadata(&self.root)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "workspace root authority was replaced",
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let admitted = self
                .root_handle
                .as_ref()
                .ok_or_else(|| io::Error::other("workspace root handle absent"))?
                .metadata()?;
            if (admitted.dev(), admitted.ino()) != (metadata.dev(), metadata.ino()) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "workspace root identity changed",
                ));
            }
        }
        #[cfg(windows)]
        {
            use crate::native::windows_identity::{self, Kind};
            let admitted = self
                .root_handle
                .as_ref()
                .ok_or_else(|| io::Error::other("workspace root handle absent"))?;
            let current = windows_identity::open(&self.root, Kind::Directory)?;
            if !windows_identity::same(admitted, &current)? {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "workspace root identity changed",
                ));
            }
        }
        // A prior durable commit may have stopped while materializing. Never expose
        // its partially updated namespace, including through the same held scope.
        crate::workspace_native::recover_locked_admitted(&self.root, self.maximum_resident_bytes)?;
        immutable_batch::recover(&self.root, self.maximum_resident_bytes)?;
        action()
    }
    fn path(&self, relative: &Path, writing: bool) -> io::Result<std::path::PathBuf> {
        use std::path::Component;
        if relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "workspace access path escapes its admitted root",
            ));
        }
        if writing && !self.writable {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "workspace access is read-only",
            ));
        }
        if !self.present {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "workspace root is absent",
            ));
        }
        let path = self.root.join(relative);
        crate::workspace_native::ensure_no_symlink(&self.root, &path)?;
        if writing {
            crate::workspace_native::protected(&self.root, &path)?;
        }
        Ok(path)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkspaceAccess for NativeAccess {
    fn capture(
        &self,
        request: WorkspaceCaptureRequest,
        maximum_resident_bytes: u64,
    ) -> io::Result<WorkspaceSnapshot> {
        self.access(false, || {
            crate::workspace_native::capture_locked_admitted(
                &self.root,
                request,
                maximum_resident_bytes.min(self.maximum_resident_bytes),
            )
        })
    }
    fn publish(
        &self,
        request: WorkspacePublicationRequest,
        maximum_resident_bytes: u64,
    ) -> io::Result<WorkspacePublicationOutcome> {
        self.access(true, || {
            // Only immediate shared memory-provider validation futures; no runtime or second journal lock.
            futures::executor::block_on(crate::workspace_native::publish_locked(
                &self.root,
                request,
                None,
                Some(maximum_resident_bytes.min(self.maximum_resident_bytes)),
            ))
        })
    }
    fn read(&self, path: &Path, maximum_bytes: u64) -> io::Result<Vec<u8>> {
        self.access(false, || {
            file::read(
                &self.path(path, false)?,
                maximum_bytes.min(self.maximum_resident_bytes),
            )
        })
    }
    fn open_read(&self, path: &Path) -> io::Result<Box<dyn FileHandle>> {
        self.access(false, || {
            Ok(Box::new(std::fs::File::open(self.path(path, false)?)?) as Box<dyn FileHandle>)
        })
    }
    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.access(false, || {
            std::fs::symlink_metadata(self.path(path, false)?).map(FsMetadata::from)
        })
    }
    fn read_dir(
        &self,
        relative: &Path,
        result_prefix: &Path,
        maximum_bytes: u64,
    ) -> io::Result<Vec<DirEntry>> {
        self.access(false, || {
            directory::read(
                &self.path(relative, false)?,
                &result_prefix.join(relative),
                maximum_bytes.min(self.maximum_resident_bytes),
            )
        })
    }
    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.access(true, || std::fs::create_dir_all(self.path(path, true)?))
    }
    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.access(true, || std::fs::remove_file(self.path(path, true)?))
    }
    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.access(true, || std::fs::remove_dir(self.path(path, true)?))
    }
    fn immutable_batch(&self) -> io::Result<Box<dyn ImmutableBatch + '_>> {
        if !cfg!(any(unix, windows)) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "durable immutable workspace batches require directory synchronization",
            ));
        }
        Ok(Box::new(immutable_batch::NativeImmutableBatch::new(self)))
    }
    fn synchronize_immutable(&self, relative: &Path) -> io::Result<()> {
        if !cfg!(any(unix, windows)) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "durable immutable workspace synchronization requires directory synchronization",
            ));
        }

        self.access(true, || {
            let path = self.path(relative, false)?;
            if !std::fs::symlink_metadata(&path)?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "immutable synchronization requires a regular file",
                ));
            }
            #[cfg(windows)]
            let file = std::fs::OpenOptions::new().write(true).open(&path)?;
            #[cfg(not(windows))]
            let file = std::fs::File::open(&path)?;
            file.sync_all()?;
            let directory = path.parent().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "immutable file has no parent")
            })?;
            crate::workspace_native::sync_ancestors(
                self.root.parent().unwrap_or(&self.root),
                directory,
            )
        })
    }
    fn atomic_create_new(
        &self,
        path: &Path,
        bytes: &[u8],
        maximum_existing_bytes: u64,
    ) -> io::Result<AtomicCreateResult> {
        self.access(true, || {
            crate::native::atomic_create_new_native_bounded(
                &self.path(path, true)?,
                bytes,
                Some(maximum_existing_bytes.min(self.maximum_resident_bytes)),
            )
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[cfg(not(target_arch = "wasm32"))]
mod directory;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod file;
