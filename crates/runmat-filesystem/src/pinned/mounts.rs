//! Host-owned immutable dependency mounts. A mount cannot replace captured
//! workspace names or run outputs, and never grants mutation authority.
use super::{normalize, PinnedEntry};
use crate::FsProvider;
#[cfg(windows)]
use std::path::Component;
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Default)]
pub(super) struct ReadonlyMounts(Mutex<BTreeMap<PathBuf, Arc<dyn FsProvider>>>);
impl ReadonlyMounts {
    pub fn insert(
        &self,
        root: &Path,
        provider: Arc<dyn FsProvider>,
        inputs: &BTreeMap<PathBuf, PinnedEntry>,
        outputs: &super::output::OutputRegistry,
    ) -> io::Result<()> {
        #[cfg(windows)]
        let root = if matches!(root.components().next(), Some(Component::Prefix(_))) {
            if provider.path_namespace() != crate::FsPathNamespace::NativeOs {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "native dependency paths require a native provider",
                ));
            }
            let root = normalize_native(root)?;
            if root.parent().is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "dependency mounts require a package directory",
                ));
            }
            root
        } else {
            normalize(root)?
        };
        #[cfg(not(windows))]
        let root = normalize(root)?;
        let overlaps = |path: &Path| path.starts_with(&root) || root.starts_with(path);
        if inputs.iter().any(|(path, entry)| {
            path.starts_with(&root) || (!entry.is_directory() && overlaps(path))
        }) || outputs
            .lock()
            .expect("output ownership lock")
            .keys()
            .any(|path| overlaps(path))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "dependency mount overlaps execution inputs or outputs",
            ));
        }
        let mut mounts = self.0.lock().expect("readonly mount lock");
        if mounts
            .iter()
            .any(|(path, existing)| overlaps(path) && !Arc::ptr_eq(existing, &provider))
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "dependency mounts overlap",
            ));
        }
        if mounts.keys().any(|path| root.starts_with(path)) {
            return Ok(());
        }
        // One verified cache can contain nested packages. Coalesce identical
        // authority rather than allowing overlapping providers to shadow it.
        mounts.retain(|path, _| !path.starts_with(&root));
        mounts.insert(root, provider);
        Ok(())
    }
    pub fn provider(&self, path: &Path) -> Option<Arc<dyn FsProvider>> {
        self.0
            .lock()
            .expect("readonly mount lock")
            .iter()
            .find(|(root, _)| path.starts_with(root))
            .map(|(_, provider)| Arc::clone(provider))
    }
    pub fn intersects(&self, path: &Path) -> bool {
        self.0
            .lock()
            .expect("readonly mount lock")
            .keys()
            .any(|root| path.starts_with(root) || root.starts_with(path))
    }

    /// Native cache paths carry authority only after an explicit host mount.
    /// Workspace manifest paths retain their separate rooted grammar.
    #[cfg(windows)]
    pub(super) fn native_access(
        &self,
        path: &Path,
    ) -> io::Result<Option<(PathBuf, Arc<dyn FsProvider>)>> {
        if !matches!(path.components().next(), Some(Component::Prefix(_))) {
            return Ok(None);
        }
        let path = normalize_native(path)?;
        let provider = self.provider(&path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "native path is outside acquired dependency mounts",
            )
        })?;
        Ok(Some((path, provider)))
    }
}

#[cfg(windows)]
fn normalize_native(path: &Path) -> io::Result<PathBuf> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dependency cache paths require a qualified native root",
        ));
    }
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(part.as_os_str()),
            Component::Normal(name) => result.push(name),
            Component::CurDir => {}
            Component::ParentDir => {
                if !result.pop() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "dependency cache path escapes its native root",
                    ));
                }
            }
        }
    }
    Ok(result)
}
