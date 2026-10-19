//! Host-owned immutable dependency mounts. A mount cannot replace captured
//! workspace names or run outputs, and never grants mutation authority.
use super::{normalize, PinnedEntry};
use crate::FsProvider;
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
}
