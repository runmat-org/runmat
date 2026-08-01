use crate::concurrency::ProcessLock;
use crate::filesystem::{make_tree_removable, CacheLayout};
use crate::NativeCacheError;
use runmat_package::ContentDigest;
use runmat_package_cache::{CacheBackend, CacheObject, GcPlan, GcPolicy};
use std::path::Path;

pub async fn execute<B: CacheBackend>(
    backend: &B,
    layout: &CacheLayout,
    policy: GcPolicy,
    retries: usize,
) -> Result<GcPlan, NativeCacheError> {
    layout.create()?;
    let plan = runmat_package_cache::execute_gc(backend, policy, retries).await?;
    remove_orphaned_trees(backend, layout).await?;
    Ok(plan)
}

pub async fn remove_orphaned_trees<B: CacheBackend>(
    backend: &B,
    layout: &CacheLayout,
) -> Result<Vec<ContentDigest>, NativeCacheError> {
    layout.create()?;
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(&layout.trees)
        .map_err(|error| NativeCacheError::io(&layout.trees, error))?
    {
        let entry = entry.map_err(|error| NativeCacheError::io(&layout.trees, error))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Ok(digest) = format!("sha256:{name}").parse::<ContentDigest>() else {
            continue;
        };
        candidates.push((digest, entry.path()));
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));

    let mut removed = Vec::new();
    for (digest, path) in candidates {
        let _lock = ProcessLock::acquire(&layout.materialization_lock(&digest))?;
        let snapshot = backend
            .snapshot()
            .await
            .map_err(runmat_package_cache::CacheError::from)?;
        if matches!(
            snapshot.state.objects.get(&digest),
            Some(CacheObject::Tree(_))
        ) {
            continue;
        }
        remove_tree_entry(&path)?;
        removed.push(digest);
    }
    Ok(removed)
}

fn remove_tree_entry(path: &Path) -> Result<(), NativeCacheError> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(NativeCacheError::io(path, error)),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        make_tree_removable(path)?;
        std::fs::remove_dir_all(path).map_err(|error| NativeCacheError::io(path, error))
    } else {
        std::fs::remove_file(path).map_err(|error| NativeCacheError::io(path, error))
    }
}
