use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use runmat_package::{
    decode_lock, encode_lock, FrozenProjectHandoff, PackageLock, PathLockDecision,
    ProjectResolveOptions, ResolvedProject,
};
use runmat_package_cache::CacheBackend;
use thiserror::Error;

use crate::{NativeCacheConfig, NativeCacheLease, NativePackageSourceProvider, SqliteCacheBackend};

pub struct NativeProjectResolveRequest {
    pub manifest_path: PathBuf,
    pub options: ProjectResolveOptions,
}

pub struct NativeResolvedProject {
    pub resolved: ResolvedProject,
    pub backend: Arc<SqliteCacheBackend>,
    pub cache_config: NativeCacheConfig,
    _cache_lease: Option<NativeCacheLease>,
    _provider: NativePackageSourceProvider,
}

impl NativeResolvedProject {
    pub fn handoff(&self) -> FrozenProjectHandoff {
        FrozenProjectHandoff::new(self.resolved.frozen.clone())
    }
}

#[derive(Debug, Error)]
pub enum NativeProjectError {
    #[error("failed to locate project manifest '{path}': {source}")]
    Manifest {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read project lockfile '{path}': {source}")]
    LockRead {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to decode project lockfile '{path}': {message}")]
    LockDecode { path: PathBuf, message: String },
    #[error("package resolution failed: {0}")]
    Resolve(String),
    #[error("package cache operation failed: {0}")]
    Cache(String),
    #[error("failed to publish project lockfile '{path}': {message}")]
    LockWrite { path: PathBuf, message: String },
}

/// Resolve and lease one native project through the portable package authority.
///
/// Host composition roots configure authenticated source transports on the
/// provider. This service owns the native lock/cache/materialization lifecycle
/// shared by CLI and Desktop, but no credentials or product policy.
pub async fn resolve_native_project(
    request: NativeProjectResolveRequest,
    backend: Arc<SqliteCacheBackend>,
    cache_config: NativeCacheConfig,
    provider: NativePackageSourceProvider,
) -> Result<NativeResolvedProject, NativeProjectError> {
    let manifest = canonical_manifest(&request.manifest_path)?;
    let lock_path = manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("runmat.lock");
    let existing = read_lock(&lock_path)?;
    let resolved = runmat_package::resolve_project_async(
        &manifest,
        existing.as_ref(),
        request.options,
        &provider,
    )
    .await
    .map_err(|error| NativeProjectError::Resolve(error.to_string()))?;
    let cached_trees = resolved
        .acquired_git_sources
        .iter()
        .map(|source| &source.tree_digest)
        .chain(
            resolved
                .acquired_server_sources
                .iter()
                .map(|source| &source.tree_digest),
        )
        .chain(
            resolved
                .acquired_registry_sources
                .iter()
                .map(|source| &source.tree_digest),
        )
        .collect::<BTreeSet<_>>();
    for inventory in &resolved.source_inventories {
        if cached_trees.contains(&inventory.tree_digest) {
            runmat_package_cache::publish_source_inventory(
                backend.as_ref(),
                inventory,
                now_ms(),
                16,
            )
            .await
            .map_err(|error| NativeProjectError::Cache(error.to_string()))?;
        }
    }
    if resolved.lock_decision == PathLockDecision::WriteGenerated {
        write_lock(&lock_path, &resolved.lock)?;
    }
    let cache_state = backend
        .snapshot()
        .await
        .map_err(|error| NativeProjectError::Cache(error.to_string()))?
        .state;
    let durable_trees = resolved
        .acquired_git_sources
        .iter()
        .map(|source| source.tree_digest.clone())
        .chain(
            resolved
                .acquired_server_sources
                .iter()
                .map(|source| source.tree_digest.clone()),
        )
        .chain(
            resolved
                .acquired_registry_sources
                .iter()
                .map(|source| source.tree_digest.clone()),
        )
        .filter(|digest| cache_state.objects.contains_key(digest))
        .collect();
    let cache_lease = NativeCacheLease::acquire(backend.clone(), durable_trees)
        .await
        .map_err(|error| NativeProjectError::Cache(error.to_string()))?;
    Ok(NativeResolvedProject {
        resolved,
        backend,
        cache_config,
        _cache_lease: cache_lease,
        _provider: provider,
    })
}

fn canonical_manifest(path: &Path) -> Result<PathBuf, NativeProjectError> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| NativeProjectError::Manifest {
                path: path.to_path_buf(),
                source,
            })?
            .join(path)
    };
    std::fs::canonicalize(&path).map_err(|source| NativeProjectError::Manifest { path, source })
}

fn read_lock(path: &Path) -> Result<Option<PackageLock>, NativeProjectError> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let text =
                std::str::from_utf8(&bytes).map_err(|error| NativeProjectError::LockDecode {
                    path: path.to_path_buf(),
                    message: error.to_string(),
                })?;
            decode_lock(text)
                .map(Some)
                .map_err(|error| NativeProjectError::LockDecode {
                    path: path.to_path_buf(),
                    message: error.to_string(),
                })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(NativeProjectError::LockRead {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn write_lock(path: &Path, lock: &PackageLock) -> Result<(), NativeProjectError> {
    let bytes = encode_lock(lock).map_err(|error| NativeProjectError::LockWrite {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| NativeProjectError::LockWrite {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    temporary
        .write_all(bytes.as_bytes())
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| NativeProjectError::LockWrite {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    temporary
        .persist(path)
        .map_err(|error| NativeProjectError::LockWrite {
            path: path.to_path_buf(),
            message: error.error.to_string(),
        })?;
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
