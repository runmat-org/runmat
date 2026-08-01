use super::private_keys::KeyringPrivateArtifactDecryptor;
use super::registry_transport::RunMatRegistryTransport;
use super::server_transport::RunMatServerSnapshotTransport;
use crate::cli::{Cli, PackageProjectArgs};
use anyhow::{Context, Result};
use runmat_package::{
    decode_lock, encode_lock, DependencyGroup, HostCapability, PackageLock, PathLockDecision,
    ProjectResolveOptions, ResolvedProject, SourceAcquisitionIntent, SourceAcquisitionPolicy,
};
use runmat_package_cache_native::{
    git::NativeGitClient, NativeCacheConfig, NativeCacheLease, NativePackageSourceProvider,
    SqliteCacheBackend,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) struct NativeResolvedProject {
    pub resolved: ResolvedProject,
    pub backend: Arc<SqliteCacheBackend>,
    pub cache_config: NativeCacheConfig,
    _cache_lease: Option<NativeCacheLease>,
    _provider: NativePackageSourceProvider,
}

pub(super) async fn resolve(
    args: &PackageProjectArgs,
    cli: &Cli,
    intent: SourceAcquisitionIntent,
) -> Result<NativeResolvedProject> {
    let manifest = canonical_manifest(&args.manifest_path)?;
    let lock_path = manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("runmat.lock");
    let existing = read_lock(&lock_path)?;
    let cache_config = NativeCacheConfig::platform_default()
        .context("failed to locate the platform package cache")?;
    let layout = cache_config.layout();
    let backend = Arc::new(
        SqliteCacheBackend::open(&cache_config)
            .context("failed to open the shared package cache")?,
    );
    let provider = NativePackageSourceProvider::new(
        NativeGitClient::new(layout.clone()),
        backend.clone(),
        layout,
    );
    let server_transport = Arc::new(RunMatServerSnapshotTransport);
    let default_server_origin = server_transport.default_origin();
    let default_registry_index = default_server_origin.clone();
    let provider = provider
        .with_server_transport(server_transport)
        .with_registry_transport(Arc::new(RunMatRegistryTransport))
        .with_private_artifact_decryptor(Arc::new(KeyringPrivateArtifactDecryptor));
    let resolved = runmat_package::resolve_project_async(
        &manifest,
        existing.as_ref(),
        ProjectResolveOptions {
            target: target_lexicon::HOST.to_string(),
            default_server_origin,
            default_registry_index,
            groups: [DependencyGroup::Runtime].into_iter().collect(),
            root_features: BTreeSet::new(),
            host_capabilities: native_capabilities(),
            source_intent: intent,
            source_policy: SourceAcquisitionPolicy {
                locked: cli.locked,
                frozen: cli.frozen,
                offline: cli.offline || cli.frozen,
            },
        },
        &provider,
    )
    .await
    .map_err(|error| anyhow::anyhow!("package resolution failed: {error}"))?;
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
            runmat_package_cache::publish_source_inventory(&backend, inventory, now_ms(), 16)
                .await
                .context("failed to cache package source inventory")?;
        }
    }
    if resolved.lock_decision == PathLockDecision::WriteGenerated {
        write_lock(&lock_path, &resolved.lock)?;
    }
    let cache_state = runmat_package_cache::CacheBackend::snapshot(backend.as_ref())
        .await
        .context("failed to inspect the resolved package cache")?
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
        .context("failed to lease the resolved package graph")?;
    Ok(NativeResolvedProject {
        resolved,
        backend,
        cache_config,
        _cache_lease: cache_lease,
        _provider: provider,
    })
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn canonical_manifest(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(path)
    };
    std::fs::canonicalize(&path)
        .with_context(|| format!("failed to locate project manifest {}", path.display()))
}

fn read_lock(path: &Path) -> Result<Option<PackageLock>> {
    match std::fs::read(path) {
        Ok(bytes) => std::str::from_utf8(&bytes)
            .context("runmat.lock is not valid UTF-8")
            .and_then(|text| decode_lock(text).map_err(anyhow::Error::from))
            .with_context(|| format!("failed to decode {}", path.display()))
            .map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn write_lock(path: &Path, lock: &PackageLock) -> Result<()> {
    let bytes = encode_lock(lock).context("failed to encode runmat.lock")?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("failed to stage runmat.lock")?;
    use std::io::Write as _;
    temporary
        .write_all(bytes.as_bytes())
        .context("failed to write staged runmat.lock")?;
    temporary
        .as_file()
        .sync_all()
        .context("failed to sync staged runmat.lock")?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to atomically publish {}", path.display()))?;
    Ok(())
}

fn native_capabilities() -> BTreeSet<HostCapability> {
    [
        HostCapability::Network,
        HostCapability::NativeLibrary,
        HostCapability::Subprocess,
    ]
    .into_iter()
    .collect()
}

pub(crate) async fn resolve_for_source(
    source: &Path,
    cli: &Cli,
) -> Result<Option<NativeResolvedProject>> {
    let Some(manifest_path) = runmat_config::project::discover_project_manifest_from(source) else {
        return Ok(None);
    };
    resolve(
        &PackageProjectArgs { manifest_path },
        cli,
        SourceAcquisitionIntent::Execute,
    )
    .await
    .map(Some)
}

pub(crate) async fn install_project_for_source(
    session: &mut runmat_core::RunMatSession,
    source: &Path,
    cli: &Cli,
) -> Result<Option<NativeResolvedProject>> {
    let Some(project) = resolve_for_source(source, cli).await? else {
        return Ok(None);
    };
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(
            project.resolved.frozen.clone(),
        ))
        .context("failed to install resolved project graph")?;
    Ok(Some(project))
}
