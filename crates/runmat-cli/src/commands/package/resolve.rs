use super::credentials::CliAccessTokenProvider;
use super::server_transport::default_server_origin;
use crate::cli::{Cli, PackageProjectArgs};
use anyhow::{Context, Result};
use runmat_package::{
    DependencyGroup, HostCapability, ProjectResolveOptions, SourceAcquisitionIntent,
    SourceAcquisitionPolicy,
};
use runmat_package_cache_native::{
    git::NativeGitClient, resolve_native_project, NativeCacheConfig, NativePackageSourceProvider,
    NativeProjectResolveRequest, SqliteCacheBackend,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) use runmat_package_cache_native::NativeResolvedProject;

pub(super) async fn resolve(
    args: &PackageProjectArgs,
    cli: &Cli,
    intent: SourceAcquisitionIntent,
) -> Result<NativeResolvedProject> {
    resolve_with_groups(
        args,
        cli,
        intent,
        [DependencyGroup::Runtime].into_iter().collect(),
    )
    .await
}

async fn resolve_with_groups(
    args: &PackageProjectArgs,
    cli: &Cli,
    intent: SourceAcquisitionIntent,
    groups: BTreeSet<DependencyGroup>,
) -> Result<NativeResolvedProject> {
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
    let credentials = Arc::new(CliAccessTokenProvider);
    let server_transport = Arc::new(
        runmat_package_cache_native::http::HttpServerSnapshotTransport::new(credentials.clone()),
    );
    let default_server_origin = default_server_origin();
    let default_registry_index = default_server_origin.clone();
    let provider = provider
        .with_server_transport(server_transport)
        .with_registry_transport(Arc::new(
            runmat_package_cache_native::http::HttpRegistryTransport::new(credentials),
        ))
        .with_private_artifact_decryptor(Arc::new(
            runmat_package_cache_native::registry::OsCredentialPrivateArtifactDecryptor,
        ));
    resolve_native_project(
        NativeProjectResolveRequest {
            manifest_path: args.manifest_path.clone(),
            options: ProjectResolveOptions {
                target: target_lexicon::HOST.to_string(),
                default_server_origin,
                default_registry_index,
                groups,
                root_features: BTreeSet::new(),
                host_capabilities: native_capabilities(),
                source_intent: intent,
                source_policy: SourceAcquisitionPolicy {
                    locked: cli.locked,
                    frozen: cli.frozen,
                    offline: cli.offline || cli.frozen,
                },
            },
        },
        backend,
        cache_config,
        provider,
    )
    .await
    .map_err(anyhow::Error::from)
}

pub(crate) async fn resolve_for_test_manifest(
    manifest_path: PathBuf,
    cli: &Cli,
) -> Result<NativeResolvedProject> {
    resolve_with_groups(
        &PackageProjectArgs { manifest_path },
        cli,
        SourceAcquisitionIntent::Execute,
        [DependencyGroup::Runtime, DependencyGroup::Test]
            .into_iter()
            .collect(),
    )
    .await
}

fn native_capabilities() -> BTreeSet<HostCapability> {
    [
        HostCapability::Network,
        HostCapability::NativeLibrary,
        HostCapability::Mex,
        HostCapability::Jvm,
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
        .await
        .context("failed to install resolved project graph")?;
    Ok(Some(project))
}

/// Project ownership comes from the frozen access row, not an absolute path
/// labeled as the root package. Pure, unprojected inputs retain their local identity.
pub(crate) fn executable_source(
    project: Option<&runmat_package::FrozenProject>,
    file: &Path,
    text: String,
) -> Result<runmat_core::ExecutableSource> {
    let Some(project) = project else {
        return Ok(runmat_core::ExecutableSource::new(
            "root",
            file.to_string_lossy(),
            text,
        ));
    };
    let canonical_file = std::fs::canonicalize(file)
        .with_context(|| format!("failed to resolve entrypoint {}", file.display()))?;
    let mut selected = None;
    for (descriptor, access) in project.all_sources() {
        let canonical_access = std::fs::canonicalize(access.as_path()).with_context(|| {
            format!(
                "failed to resolve frozen source {}",
                access.as_path().display()
            )
        })?;
        if canonical_access == canonical_file {
            anyhow::ensure!(
                selected.is_none(),
                "entrypoint has ambiguous frozen project ownership"
            );
            selected = Some(descriptor);
        }
    }
    let descriptor =
        selected.context("entrypoint is absent from the frozen project source catalog")?;
    runmat_core::ExecutableSource::from_project_source(&descriptor.id, text)
        .map_err(anyhow::Error::msg)
}
