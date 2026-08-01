use super::loader::{Loader, PackageOrigin};
use super::source::{canonical_path, is_file};
use super::{PackageSourceProvider, ProjectResolveError, ProjectResolveOptions, ResolvedProject};
use crate::source::catalog::{assemble_frozen_project, FrozenPackageInput, FrozenSourceInput};
use crate::{
    build_resolved_graph, reconcile_path_lock, PackageLock, PathLockMode, ResolvedDependencyInput,
    ResolvedGraphInput, ResolvedPackageInput,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub async fn resolve_project_async(
    root_manifest: &Path,
    existing_lock: Option<&PackageLock>,
    options: ProjectResolveOptions,
    sources: &dyn PackageSourceProvider,
) -> Result<ResolvedProject, ProjectResolveError> {
    if let Some(lock) = existing_lock {
        lock.validate()?;
    }
    let root_manifest = canonical_path(root_manifest).await;
    let workspace_root = root_manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let vendor =
        load_vendor_manifest(&workspace_root, existing_lock, options.source_policy.frozen).await?;
    let mut loader = Loader {
        workspace_root: workspace_root.clone(),
        existing_lock,
        options: &options,
        sources,
        packages: BTreeMap::new(),
        acquired_git_sources: BTreeSet::new(),
        acquired_server_sources: BTreeSet::new(),
        acquired_registry_sources: BTreeSet::new(),
        pending_registry: Vec::new(),
        vendor: vendor.as_ref(),
    };
    let mut root_features = options.root_features.clone();
    root_features.insert("default".to_string());
    let root = loader
        .load(
            root_manifest.clone(),
            PackageOrigin::Workspace,
            root_features,
            true,
            &mut Vec::new(),
        )
        .await?;
    super::registry::resolve_dependencies(&mut loader, &root).await?;

    let mut graph_packages = BTreeMap::new();
    let mut features = BTreeMap::new();
    for (key, package) in &loader.packages {
        features.insert(
            package.instance.identity_digest.clone(),
            package.enabled_features.clone(),
        );
        graph_packages.insert(
            key.clone(),
            ResolvedPackageInput {
                instance: package.instance.clone(),
                local_name: package.domain.local_name.clone(),
                dependencies: package
                    .dependencies
                    .iter()
                    .map(|dependency| ResolvedDependencyInput {
                        alias: dependency.spec.alias.clone(),
                        target: dependency.target.clone(),
                        group: dependency.spec.group,
                        optional: dependency.spec.optional,
                        target_predicate: dependency.spec.target.clone(),
                    })
                    .collect(),
                required_capabilities: package.domain.required_capabilities.clone(),
                singleton: package.domain.singleton,
            },
        );
    }
    let graph = build_resolved_graph(ResolvedGraphInput {
        root,
        packages: graph_packages,
        host_capabilities: options.host_capabilities.clone(),
    })?;
    let lock = PackageLock::from_graph_with_features(&graph, options.lock_selection(), &features)?;
    let lock_mode = if options.source_policy.locked || options.source_policy.frozen {
        PathLockMode::Locked
    } else {
        PathLockMode::Live
    };
    let lock_decision = reconcile_path_lock(&lock, existing_lock, lock_mode)?;

    let mut source_inventories = loader
        .packages
        .values()
        .map(|package| package.inventory.clone())
        .collect::<Vec<_>>();
    source_inventories.sort_by(|left, right| left.tree_digest.cmp(&right.tree_digest));
    source_inventories.dedup_by(|left, right| left.tree_digest == right.tree_digest);
    let package_inputs = loader
        .packages
        .into_values()
        .map(|package| FrozenPackageInput {
            instance: package.instance.identity_digest,
            local_name: package.domain.local_name,
            source: package.instance.source,
            root: package.root,
            files: package
                .sources
                .into_iter()
                .map(|source| FrozenSourceInput {
                    descriptor: source.descriptor,
                    bytes: source.bytes,
                })
                .collect(),
        })
        .collect();
    let frozen = assemble_frozen_project(root_manifest, workspace_root, graph, package_inputs)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    Ok(ResolvedProject {
        frozen,
        lock,
        lock_decision,
        acquired_git_sources: loader.acquired_git_sources.into_iter().collect(),
        acquired_server_sources: loader.acquired_server_sources.into_iter().collect(),
        acquired_registry_sources: loader.acquired_registry_sources.into_iter().collect(),
        source_inventories,
    })
}

async fn load_vendor_manifest(
    workspace_root: &Path,
    existing_lock: Option<&PackageLock>,
    frozen: bool,
) -> Result<Option<crate::VendorManifest>, ProjectResolveError> {
    if !frozen {
        return Ok(None);
    }
    let path = workspace_root.join(crate::VENDOR_MANIFEST_FILENAME);
    if !is_file(&path).await {
        return Ok(None);
    }
    let bytes = runmat_filesystem::read_async(&path)
        .await
        .map_err(|error| {
            ProjectResolveError::Invalid(format!("cannot read {}: {error}", path.display()))
        })?;
    let manifest: crate::VendorManifest = serde_json::from_slice(&bytes).map_err(|error| {
        ProjectResolveError::Invalid(format!("cannot decode {}: {error}", path.display()))
    })?;
    manifest.validate().map_err(|error| {
        ProjectResolveError::Invalid(format!("invalid {}: {error}", path.display()))
    })?;
    let lock = existing_lock.ok_or_else(|| {
        ProjectResolveError::Invalid(
            "frozen vendoring requires an existing runmat.lock".to_string(),
        )
    })?;
    if manifest.lock_digest != lock.graph_digest {
        return Err(ProjectResolveError::Invalid(format!(
            "{} records lock digest {}, but runmat.lock has {}",
            path.display(),
            manifest.lock_digest,
            lock.graph_digest
        )));
    }
    Ok(Some(manifest))
}
