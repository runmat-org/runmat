use super::load::load_path_project;
use super::model::{LoadedPathPackage, LoadedPathProject, LoadedSource};
use super::FrozenProjectError;
use crate::{
    build_path_graph, CanonicalPackageId, ContentDigest, GraphError, HostCapability,
    NormalizedRelativePath, PackageAlias, PackageGraph, PackageManifest, PathGraphInput,
    PathPackageInput, RegistryId, SourceId,
};
use runmat_config::project::ProjectSourceFile;
use semver::{Version, VersionReq};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::source::{
    catalog::{assemble_frozen_project, FrozenPackageInput, FrozenSourceInput},
    FrozenProject,
};

#[derive(Debug, Error)]
pub enum PathProjectError {
    #[error("failed to read graph-declared source {path}: {reason}")]
    ReadSource { path: PathBuf, reason: String },
    #[error("invalid project source path {path}: {reason}")]
    SourcePath { path: PathBuf, reason: String },
    #[error("package graph has no instance for manifest {0}")]
    MissingInstance(PathBuf),
    #[error("failed to encode source revision: {0}")]
    Revision(String),
    #[error("{0}")]
    Invalid(String),
}

pub async fn build_frozen_project_async(
    manifest_path: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<FrozenProject, FrozenProjectError> {
    let loaded = load_path_project(manifest_path).await?;
    validate_path_versions(&loaded)?;
    let (input, package_keys) = graph_input(&loaded, host_capabilities)?;
    let graph = build_path_graph(input)?;
    build_catalog(loaded, graph, &package_keys).map_err(Into::into)
}

fn graph_input(
    project: &LoadedPathProject,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<(PathGraphInput, BTreeMap<PathBuf, String>), FrozenProjectError> {
    let package_keys = project
        .packages
        .keys()
        .map(|manifest| {
            stable_package_key(&project.workspace_root, manifest).map(|key| (manifest.clone(), key))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut packages = BTreeMap::new();
    for (manifest_path, package) in &project.packages {
        let key = package_keys[manifest_path].clone();
        let domain = PackageManifest::try_from(&package.manifest)
            .map_err(|error| GraphError::Invalid(error.to_string()))?;
        let workspace_path = workspace_path(&project.workspace_root, &package.project_root)?;
        let canonical_manifest = toml::to_string(&package.manifest).map_err(|error| {
            GraphError::Invalid(format!(
                "cannot encode manifest {}: {error}",
                package.manifest_path.display()
            ))
        })?;
        let dependencies = package
            .dependencies
            .iter()
            .map(|(alias, target)| {
                let alias = alias.parse::<PackageAlias>().map_err(|error| {
                    GraphError::Invalid(format!("invalid dependency alias `{alias}`: {error}"))
                })?;
                let target = package_keys.get(target).cloned().ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "dependency `{alias}` references unloaded manifest {}",
                        target.display()
                    ))
                })?;
                Ok((alias, target))
            })
            .collect::<Result<_, GraphError>>()?;
        packages.insert(
            key,
            PathPackageInput {
                package: canonical_package(package)?,
                local_name: package.manifest.package.name.clone(),
                workspace_path,
                manifest_digest: ContentDigest::sha256(canonical_manifest),
                tree_digest: source_tree_digest(&package.sources)?,
                version: domain.version,
                dependencies,
                required_capabilities: domain.required_capabilities,
                singleton: domain.singleton,
            },
        );
    }
    Ok((
        PathGraphInput {
            root: package_keys[&project.root_manifest].clone(),
            packages,
            host_capabilities,
        },
        package_keys,
    ))
}

fn build_catalog(
    project: LoadedPathProject,
    graph: PackageGraph,
    package_keys: &BTreeMap<PathBuf, String>,
) -> Result<FrozenProject, PathProjectError> {
    let inputs = project
        .packages
        .values()
        .map(|package| {
            package_keys
                .get(&package.manifest_path)
                .ok_or_else(|| PathProjectError::MissingInstance(package.manifest_path.clone()))?;
            let expected_path = workspace_path(&project.workspace_root, &package.project_root)
                .map_err(|error| PathProjectError::Invalid(error.to_string()))?;
            let graph_package = graph
                .packages
                .values()
                .find(|candidate| {
                    candidate.local_name == package.manifest.package.name
                        && matches!(
                            &candidate.instance.source,
                            SourceId::Path(path) if path.workspace_path == expected_path
                        )
                })
                .ok_or_else(|| PathProjectError::MissingInstance(package.manifest_path.clone()))?;
            Ok(FrozenPackageInput {
                instance: graph_package.instance.identity_digest.clone(),
                local_name: package.manifest.package.name.clone(),
                source: graph_package.instance.source.clone(),
                root: package.project_root.clone(),
                files: package
                    .sources
                    .iter()
                    .map(|source| FrozenSourceInput {
                        descriptor: source.descriptor.clone(),
                        bytes: source.bytes.clone(),
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, PathProjectError>>()?;
    assemble_frozen_project(project.root_manifest, project.workspace_root, graph, inputs)
        .map_err(|error| PathProjectError::Invalid(error.to_string()))
}

fn stable_package_key(workspace_root: &Path, manifest: &Path) -> Result<String, GraphError> {
    manifest
        .strip_prefix(workspace_root)
        .map_err(|_| {
            GraphError::Invalid(format!(
                "manifest {} is outside workspace {}",
                manifest.display(),
                workspace_root.display()
            ))
        })
        .and_then(|path| {
            NormalizedRelativePath::new(path)
                .map(|path| path.as_str().to_string())
                .map_err(|error| GraphError::Invalid(error.to_string()))
        })
}

fn workspace_path(
    workspace_root: &Path,
    project_root: &Path,
) -> Result<NormalizedRelativePath, GraphError> {
    let relative = project_root.strip_prefix(workspace_root).map_err(|_| {
        GraphError::Invalid(format!(
            "package root {} is outside workspace {}",
            project_root.display(),
            workspace_root.display()
        ))
    })?;
    NormalizedRelativePath::new(relative).map_err(|error| GraphError::Invalid(error.to_string()))
}

fn canonical_package(package: &LoadedPathPackage) -> Result<CanonicalPackageId, GraphError> {
    let declaration = &package.manifest.package;
    if let Some(organization) = declaration.organization.as_deref() {
        return CanonicalPackageId::new(
            declaration
                .registry
                .as_deref()
                .unwrap_or("default")
                .parse::<RegistryId>()
                .map_err(|error| GraphError::Invalid(error.to_string()))?,
            organization,
            &declaration.name,
        )
        .map_err(|error| GraphError::Invalid(error.to_string()));
    }
    CanonicalPackageId::new("workspace".parse().unwrap(), "local", &declaration.name)
        .map_err(|error| GraphError::Invalid(error.to_string()))
}

fn source_tree_digest(sources: &[LoadedSource]) -> Result<ContentDigest, GraphError> {
    let mut input = Vec::new();
    for source in sources {
        append_tree_entry(&mut input, &source.descriptor, &source.bytes)?;
    }
    Ok(ContentDigest::sha256(input))
}

fn append_tree_entry(
    output: &mut Vec<u8>,
    source: &ProjectSourceFile,
    bytes: &[u8],
) -> Result<(), GraphError> {
    let path = NormalizedRelativePath::new(source.source_root.join(&source.relative_path))
        .map_err(|error| GraphError::Invalid(error.to_string()))?;
    output.extend_from_slice(path.as_str().as_bytes());
    output.push(0);
    output.extend_from_slice(bytes.len().to_string().as_bytes());
    output.push(0);
    output.extend_from_slice(bytes);
    output.push(0);
    Ok(())
}

fn validate_path_versions(project: &LoadedPathProject) -> Result<(), PathProjectError> {
    for package in project.packages.values() {
        for (alias, target_manifest) in &package.dependencies {
            let dependency = package.manifest.dependencies.get(alias).ok_or_else(|| {
                PathProjectError::Invalid(format!(
                    "dependency edge `{alias}` is absent from package `{}` manifest",
                    package.manifest.package.name
                ))
            })?;
            let Some(requirement) = dependency.version.as_deref() else {
                continue;
            };
            let requirement = VersionReq::parse(requirement).map_err(|error| {
                PathProjectError::Invalid(format!(
                    "dependency `{alias}` of `{}` has invalid version requirement `{requirement}`: {error}",
                    package.manifest.package.name
                ))
            })?;
            let target = project.packages.get(target_manifest).ok_or_else(|| {
                PathProjectError::Invalid(format!(
                    "dependency `{alias}` references unloaded manifest {}",
                    target_manifest.display()
                ))
            })?;
            let version_text = target.manifest.package.version.as_deref().ok_or_else(|| {
                PathProjectError::Invalid(format!(
                    "path dependency `{alias}` requires {requirement}, but package `{}` has no version",
                    target.manifest.package.name
                ))
            })?;
            let version = Version::parse(version_text).map_err(|error| {
                PathProjectError::Invalid(format!(
                    "path dependency `{alias}` has invalid declared version `{version_text}`: {error}"
                ))
            })?;
            if !requirement.matches(&version) {
                return Err(PathProjectError::Invalid(format!(
                    "path dependency `{alias}` of `{}` requires {requirement}, but package `{}` declares {version}",
                    package.manifest.package.name, target.manifest.package.name
                )));
            }
        }
    }
    Ok(())
}
