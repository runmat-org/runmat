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
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::source::{
    FrozenProject, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog, SourceCatalog,
    StableSourceId,
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
    _package_keys: &BTreeMap<PathBuf, String>,
) -> Result<FrozenProject, PathProjectError> {
    let mut packages = BTreeMap::new();
    let mut access_paths = BTreeMap::new();
    for package in project.packages.values() {
        let workspace_path = workspace_path(&project.workspace_root, &package.project_root)
            .map_err(|error| PathProjectError::Invalid(error.to_string()))?;
        let graph_package = graph
            .packages
            .values()
            .find(|candidate| {
                matches!(
                    &candidate.instance.source,
                    SourceId::Path(path) if path.workspace_path == workspace_path
                )
            })
            .ok_or_else(|| PathProjectError::MissingInstance(package.manifest_path.clone()))?;
        let package_instance = graph_package.instance.identity_digest.clone();
        let mut sources = Vec::with_capacity(package.sources.len());
        for source in &package.sources {
            let access_path = source_access_path(&package.project_root, &source.descriptor);
            let id = StableSourceId {
                package_instance: package_instance.clone(),
                relative_path: source_relative_path(&source.descriptor)?,
                content_digest: ContentDigest::sha256(&source.bytes),
            };
            access_paths.insert(id.clone(), access_path);
            sources.push(FrozenSourceDescriptor {
                id,
                qualified_name: source.descriptor.qualified_name.clone(),
                package_path: source.descriptor.package_path.clone(),
                class_name: source.descriptor.class_name.clone(),
                class_qualified_name: source.descriptor.class_qualified_name.clone(),
                is_private: source.descriptor.is_private,
            });
        }
        sources.sort_by(|left, right| left.id.cmp(&right.id));
        packages.insert(
            package_instance.clone(),
            PackageSourceCatalog {
                package_instance: package_instance.clone(),
                local_name: package.manifest.package.name.clone(),
                mount: PackageMount {
                    package_instance: package_instance.clone(),
                    source: graph_package.instance.source.clone(),
                    logical_root: logical_mount_root(&package_instance)?,
                },
                sources,
            },
        );
    }
    let revision = source_revision(&graph.graph_digest, &packages)?;
    Ok(FrozenProject {
        manifest_path: project.root_manifest,
        workspace_root: project.workspace_root,
        graph,
        sources: SourceCatalog { packages, revision },
        access_paths,
    })
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

fn source_access_path(project_root: &Path, source: &ProjectSourceFile) -> PathBuf {
    project_root
        .join(&source.source_root)
        .join(&source.relative_path)
}

fn source_relative_path(
    source: &ProjectSourceFile,
) -> Result<NormalizedRelativePath, PathProjectError> {
    let path = source.source_root.join(&source.relative_path);
    NormalizedRelativePath::new(&path).map_err(|error| PathProjectError::SourcePath {
        path,
        reason: error.to_string(),
    })
}

fn logical_mount_root(
    identity: &ContentDigest,
) -> Result<NormalizedRelativePath, PathProjectError> {
    let key = identity.to_string().replace(':', "_");
    NormalizedRelativePath::new(format!("packages/{key}")).map_err(|error| {
        PathProjectError::SourcePath {
            path: PathBuf::from(key),
            reason: error.to_string(),
        }
    })
}

fn source_revision(
    graph_digest: &ContentDigest,
    packages: &BTreeMap<ContentDigest, PackageSourceCatalog>,
) -> Result<ContentDigest, PathProjectError> {
    #[derive(Serialize)]
    struct Input<'a> {
        format: &'static str,
        graph_digest: &'a ContentDigest,
        packages: &'a BTreeMap<ContentDigest, PackageSourceCatalog>,
    }
    serde_json::to_vec(&Input {
        format: "runmat-source-catalog-v1",
        graph_digest,
        packages,
    })
    .map(ContentDigest::sha256)
    .map_err(|error| PathProjectError::Revision(error.to_string()))
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
