use super::{
    FrozenProject, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog, SourceCatalog,
    StableSourceId,
};
use crate::{
    build_project_path_graph, build_project_path_graph_async, ContentDigest, GraphError,
    HostCapability, NormalizedRelativePath,
};
use runmat_config::project::{
    build_project_composition_graph, build_project_composition_graph_async,
    discover_project_manifest_from, discover_project_manifest_from_async, ProjectCompositionError,
    ProjectCompositionGraph, ProjectSourceFile,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrozenProjectError {
    #[error("failed to load project composition: {0}")]
    Composition(#[from] ProjectCompositionError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("project composition is missing root package `{0}`")]
    MissingRoot(String),
    #[error("package graph has no instance for local package `{0}`")]
    MissingInstance(String),
    #[error("failed to read graph-declared source {path}: {reason}")]
    ReadSource { path: PathBuf, reason: String },
    #[error("invalid source path {path}: {reason}")]
    SourcePath { path: PathBuf, reason: String },
    #[error("failed to encode source revision: {0}")]
    Revision(String),
}

pub fn discover_frozen_project_from(
    start: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<Option<FrozenProject>, FrozenProjectError> {
    let Some(manifest_path) = discover_project_manifest_from(start) else {
        return Ok(None);
    };
    build_frozen_project(&manifest_path, host_capabilities).map(Some)
}

pub async fn discover_frozen_project_from_async(
    start: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<Option<FrozenProject>, FrozenProjectError> {
    let Some(manifest_path) = discover_project_manifest_from_async(start).await else {
        return Ok(None);
    };
    build_frozen_project_async(&manifest_path, host_capabilities)
        .await
        .map(Some)
}

pub fn build_frozen_project(
    manifest_path: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<FrozenProject, FrozenProjectError> {
    let composition = build_project_composition_graph(manifest_path)?;
    let workspace_root = root_path(&composition)?;
    let graph = build_project_path_graph(&composition, &workspace_root, host_capabilities)?;
    build_catalog(manifest_path, workspace_root, graph, &composition, |path| {
        std::fs::read(path).map_err(|error| FrozenProjectError::ReadSource {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })
    })
}

pub async fn build_frozen_project_async(
    manifest_path: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<FrozenProject, FrozenProjectError> {
    let composition = build_project_composition_graph_async(manifest_path).await?;
    let workspace_root = root_path(&composition)?;
    let graph =
        build_project_path_graph_async(&composition, &workspace_root, host_capabilities).await?;
    let mut contents = BTreeMap::new();
    for package in composition.packages.values() {
        for source in &package.source_index.files {
            let path = source_access_path(&package.project_root, source);
            let bytes = runmat_filesystem::read_async(&path)
                .await
                .map_err(|error| FrozenProjectError::ReadSource {
                    path: path.clone(),
                    reason: error.to_string(),
                })?;
            contents.insert(path, bytes);
        }
    }
    build_catalog(manifest_path, workspace_root, graph, &composition, |path| {
        contents
            .remove(path)
            .ok_or_else(|| FrozenProjectError::ReadSource {
                path: path.to_path_buf(),
                reason: "source disappeared from frozen read set".to_string(),
            })
    })
}

fn build_catalog(
    manifest_path: &Path,
    workspace_root: PathBuf,
    graph: crate::PackageGraph,
    composition: &ProjectCompositionGraph,
    mut read: impl FnMut(&Path) -> Result<Vec<u8>, FrozenProjectError>,
) -> Result<FrozenProject, FrozenProjectError> {
    let mut packages = BTreeMap::new();
    let mut access_paths = BTreeMap::new();
    for package in composition.packages.values() {
        let graph_package = graph
            .packages
            .values()
            .find(|candidate| candidate.local_name == package.package_name)
            .ok_or_else(|| FrozenProjectError::MissingInstance(package.package_name.clone()))?;
        let identity = graph_package.instance.identity_digest.clone();
        let mut sources = Vec::new();
        for source in &package.source_index.files {
            let path = source_access_path(&package.project_root, source);
            let bytes = read(&path)?;
            let relative_path = source_relative_path(source)?;
            let id = StableSourceId {
                package_instance: identity.clone(),
                relative_path,
                content_digest: ContentDigest::sha256(bytes),
            };
            access_paths.insert(id.clone(), path);
            sources.push(FrozenSourceDescriptor {
                id,
                qualified_name: source.qualified_name.clone(),
                package_path: source.package_path.clone(),
                class_name: source.class_name.clone(),
                class_qualified_name: source.class_qualified_name.clone(),
                is_private: source.is_private,
            });
        }
        sources.sort_by(|left, right| left.id.cmp(&right.id));
        let logical_root = logical_mount_root(&identity)?;
        packages.insert(
            identity.clone(),
            PackageSourceCatalog {
                package_instance: identity.clone(),
                local_name: package.package_name.clone(),
                mount: PackageMount {
                    package_instance: identity,
                    source: graph_package.instance.source.clone(),
                    logical_root,
                },
                sources,
            },
        );
    }
    let revision = source_revision(&graph.graph_digest, &packages)?;
    Ok(FrozenProject {
        manifest_path: manifest_path.to_path_buf(),
        workspace_root,
        graph,
        sources: SourceCatalog { packages, revision },
        access_paths,
    })
}

fn root_path(composition: &ProjectCompositionGraph) -> Result<PathBuf, FrozenProjectError> {
    composition
        .packages
        .get(&composition.root_package)
        .map(|package| package.project_root.clone())
        .ok_or_else(|| FrozenProjectError::MissingRoot(composition.root_package.clone()))
}

fn source_access_path(project_root: &Path, source: &ProjectSourceFile) -> PathBuf {
    project_root
        .join(&source.source_root)
        .join(&source.relative_path)
}

fn source_relative_path(
    source: &ProjectSourceFile,
) -> Result<NormalizedRelativePath, FrozenProjectError> {
    let path = source.source_root.join(&source.relative_path);
    NormalizedRelativePath::new(&path).map_err(|error| FrozenProjectError::SourcePath {
        path,
        reason: error.to_string(),
    })
}

fn logical_mount_root(
    identity: &ContentDigest,
) -> Result<NormalizedRelativePath, FrozenProjectError> {
    let key = identity.to_string().replace(':', "_");
    NormalizedRelativePath::new(format!("packages/{key}")).map_err(|error| {
        FrozenProjectError::SourcePath {
            path: PathBuf::from(key),
            reason: error.to_string(),
        }
    })
}

fn source_revision(
    graph_digest: &ContentDigest,
    packages: &BTreeMap<ContentDigest, PackageSourceCatalog>,
) -> Result<ContentDigest, FrozenProjectError> {
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
    .map_err(|error| FrozenProjectError::Revision(error.to_string()))
}
