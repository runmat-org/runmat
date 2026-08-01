use crate::{ContentDigest, NormalizedRelativePath, PackageAlias, PackageGraph, SourceId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StableSourceId {
    pub package_instance: ContentDigest,
    pub relative_path: NormalizedRelativePath,
    pub content_digest: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenSourceDescriptor {
    pub id: StableSourceId,
    pub qualified_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_qualified_name: Option<String>,
    pub is_private: bool,
}

impl FrozenSourceDescriptor {
    pub fn class_definition_qualified_name(&self) -> Option<&str> {
        self.class_qualified_name.as_deref().or_else(|| {
            self.package_path
                .as_ref()
                .map(|_| self.qualified_name.as_str())
        })
    }

    pub fn function_qualified_name(&self) -> Option<&str> {
        if self.is_private {
            return None;
        }
        (self.package_path.is_some() || self.class_name.is_some())
            .then_some(self.qualified_name.as_str())
            .filter(|name| name.contains('.'))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageMount {
    pub package_instance: ContentDigest,
    pub source: SourceId,
    pub logical_root: NormalizedRelativePath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageSourceCatalog {
    pub package_instance: ContentDigest,
    pub local_name: String,
    pub mount: PackageMount,
    pub sources: Vec<FrozenSourceDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCatalog {
    pub packages: BTreeMap<ContentDigest, PackageSourceCatalog>,
    pub revision: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectRevision {
    pub graph_digest: ContentDigest,
    pub source_revision: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenProject {
    pub manifest_path: PathBuf,
    pub workspace_root: PathBuf,
    pub graph: PackageGraph,
    pub sources: SourceCatalog,
    pub access_paths: BTreeMap<StableSourceId, PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub struct VisibleProjectSource<'a> {
    pub package: &'a PackageSourceCatalog,
    pub source: &'a FrozenSourceDescriptor,
    pub access_path: &'a Path,
    pub dependency_alias: Option<&'a PackageAlias>,
    pub directly_visible: bool,
}

impl FrozenProject {
    pub fn source_revision(&self) -> &ContentDigest {
        &self.sources.revision
    }

    pub fn graph_digest(&self) -> &ContentDigest {
        &self.graph.graph_digest
    }

    pub fn revision(&self) -> ProjectRevision {
        ProjectRevision {
            graph_digest: self.graph.graph_digest.clone(),
            source_revision: self.sources.revision.clone(),
        }
    }

    pub fn all_sources(&self) -> impl Iterator<Item = (&FrozenSourceDescriptor, &PathBuf)> {
        self.sources
            .packages
            .values()
            .flat_map(|package| package.sources.iter())
            .filter_map(|source| self.access_paths.get(&source.id).map(|path| (source, path)))
    }

    pub fn requester_instance(&self, requester_path: &Path) -> &ContentDigest {
        self.all_sources()
            .find(|(_, path)| paths_equivalent(path, requester_path))
            .map(|(source, _)| &source.id.package_instance)
            .unwrap_or(&self.graph.root)
    }

    pub fn visible_sources(&self, requester_path: &Path) -> Vec<VisibleProjectSource<'_>> {
        let requester = self.requester_instance(requester_path);
        let mut visible = Vec::new();
        if let Some(package) = self.sources.packages.get(requester) {
            for source in &package.sources {
                let Some(access_path) = self.access_paths.get(&source.id) else {
                    continue;
                };
                visible.push(VisibleProjectSource {
                    package,
                    source,
                    access_path,
                    dependency_alias: None,
                    directly_visible: !source.is_private
                        || private_source_visible(requester_path, access_path),
                });
            }
        }
        for edge in self
            .graph
            .edges
            .iter()
            .filter(|edge| &edge.from == requester)
        {
            let Some(package) = self.sources.packages.get(&edge.to) else {
                continue;
            };
            for source in package.sources.iter().filter(|source| !source.is_private) {
                if let Some(access_path) = self.access_paths.get(&source.id) {
                    visible.push(VisibleProjectSource {
                        package,
                        source,
                        access_path,
                        dependency_alias: Some(&edge.alias),
                        directly_visible: true,
                    });
                }
            }
        }
        visible
    }
}

fn private_source_visible(requester_path: &Path, private_source: &Path) -> bool {
    private_source
        .parent()
        .and_then(Path::parent)
        .zip(requester_path.parent())
        .is_some_and(|(owner, requester)| paths_equivalent(owner, requester))
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}
