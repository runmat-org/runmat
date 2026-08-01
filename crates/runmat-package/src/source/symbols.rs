use super::{discover_frozen_project_from, discover_frozen_project_from_async, FrozenProject};
use crate::{ContentDigest, FrozenProjectError, StableSourceId};
use runmat_config::project::{
    build_loose_source_index, build_loose_source_index_async, ProjectSourceFile,
    ProjectSourceIndex, ProjectSourceIndexError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSymbolDefinition {
    pub name: String,
    pub qualified_name: String,
    pub source_path: PathBuf,
    pub package_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_instance: Option<ContentDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<StableSourceId>,
    pub is_private: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredSourceSymbols {
    pub manifest_path: Option<PathBuf>,
    pub project_root: PathBuf,
    pub graph_digest: Option<ContentDigest>,
    pub source_revision: Option<ContentDigest>,
    pub symbols: HashSet<String>,
    pub definitions: Vec<ProjectSymbolDefinition>,
}

impl DiscoveredSourceSymbols {
    pub fn project_revision(&self) -> Option<crate::ProjectRevision> {
        Some(crate::ProjectRevision {
            graph_digest: self.graph_digest.clone()?,
            source_revision: self.source_revision.clone()?,
        })
    }
}

#[derive(Debug, Error)]
pub enum DiscoverSourceSymbolsError {
    #[error(transparent)]
    Frozen(#[from] FrozenProjectError),
    #[error("failed to index loose MATLAB sources under {root}: {source}")]
    LooseSourceIndex {
        root: PathBuf,
        #[source]
        source: ProjectSourceIndexError,
    },
}

pub fn discover_source_symbols_from_source_name(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredSourceSymbols>, DiscoverSourceSymbolsError> {
    let Some((source_path, root)) = local_source_and_parent(source_name, cwd) else {
        return Ok(None);
    };
    if let Some(frozen) = discover_frozen_project_from(&source_path, BTreeSet::new())? {
        return Ok(Some(source_symbols_from_frozen(frozen, &source_path)));
    }
    let index = build_loose_source_index(&root).map_err(|source| {
        DiscoverSourceSymbolsError::LooseSourceIndex {
            root: root.clone(),
            source,
        }
    })?;
    Ok(Some(source_symbols_from_index(
        &index,
        &root,
        &source_path,
        None,
    )))
}

pub async fn discover_source_symbols_from_source_name_async(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredSourceSymbols>, DiscoverSourceSymbolsError> {
    let Some((source_path, root)) = local_source_and_parent_async(source_name, cwd).await else {
        return Ok(None);
    };
    if let Some(frozen) = discover_frozen_project_from_async(&source_path, BTreeSet::new()).await? {
        return Ok(Some(source_symbols_from_frozen(frozen, &source_path)));
    }
    let index = build_loose_source_index_async(&root)
        .await
        .map_err(|source| DiscoverSourceSymbolsError::LooseSourceIndex {
            root: root.clone(),
            source,
        })?;
    Ok(Some(source_symbols_from_index(
        &index,
        &root,
        &source_path,
        None,
    )))
}

pub fn source_symbols_from_index(
    index: &ProjectSourceIndex,
    root: &Path,
    primary_source: &Path,
    manifest_path: Option<PathBuf>,
) -> DiscoveredSourceSymbols {
    let mut symbols = HashSet::new();
    let mut definitions = Vec::new();
    for source in &index.files {
        extend_loose_source(&mut symbols, &mut definitions, source, root);
    }
    add_visible_private_symbols(&mut symbols, &definitions, primary_source);
    DiscoveredSourceSymbols {
        manifest_path,
        project_root: root.to_path_buf(),
        graph_digest: None,
        source_revision: None,
        symbols,
        definitions,
    }
}

pub fn discover_known_project_symbols_from_source_name(
    source_name: Option<&str>,
    cwd: &Path,
) -> HashSet<String> {
    source_name
        .and_then(|source_name| {
            discover_source_symbols_from_source_name(source_name, cwd)
                .ok()
                .flatten()
        })
        .map(|discovered| discovered.symbols)
        .unwrap_or_default()
}

pub async fn discover_known_project_symbols_from_source_name_async(
    source_name: Option<&str>,
    cwd: &Path,
) -> HashSet<String> {
    let Some(source_name) = source_name else {
        return HashSet::new();
    };
    discover_source_symbols_from_source_name_async(source_name, cwd)
        .await
        .ok()
        .flatten()
        .map(|discovered| discovered.symbols)
        .unwrap_or_default()
}

fn source_symbols_from_frozen(
    frozen: FrozenProject,
    primary_source: &Path,
) -> DiscoveredSourceSymbols {
    let owner = frozen
        .all_sources()
        .find(|(_, path)| paths_equivalent(path, primary_source))
        .map(|(source, _)| source.id.package_instance.clone())
        .unwrap_or_else(|| frozen.graph.root.clone());
    let visible_edges = frozen
        .graph
        .edges
        .iter()
        .filter(|edge| edge.from == owner)
        .map(|edge| (edge.to.clone(), edge.alias.to_string()))
        .collect::<BTreeMap<_, _>>();
    let mut symbols = HashSet::new();
    let mut definitions = Vec::new();
    let mut unqualified_candidates =
        BTreeMap::<String, BTreeMap<ContentDigest, ProjectSymbolDefinition>>::new();
    for (instance, package) in &frozen.sources.packages {
        let is_owner = instance == &owner;
        let alias = visible_edges.get(instance);
        if !is_owner && alias.is_none() {
            continue;
        }
        for source in &package.sources {
            let source_path = frozen.access_paths[&source.id].clone();
            let names = source_names(
                &source.qualified_name,
                source.class_definition_qualified_name(),
            );
            for name in names {
                if is_owner {
                    push_definition(
                        &mut definitions,
                        name.clone(),
                        source,
                        &source_path,
                        &package.local_name,
                    );
                    if !source.is_private {
                        symbols.insert(name.clone());
                    }
                }
                if let Some(alias) = alias {
                    let exposed = format!("{alias}.{name}");
                    push_definition(
                        &mut definitions,
                        exposed.clone(),
                        source,
                        &source_path,
                        &package.local_name,
                    );
                    if !source.is_private {
                        symbols.insert(exposed);
                        unqualified_candidates.entry(name).or_default().insert(
                            instance.clone(),
                            project_definition(
                                source.qualified_name.clone(),
                                source,
                                &source_path,
                                &package.local_name,
                            ),
                        );
                    }
                }
            }
        }
    }
    for (name, candidates) in unqualified_candidates {
        if candidates.len() == 1 && !symbols.contains(&name) {
            let mut definition = candidates
                .into_values()
                .next()
                .expect("one unqualified candidate");
            definition.name = name.clone();
            definitions.push(definition);
            symbols.insert(name);
        }
    }
    add_visible_private_symbols(&mut symbols, &definitions, primary_source);
    DiscoveredSourceSymbols {
        manifest_path: Some(frozen.manifest_path),
        project_root: frozen.workspace_root,
        graph_digest: Some(frozen.graph.graph_digest),
        source_revision: Some(frozen.sources.revision),
        symbols,
        definitions,
    }
}

fn push_definition(
    definitions: &mut Vec<ProjectSymbolDefinition>,
    name: String,
    source: &crate::FrozenSourceDescriptor,
    source_path: &Path,
    package_name: &str,
) {
    if definitions
        .iter()
        .any(|definition| definition.name == name && definition.source_path == source_path)
    {
        return;
    }
    definitions.push(project_definition(name, source, source_path, package_name));
}

fn project_definition(
    name: String,
    source: &crate::FrozenSourceDescriptor,
    source_path: &Path,
    package_name: &str,
) -> ProjectSymbolDefinition {
    ProjectSymbolDefinition {
        name,
        qualified_name: source.qualified_name.clone(),
        source_path: source_path.to_path_buf(),
        package_name: package_name.to_string(),
        package_instance: Some(source.id.package_instance.clone()),
        source_id: Some(source.id.clone()),
        is_private: source.is_private,
    }
}

fn extend_loose_source(
    symbols: &mut HashSet<String>,
    definitions: &mut Vec<ProjectSymbolDefinition>,
    source: &ProjectSourceFile,
    root: &Path,
) {
    let source_path = root.join(&source.source_root).join(&source.relative_path);
    for name in source_names(
        &source.qualified_name,
        source.class_definition_qualified_name(),
    ) {
        definitions.push(ProjectSymbolDefinition {
            name: name.clone(),
            qualified_name: source.qualified_name.clone(),
            source_path: source_path.clone(),
            package_name: String::new(),
            package_instance: None,
            source_id: None,
            is_private: source.is_private,
        });
        if !source.is_private {
            symbols.insert(name);
        }
    }
}

fn source_names(qualified_name: &str, class_name: Option<&str>) -> Vec<String> {
    let mut names = vec![qualified_name.to_string()];
    if let Some(class_name) = class_name {
        if class_name != qualified_name {
            names.push(class_name.to_string());
        }
    }
    names
}

fn add_visible_private_symbols(
    symbols: &mut HashSet<String>,
    definitions: &[ProjectSymbolDefinition],
    primary_source: &Path,
) {
    let primary_parent = primary_source.parent();
    for definition in definitions
        .iter()
        .filter(|definition| definition.is_private)
    {
        let private_owner = definition.source_path.parent().and_then(Path::parent);
        if private_owner.is_some() && private_owner == primary_parent {
            symbols.insert(definition.name.clone());
        }
    }
}

fn local_source_and_parent(source_name: &str, cwd: &Path) -> Option<(PathBuf, PathBuf)> {
    let source_path = PathBuf::from(source_name);
    let local = absolute_or_join(cwd, &source_path);
    if (source_name.contains(':') && !local.exists()) || !local.is_file() {
        return None;
    }
    Some((local.clone(), local.parent()?.to_path_buf()))
}

async fn local_source_and_parent_async(
    source_name: &str,
    cwd: &Path,
) -> Option<(PathBuf, PathBuf)> {
    let source_path = PathBuf::from(source_name);
    let local = absolute_or_join(cwd, &source_path);
    if (source_name.contains(':') && runmat_filesystem::metadata_async(&local).await.is_err())
        || !runmat_filesystem::metadata_async(&local)
            .await
            .is_ok_and(|metadata| metadata.is_file())
    {
        return None;
    }
    Some((local.clone(), local.parent()?.to_path_buf()))
}

fn absolute_or_join(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
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
