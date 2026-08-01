use super::composition::{
    discover_project_composition_from, discover_project_composition_from_async,
    DiscoverProjectCompositionError, DiscoveredProjectComposition, ProjectCompositionError,
};
use super::manifest::{path_exists_async, path_is_file_async};
use super::source_index::{
    build_loose_source_index, build_loose_source_index_async, ProjectSourceFile,
    ProjectSourceIndex, ProjectSourceIndexError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProjectSymbols {
    pub manifest_path: PathBuf,
    pub root_package: String,
    pub project_root: PathBuf,
    pub symbols: HashSet<String>,
    pub definitions: Vec<ProjectSymbolDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSymbolDefinition {
    pub name: String,
    pub qualified_name: String,
    pub source_path: PathBuf,
    pub package_name: String,
    pub is_private: bool,
}

#[derive(Debug, Error)]
pub enum DiscoverProjectSymbolsError {
    #[error(
        "failed to build project composition from discovered manifest {manifest_path}: {source}"
    )]
    Composition {
        manifest_path: PathBuf,
        #[source]
        source: Box<ProjectCompositionError>,
    },
    #[error("project composition for {manifest_path} is missing root package `{package}`")]
    MissingRootPackage {
        manifest_path: PathBuf,
        package: String,
    },
}

/// Source lookup context shared by static analysis, the LSP, and runtime
/// compilation for both manifest projects and loose MATLAB folders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredSourceSymbols {
    pub manifest_path: Option<PathBuf>,
    pub project_root: PathBuf,
    pub symbols: HashSet<String>,
    pub definitions: Vec<ProjectSymbolDefinition>,
}

#[derive(Debug, Error)]
pub enum DiscoverSourceSymbolsError {
    #[error(transparent)]
    Project(#[from] DiscoverProjectSymbolsError),
    #[error("failed to index loose MATLAB sources under {root}: {source}")]
    LooseSourceIndex {
        root: PathBuf,
        #[source]
        source: ProjectSourceIndexError,
    },
}

pub fn discover_project_symbols_from(
    start: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    discover_project_composition_from(start)
        .map_err(map_composition_error)?
        .map(project_symbols_from_composition)
        .transpose()
}

pub async fn discover_project_symbols_from_async(
    start: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    discover_project_composition_from_async(start)
        .await
        .map_err(map_composition_error)?
        .map(project_symbols_from_composition)
        .transpose()
}

fn project_symbols_from_composition(
    discovered: DiscoveredProjectComposition,
) -> Result<DiscoveredProjectSymbols, DiscoverProjectSymbolsError> {
    let manifest_path = discovered.manifest_path;
    let root_package = discovered.root_package;
    let root = discovered
        .composition
        .packages
        .get(&root_package)
        .ok_or_else(|| DiscoverProjectSymbolsError::MissingRootPackage {
            manifest_path: manifest_path.clone(),
            package: root_package.clone(),
        })?;
    let project_root = root.project_root.clone();
    let root_dependencies = root.dependencies.clone();
    let mut symbols = HashSet::new();
    let mut definitions = Vec::new();
    for package in discovered.composition.packages.values() {
        for source in &package.source_index.files {
            extend_project_source_symbols(
                &mut symbols,
                &mut definitions,
                source,
                &package.package_name,
                &package.project_root,
                &root_dependencies,
            );
        }
    }
    Ok(DiscoveredProjectSymbols {
        manifest_path,
        root_package,
        project_root,
        symbols,
        definitions,
    })
}

fn map_composition_error(error: DiscoverProjectCompositionError) -> DiscoverProjectSymbolsError {
    match error {
        DiscoverProjectCompositionError::Composition {
            manifest_path,
            source,
        } => DiscoverProjectSymbolsError::Composition {
            manifest_path,
            source,
        },
        DiscoverProjectCompositionError::MissingRootPackage {
            manifest_path,
            package,
        } => DiscoverProjectSymbolsError::MissingRootPackage {
            manifest_path,
            package,
        },
    }
}

pub fn discover_project_symbols_from_source_name(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    let Some(start) = symbol_discovery_start(
        source_name,
        cwd,
        |path| path.exists(),
        |path| path.is_file(),
    ) else {
        return Ok(None);
    };
    discover_project_symbols_from(&start)
}

pub async fn discover_project_symbols_from_source_name_async(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    let source_path = PathBuf::from(source_name);
    let local_candidate = absolute_or_join(cwd, &source_path);
    if source_name.contains(':') && !path_exists_async(&local_candidate).await {
        return Ok(None);
    }
    if (source_path.is_absolute() || source_path.components().count() > 1)
        && !path_exists_async(&local_candidate).await
    {
        return Ok(None);
    }
    let start = if path_is_file_async(&local_candidate).await {
        local_candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.components().count() > 1 {
        local_candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else {
        cwd.to_path_buf()
    };
    discover_project_symbols_from_async(&start).await
}

fn symbol_discovery_start(
    source_name: &str,
    cwd: &Path,
    exists: impl Fn(&Path) -> bool,
    is_file: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let source_path = PathBuf::from(source_name);
    let local_candidate = absolute_or_join(cwd, &source_path);
    if source_name.contains(':') && !exists(&local_candidate) {
        return None;
    }
    if (source_path.is_absolute() || source_path.components().count() > 1)
        && !exists(&local_candidate)
    {
        return None;
    }
    Some(if is_file(&local_candidate) {
        local_candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.components().count() > 1 {
        local_candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else {
        cwd.to_path_buf()
    })
}

pub fn discover_source_symbols_from_source_name(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredSourceSymbols>, DiscoverSourceSymbolsError> {
    let Some((source_path, root)) = local_source_and_parent(source_name, cwd) else {
        return Ok(None);
    };
    if let Some(project) = discover_project_symbols_from_source_name(source_name, cwd)? {
        return Ok(Some(source_symbols_from_project(project, &source_path)));
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
    if let Some(project) = discover_project_symbols_from_source_name_async(source_name, cwd).await?
    {
        return Ok(Some(source_symbols_from_project(project, &source_path)));
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

fn source_symbols_from_project(
    project: DiscoveredProjectSymbols,
    source_path: &Path,
) -> DiscoveredSourceSymbols {
    let mut symbols = project.symbols;
    add_visible_private_symbols(&mut symbols, &project.definitions, source_path);
    DiscoveredSourceSymbols {
        manifest_path: Some(project.manifest_path),
        project_root: project.project_root,
        symbols,
        definitions: project.definitions,
    }
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
        extend_project_source_symbols(
            &mut symbols,
            &mut definitions,
            source,
            "",
            root,
            &BTreeMap::new(),
        );
    }
    add_visible_private_symbols(&mut symbols, &definitions, primary_source);
    DiscoveredSourceSymbols {
        manifest_path,
        project_root: root.to_path_buf(),
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

fn extend_project_source_symbols(
    symbols: &mut HashSet<String>,
    definitions: &mut Vec<ProjectSymbolDefinition>,
    source: &ProjectSourceFile,
    package_name: &str,
    package_root: &Path,
    root_dependencies: &BTreeMap<String, String>,
) {
    let mut names = vec![source.qualified_name.as_str()];
    if let Some(class_name) = source.class_definition_qualified_name() {
        if class_name != source.qualified_name {
            names.push(class_name);
        }
    }
    let source_path = package_root
        .join(&source.source_root)
        .join(&source.relative_path);
    for name in names {
        let package_name_variant =
            (!package_name.is_empty()).then(|| format!("{package_name}.{name}"));
        let exposed_names = std::iter::once(name.to_string())
            .chain(package_name_variant)
            .chain(
                root_dependencies
                    .iter()
                    .filter(|(_, dependency_package)| *dependency_package == package_name)
                    .map(|(alias, _)| format!("{alias}.{name}")),
            );
        for exposed_name in exposed_names {
            if !definitions.iter().any(|definition| {
                definition.name == exposed_name && definition.source_path == source_path
            }) {
                definitions.push(ProjectSymbolDefinition {
                    name: exposed_name.clone(),
                    qualified_name: source.qualified_name.clone(),
                    source_path: source_path.clone(),
                    package_name: package_name.to_string(),
                    is_private: source.is_private,
                });
            }
            if !source.is_private {
                symbols.insert(exposed_name);
            }
        }
    }
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
    let parent = local.parent()?.to_path_buf();
    Some((local, parent))
}

async fn local_source_and_parent_async(
    source_name: &str,
    cwd: &Path,
) -> Option<(PathBuf, PathBuf)> {
    let source_path = PathBuf::from(source_name);
    let local = absolute_or_join(cwd, &source_path);
    if (source_name.contains(':') && !path_exists_async(&local).await)
        || !path_is_file_async(&local).await
    {
        return None;
    }
    let parent = local.parent()?.to_path_buf();
    Some((local, parent))
}

fn absolute_or_join(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}
