use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

pub const PROJECT_MANIFEST_FILENAME: &str = "runmat.toml";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub package: ProjectPackage,
    pub sources: ProjectSources,
    #[serde(default)]
    pub dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default)]
    pub entrypoints: Vec<ProjectEntrypoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectPackage {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSources {
    #[serde(default)]
    pub roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawProjectDependency")]
pub struct ProjectDependency {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct RawProjectDependency {
    path: Option<PathBuf>,
    #[serde(flatten)]
    other: BTreeMap<String, toml::Value>,
}

impl TryFrom<RawProjectDependency> for ProjectDependency {
    type Error = String;

    fn try_from(value: RawProjectDependency) -> Result<Self, Self::Error> {
        if !value.other.is_empty() {
            let fields = value.other.keys().cloned().collect::<Vec<_>>().join(", ");
            return Err(format!(
                "unsupported dependency fields: {fields} (only `path` is currently supported)"
            ));
        }
        let Some(path) = value.path else {
            return Err("dependency is missing required `path` field".to_string());
        };
        Ok(Self { path })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectEntrypoint {
    pub name: String,
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub module: Option<String>,
    #[serde(default)]
    pub function: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectManifestValidationError {
    pub messages: Vec<String>,
}

impl Display for ProjectManifestValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "project manifest validation failed:\n- {}",
            self.messages.join("\n- ")
        )
    }
}

impl std::error::Error for ProjectManifestValidationError {}

#[derive(Debug, Error)]
pub enum ProjectManifestLoadError {
    #[error("failed to read project manifest {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse project manifest {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid project manifest {path}: {source}")]
    Validation {
        path: PathBuf,
        #[source]
        source: ProjectManifestValidationError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProjectSourceIndex {
    pub files: Vec<ProjectSourceFile>,
    pub package_dirs: Vec<PathBuf>,
    pub class_dirs: Vec<PathBuf>,
    pub private_dirs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSourceFile {
    pub source_root: PathBuf,
    pub relative_path: PathBuf,
    pub qualified_name: String,
    #[serde(default)]
    pub package_path: Option<String>,
    #[serde(default)]
    pub class_name: Option<String>,
    pub is_private: bool,
}

#[derive(Debug, Error)]
pub enum ProjectSourceIndexError {
    #[error("source root does not exist or is not a directory: {root}")]
    InvalidSourceRoot { root: PathBuf },
    #[error("failed to read source path {path}: {source}")]
    ReadDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read source entry under {path}: {source}")]
    ReadEntry {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProjectEntrypoint {
    pub name: String,
    pub source_file: PathBuf,
    pub module: Option<String>,
    pub function: Option<String>,
    pub target: ResolvedEntrypointTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedEntrypointTarget {
    Path,
    ModuleFunction,
}

#[derive(Debug, Error)]
pub enum ProjectEntrypointResolveError {
    #[error("entrypoint `{entrypoint}` path target `{path}` did not resolve to an existing file")]
    MissingPathTarget { entrypoint: String, path: PathBuf },
    #[error("entrypoint `{entrypoint}` module/function target `{module}.{function}` did not resolve under configured source roots")]
    MissingModuleTarget {
        entrypoint: String,
        module: String,
        function: String,
    },
    #[error("failed to resolve entrypoint `{entrypoint}` via project source index: {source}")]
    SourceIndex {
        entrypoint: String,
        #[source]
        source: ProjectSourceIndexError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectCompositionGraph {
    pub root_package: String,
    pub packages: BTreeMap<String, ProjectCompositionPackage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectCompositionPackage {
    pub package_name: String,
    pub manifest_path: PathBuf,
    pub project_root: PathBuf,
    pub manifest: ProjectManifest,
    pub source_index: ProjectSourceIndex,
    pub dependencies: BTreeMap<String, String>,
}

#[derive(Debug, Error)]
pub enum ProjectCompositionError {
    #[error("failed to load root project manifest {path}: {source}")]
    RootManifestLoad {
        path: PathBuf,
        #[source]
        source: ProjectManifestLoadError,
    },
    #[error("dependency `{dependency}` in package `{package}` points to missing manifest {path}")]
    MissingDependencyManifest {
        package: String,
        dependency: String,
        path: PathBuf,
    },
    #[error(
        "failed to load dependency manifest {path} for dependency `{dependency}` of package `{package}`: {source}"
    )]
    DependencyManifestLoad {
        package: String,
        dependency: String,
        path: PathBuf,
        #[source]
        source: ProjectManifestLoadError,
    },
    #[error("failed to build source index for package `{package}`: {source}")]
    SourceIndex {
        package: String,
        #[source]
        source: ProjectSourceIndexError,
    },
    #[error("duplicate package name `{package}` found in {first_manifest} and {second_manifest}")]
    DuplicatePackageName {
        package: String,
        first_manifest: PathBuf,
        second_manifest: PathBuf,
    },
    #[error("dependency cycle detected while loading project composition: {cycle}")]
    DependencyCycle { cycle: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProjectEntrypoint {
    pub manifest_path: PathBuf,
    pub root_package: String,
    pub project_root: PathBuf,
    pub entrypoint: ResolvedProjectEntrypoint,
}

#[derive(Debug, Error)]
pub enum DiscoverProjectEntrypointError {
    #[error(
        "failed to build project composition from discovered manifest {manifest_path}: {source}"
    )]
    Composition {
        manifest_path: PathBuf,
        #[source]
        source: ProjectCompositionError,
    },
    #[error("project composition for {manifest_path} is missing root package `{package}`")]
    MissingRootPackage {
        manifest_path: PathBuf,
        package: String,
    },
    #[error("failed to resolve project entrypoint `{entrypoint}` from {manifest_path}: {source}")]
    Resolve {
        manifest_path: PathBuf,
        entrypoint: String,
        #[source]
        source: ProjectEntrypointResolveError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProjectSymbols {
    pub manifest_path: PathBuf,
    pub root_package: String,
    pub project_root: PathBuf,
    pub symbols: HashSet<String>,
}

#[derive(Debug, Error)]
pub enum DiscoverProjectSymbolsError {
    #[error(
        "failed to build project composition from discovered manifest {manifest_path}: {source}"
    )]
    Composition {
        manifest_path: PathBuf,
        #[source]
        source: ProjectCompositionError,
    },
    #[error("project composition for {manifest_path} is missing root package `{package}`")]
    MissingRootPackage {
        manifest_path: PathBuf,
        package: String,
    },
}

#[derive(Debug, Error)]
pub enum ResolveProjectSourceInputError {
    #[error(
        "failed to resolve named project entrypoint `{entrypoint}` from working directory {cwd}: {source}"
    )]
    EntrypointResolve {
        cwd: PathBuf,
        entrypoint: String,
        #[source]
        source: DiscoverProjectEntrypointError,
    },
}

#[derive(Debug, Error)]
enum DiscoverProjectCompositionError {
    #[error(
        "failed to build project composition from discovered manifest {manifest_path}: {source}"
    )]
    Composition {
        manifest_path: PathBuf,
        #[source]
        source: ProjectCompositionError,
    },
    #[error("project composition for {manifest_path} is missing root package `{package}`")]
    MissingRootPackage {
        manifest_path: PathBuf,
        package: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoveredProjectComposition {
    manifest_path: PathBuf,
    composition: ProjectCompositionGraph,
    root_package: String,
}

impl ProjectManifest {
    pub fn validate(&self, project_root: &Path) -> Result<(), ProjectManifestValidationError> {
        let mut messages = Vec::new();
        let package_name = self.package.name.trim();
        if package_name.is_empty() {
            messages.push("[package].name is required and must be non-empty".to_string());
        }

        if self.sources.roots.is_empty() {
            messages.push("[sources].roots is required and must be non-empty".to_string());
        }

        for root in &self.sources.roots {
            if !is_relative_without_parent(root) {
                messages.push(format!(
                    "source root `{}` must be project-relative without `..` segments",
                    root.display()
                ));
                continue;
            }
            let resolved = project_root.join(root);
            if !resolved.is_dir() {
                messages.push(format!(
                    "source root `{}` does not exist as a directory under project root",
                    root.display()
                ));
            }
        }

        for (name, dep) in &self.dependencies {
            if name.trim().is_empty() {
                messages.push("dependency names must be non-empty".to_string());
            }
            if !is_relative_without_parent(&dep.path) {
                messages.push(format!(
                    "dependency `{name}` path `{}` must be project-relative without `..` segments",
                    dep.path.display()
                ));
                continue;
            }
            let resolved = project_root.join(&dep.path);
            if !resolved.is_dir() {
                messages.push(format!(
                    "dependency `{name}` path `{}` does not exist as a directory",
                    dep.path.display()
                ));
            }
        }

        let mut entrypoint_names = HashSet::new();
        for entrypoint in &self.entrypoints {
            let name = entrypoint.name.trim();
            if name.is_empty() {
                messages.push("entrypoint name must be non-empty".to_string());
                continue;
            }
            if !entrypoint_names.insert(name.to_string()) {
                messages.push(format!("duplicate entrypoint name `{name}`"));
            }

            let has_path = entrypoint.path.is_some();
            let has_module_function = entrypoint
                .module
                .as_ref()
                .is_some_and(|m| !m.trim().is_empty())
                && entrypoint
                    .function
                    .as_ref()
                    .is_some_and(|f| !f.trim().is_empty());

            if has_path == has_module_function {
                messages.push(format!(
                    "entrypoint `{name}` must use exactly one target form: either `path` or (`module` + `function`)"
                ));
                continue;
            }

            if let Some(path) = &entrypoint.path {
                if !is_relative_without_parent(path) {
                    messages.push(format!(
                        "entrypoint `{name}` path `{}` must be project-relative without `..` segments",
                        path.display()
                    ));
                    continue;
                }
                let resolved = resolve_entrypoint_path(project_root, path);
                if resolved.is_none() {
                    messages.push(format!(
                        "entrypoint `{name}` path `{}` does not resolve to an existing file (with optional `.m` inference)",
                        path.display()
                    ));
                }
            } else {
                if entrypoint
                    .module
                    .as_ref()
                    .is_some_and(|module| module.trim().is_empty())
                {
                    messages.push(format!("entrypoint `{name}` has an empty `module`"));
                }
                if entrypoint
                    .function
                    .as_ref()
                    .is_some_and(|function| function.trim().is_empty())
                {
                    messages.push(format!("entrypoint `{name}` has an empty `function`"));
                }
            }
        }

        if messages.is_empty() {
            Ok(())
        } else {
            Err(ProjectManifestValidationError { messages })
        }
    }
}

pub fn parse_project_manifest_toml(input: &str) -> Result<ProjectManifest, toml::de::Error> {
    toml::from_str(input)
}

pub fn load_project_manifest(path: &Path) -> Result<ProjectManifest, ProjectManifestLoadError> {
    let content = fs::read_to_string(path).map_err(|source| ProjectManifestLoadError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let manifest: ProjectManifest = parse_project_manifest_toml(&content).map_err(|source| {
        ProjectManifestLoadError::Parse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    let project_root = path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .validate(project_root)
        .map_err(|source| ProjectManifestLoadError::Validation {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(manifest)
}

pub fn discover_project_manifest_from(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        let candidate = current.join(PROJECT_MANIFEST_FILENAME);
        if candidate.is_file() {
            return Some(candidate);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

pub fn build_project_source_index(
    project_root: &Path,
    manifest: &ProjectManifest,
) -> Result<ProjectSourceIndex, ProjectSourceIndexError> {
    let mut index = ProjectSourceIndex::default();
    for source_root in &manifest.sources.roots {
        let abs_root = project_root.join(source_root);
        if !abs_root.is_dir() {
            return Err(ProjectSourceIndexError::InvalidSourceRoot {
                root: source_root.clone(),
            });
        }
        let state = ScanState::default();
        scan_source_dir(
            &abs_root,
            &abs_root,
            source_root,
            &state,
            &mut index,
            project_root,
        )?;
    }

    index
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    index.package_dirs.sort();
    index.package_dirs.dedup();
    index.class_dirs.sort();
    index.class_dirs.dedup();
    index.private_dirs.sort();
    index.private_dirs.dedup();
    Ok(index)
}

pub fn resolve_project_entrypoint(
    project_root: &Path,
    manifest: &ProjectManifest,
    entrypoint_name: &str,
) -> Result<Option<ResolvedProjectEntrypoint>, ProjectEntrypointResolveError> {
    let Some(entrypoint) = manifest
        .entrypoints
        .iter()
        .find(|entry| entry.name == entrypoint_name)
    else {
        return Ok(None);
    };

    if let Some(path) = &entrypoint.path {
        let Some(source_file) = resolve_entrypoint_path(project_root, path) else {
            return Err(ProjectEntrypointResolveError::MissingPathTarget {
                entrypoint: entrypoint_name.to_string(),
                path: path.clone(),
            });
        };
        return Ok(Some(ResolvedProjectEntrypoint {
            name: entrypoint_name.to_string(),
            source_file,
            module: None,
            function: None,
            target: ResolvedEntrypointTarget::Path,
        }));
    }

    if let (Some(module), Some(function)) = (&entrypoint.module, &entrypoint.function) {
        let Some(source_file) =
            resolve_module_function_source_file(project_root, manifest, module, function).map_err(
                |source| ProjectEntrypointResolveError::SourceIndex {
                    entrypoint: entrypoint_name.to_string(),
                    source,
                },
            )?
        else {
            return Err(ProjectEntrypointResolveError::MissingModuleTarget {
                entrypoint: entrypoint_name.to_string(),
                module: module.clone(),
                function: function.clone(),
            });
        };
        return Ok(Some(ResolvedProjectEntrypoint {
            name: entrypoint_name.to_string(),
            source_file,
            module: Some(module.clone()),
            function: Some(function.clone()),
            target: ResolvedEntrypointTarget::ModuleFunction,
        }));
    }

    Ok(None)
}

pub fn resolve_named_entrypoint_from(
    start: &Path,
    entrypoint_name: &str,
) -> Result<Option<DiscoveredProjectEntrypoint>, DiscoverProjectEntrypointError> {
    let Some(discovered) = discover_project_composition_from(start).map_err(|err| match err {
        DiscoverProjectCompositionError::Composition {
            manifest_path,
            source,
        } => DiscoverProjectEntrypointError::Composition {
            manifest_path,
            source,
        },
        DiscoverProjectCompositionError::MissingRootPackage {
            manifest_path,
            package,
        } => DiscoverProjectEntrypointError::MissingRootPackage {
            manifest_path,
            package,
        },
    })?
    else {
        return Ok(None);
    };
    let manifest_path = discovered.manifest_path.clone();
    let root_package = discovered.root_package.clone();
    let root = discovered
        .composition
        .packages
        .get(&root_package)
        .expect("root package should be present");
    let Some(entrypoint) =
        resolve_project_entrypoint(&root.project_root, &root.manifest, entrypoint_name).map_err(
            |source| DiscoverProjectEntrypointError::Resolve {
                manifest_path: manifest_path.clone(),
                entrypoint: entrypoint_name.to_string(),
                source,
            },
        )?
    else {
        return Ok(None);
    };
    Ok(Some(DiscoveredProjectEntrypoint {
        manifest_path,
        root_package,
        project_root: root.project_root.clone(),
        entrypoint,
    }))
}

pub fn discover_project_symbols_from(
    start: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    let Some(discovered) = discover_project_composition_from(start).map_err(|err| match err {
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
    })?
    else {
        return Ok(None);
    };
    let manifest_path = discovered.manifest_path.clone();
    let root_package = discovered.root_package.clone();
    let root = discovered
        .composition
        .packages
        .get(&root_package)
        .expect("root package should be present");
    let root_dependencies = root.dependencies.clone();
    let mut symbols = HashSet::new();
    for package in discovered.composition.packages.values() {
        for source in &package.source_index.files {
            symbols.insert(source.qualified_name.clone());
            symbols.insert(format!(
                "{}.{}",
                package.package_name, source.qualified_name
            ));
            for (alias, dependency_package) in &root_dependencies {
                if dependency_package == &package.package_name {
                    symbols.insert(format!("{alias}.{}", source.qualified_name));
                }
            }
        }
    }
    Ok(Some(DiscoveredProjectSymbols {
        manifest_path,
        root_package,
        project_root: root.project_root.clone(),
        symbols,
    }))
}

pub fn discover_project_symbols_from_source_name(
    source_name: &str,
    cwd: &Path,
) -> Result<Option<DiscoveredProjectSymbols>, DiscoverProjectSymbolsError> {
    let source_path = PathBuf::from(source_name);
    let start = if source_path.is_file() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.is_absolute() {
        source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else if source_path.components().count() > 1 {
        let joined = cwd.join(&source_path);
        joined
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf())
    } else {
        cwd.to_path_buf()
    };
    discover_project_symbols_from(&start)
}

pub fn resolve_project_source_input_from(
    cwd: &Path,
    source_input: &Path,
) -> Result<PathBuf, ResolveProjectSourceInputError> {
    let candidate = if source_input.is_absolute() {
        source_input.to_path_buf()
    } else {
        cwd.join(source_input)
    };
    if candidate.is_file() {
        return Ok(source_input.to_path_buf());
    }
    if source_input.extension().is_none() {
        let with_ext = if source_input.is_absolute() {
            source_input.with_extension("m")
        } else {
            cwd.join(source_input).with_extension("m")
        };
        if with_ext.is_file() {
            return Ok(source_input.with_extension("m"));
        }
    }

    let Some(entrypoint_name) = source_input_entrypoint_name_candidate(source_input) else {
        return Ok(source_input.to_path_buf());
    };
    let Some(discovered) =
        resolve_named_entrypoint_from(cwd, &entrypoint_name).map_err(|source| {
            ResolveProjectSourceInputError::EntrypointResolve {
                cwd: cwd.to_path_buf(),
                entrypoint: entrypoint_name.clone(),
                source,
            }
        })?
    else {
        return Ok(source_input.to_path_buf());
    };
    Ok(discovered.entrypoint.source_file)
}

pub fn build_project_composition_graph(
    root_manifest_path: &Path,
) -> Result<ProjectCompositionGraph, ProjectCompositionError> {
    let mut loader = CompositionGraphLoader::default();
    let root_package = loader.load_package(
        root_manifest_path,
        None,
        true,
        &mut Vec::new(),
        &mut Vec::new(),
    )?;
    Ok(ProjectCompositionGraph {
        root_package,
        packages: loader.packages,
    })
}

fn is_relative_without_parent(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
}

fn discover_project_composition_from(
    start: &Path,
) -> Result<Option<DiscoveredProjectComposition>, DiscoverProjectCompositionError> {
    let Some(manifest_path) = discover_project_manifest_from(start) else {
        return Ok(None);
    };
    let composition = build_project_composition_graph(&manifest_path).map_err(|source| {
        DiscoverProjectCompositionError::Composition {
            manifest_path: manifest_path.clone(),
            source,
        }
    })?;
    let root_package = composition.root_package.clone();
    if !composition.packages.contains_key(&root_package) {
        return Err(DiscoverProjectCompositionError::MissingRootPackage {
            manifest_path,
            package: root_package,
        });
    }
    Ok(Some(DiscoveredProjectComposition {
        manifest_path,
        composition,
        root_package,
    }))
}

fn source_input_entrypoint_name_candidate(path: &Path) -> Option<String> {
    if path.extension().is_some() {
        return None;
    }
    if path.components().count() != 1 {
        return None;
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn resolve_entrypoint_path(project_root: &Path, path: &Path) -> Option<PathBuf> {
    let direct = project_root.join(path);
    if direct.is_file() {
        return Some(direct);
    }
    if direct.extension().is_none() {
        let with_ext = direct.with_extension("m");
        if with_ext.is_file() {
            return Some(with_ext);
        }
    }
    None
}

fn resolve_module_function_source_file(
    project_root: &Path,
    manifest: &ProjectManifest,
    module: &str,
    function: &str,
) -> Result<Option<PathBuf>, ProjectSourceIndexError> {
    let index = build_project_source_index(project_root, manifest)?;
    let module_function = format!("{module}.{function}");
    for file in &index.files {
        if file.qualified_name == module || file.qualified_name == module_function {
            return Ok(Some(
                project_root
                    .join(&file.source_root)
                    .join(&file.relative_path),
            ));
        }
    }
    Ok(None)
}

#[derive(Default)]
struct CompositionGraphLoader {
    packages: BTreeMap<String, ProjectCompositionPackage>,
    package_by_manifest: BTreeMap<PathBuf, String>,
}

impl CompositionGraphLoader {
    fn load_package(
        &mut self,
        manifest_path: &Path,
        from: Option<(&str, &str)>,
        is_root: bool,
        active_paths: &mut Vec<PathBuf>,
        active_package_names: &mut Vec<String>,
    ) -> Result<String, ProjectCompositionError> {
        let manifest_path = canonical_manifest_path(manifest_path);
        if let Some(existing) = self.package_by_manifest.get(&manifest_path) {
            return Ok(existing.clone());
        }

        if let Some(idx) = active_paths.iter().position(|path| path == &manifest_path) {
            let mut cycle = active_package_names[idx..].to_vec();
            if let Some(last) = active_package_names.last() {
                cycle.push(last.clone());
            }
            return Err(ProjectCompositionError::DependencyCycle {
                cycle: cycle.join(" -> "),
            });
        }

        let manifest = if is_root {
            load_project_manifest(&manifest_path).map_err(|source| {
                ProjectCompositionError::RootManifestLoad {
                    path: manifest_path.clone(),
                    source,
                }
            })?
        } else {
            let (package, dependency) = from.expect("dependency context is required");
            load_project_manifest(&manifest_path).map_err(|source| {
                ProjectCompositionError::DependencyManifestLoad {
                    package: package.to_string(),
                    dependency: dependency.to_string(),
                    path: manifest_path.clone(),
                    source,
                }
            })?
        };

        let package_name = manifest.package.name.clone();
        let project_root = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let source_index =
            build_project_source_index(&project_root, &manifest).map_err(|source| {
                ProjectCompositionError::SourceIndex {
                    package: package_name.clone(),
                    source,
                }
            })?;

        if let Some(existing) = self.packages.get(&package_name) {
            if existing.manifest_path != manifest_path {
                return Err(ProjectCompositionError::DuplicatePackageName {
                    package: package_name,
                    first_manifest: existing.manifest_path.clone(),
                    second_manifest: manifest_path,
                });
            }
            return Ok(existing.package_name.clone());
        }

        active_paths.push(manifest_path.clone());
        active_package_names.push(package_name.clone());

        let mut dependency_map = BTreeMap::new();
        for (dependency_name, dep) in &manifest.dependencies {
            let dep_manifest_path = project_root.join(&dep.path).join(PROJECT_MANIFEST_FILENAME);
            if !dep_manifest_path.is_file() {
                return Err(ProjectCompositionError::MissingDependencyManifest {
                    package: package_name.clone(),
                    dependency: dependency_name.clone(),
                    path: dep_manifest_path,
                });
            }
            let dep_package_name = self.load_package(
                &dep_manifest_path,
                Some((&package_name, dependency_name)),
                false,
                active_paths,
                active_package_names,
            )?;
            dependency_map.insert(dependency_name.clone(), dep_package_name);
        }

        active_paths.pop();
        active_package_names.pop();

        if let Some(existing) = self.packages.get(&package_name) {
            if existing.manifest_path != manifest_path {
                return Err(ProjectCompositionError::DuplicatePackageName {
                    package: package_name,
                    first_manifest: existing.manifest_path.clone(),
                    second_manifest: manifest_path,
                });
            }
            return Ok(existing.package_name.clone());
        }

        self.package_by_manifest
            .insert(manifest_path.clone(), package_name.clone());
        self.packages.insert(
            package_name.clone(),
            ProjectCompositionPackage {
                package_name: package_name.clone(),
                manifest_path,
                project_root,
                manifest,
                source_index,
                dependencies: dependency_map,
            },
        );

        Ok(package_name)
    }
}

fn canonical_manifest_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[derive(Debug, Clone, Default)]
struct ScanState {
    package_segments: Vec<String>,
    module_segments: Vec<String>,
    class_name: Option<String>,
    in_private: bool,
}

fn scan_source_dir(
    dir: &Path,
    root_abs: &Path,
    source_root: &Path,
    state: &ScanState,
    index: &mut ProjectSourceIndex,
    project_root: &Path,
) -> Result<(), ProjectSourceIndexError> {
    let mut entries = fs::read_dir(dir).map_err(|source| ProjectSourceIndexError::ReadDir {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut sorted = Vec::new();
    for entry in &mut entries {
        let entry = entry.map_err(|source| ProjectSourceIndexError::ReadEntry {
            path: dir.to_path_buf(),
            source,
        })?;
        sorted.push(entry);
    }
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in sorted {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| ProjectSourceIndexError::ReadEntry {
                path: dir.to_path_buf(),
                source,
            })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if file_type.is_dir() {
            let mut next = state.clone();
            if let Some(pkg) = name.strip_prefix('+') {
                if !pkg.is_empty() {
                    next.package_segments.push(pkg.to_string());
                    if let Ok(rel) = path.strip_prefix(project_root) {
                        index.package_dirs.push(rel.to_path_buf());
                    }
                }
            } else if let Some(class) = name.strip_prefix('@') {
                if !class.is_empty() {
                    next.class_name = Some(class.to_string());
                    if let Ok(rel) = path.strip_prefix(project_root) {
                        index.class_dirs.push(rel.to_path_buf());
                    }
                }
            } else if name == "private" {
                next.in_private = true;
                if let Ok(rel) = path.strip_prefix(project_root) {
                    index.private_dirs.push(rel.to_path_buf());
                }
            } else {
                next.module_segments.push(name.to_string());
            }
            scan_source_dir(&path, root_abs, source_root, &next, index, project_root)?;
            continue;
        }

        let is_m_file = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("m"))
            .unwrap_or(false);
        if !is_m_file {
            continue;
        }

        let relative_path = path
            .strip_prefix(root_abs)
            .unwrap_or(path.as_path())
            .to_path_buf();
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("")
            .trim();
        if stem.is_empty() {
            continue;
        }

        let mut qualified_segments = Vec::new();
        qualified_segments.extend(state.package_segments.iter().cloned());
        qualified_segments.extend(state.module_segments.iter().cloned());
        if let Some(class_name) = &state.class_name {
            qualified_segments.push(class_name.clone());
        }
        qualified_segments.push(stem.to_string());
        let qualified_name = qualified_segments.join(".");
        if qualified_name.is_empty() {
            continue;
        }

        let package_path = if state.package_segments.is_empty() {
            None
        } else {
            Some(state.package_segments.join("."))
        };

        index.files.push(ProjectSourceFile {
            source_root: source_root.to_path_buf(),
            relative_path,
            qualified_name,
            package_path,
            class_name: state.class_name.clone(),
            is_private: state.in_private,
        });
    }

    Ok(())
}
