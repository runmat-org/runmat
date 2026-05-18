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

fn is_relative_without_parent(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
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
