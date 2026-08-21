use super::manifest::{path_is_dir_async, ProjectManifest};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

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
    /// Canonical class scope contributed by `@Class` folders. This intentionally
    /// excludes the file stem: `@Report/Report.m` and `@Report/title.m` both
    /// belong to class `Report`, while their callable file identities remain
    /// `Report.Report` and `Report.title` respectively.
    #[serde(default)]
    pub class_qualified_name: Option<String>,
    pub is_private: bool,
}

impl ProjectSourceFile {
    /// The canonical name to apply to a parsed `classdef` source.
    pub fn class_definition_qualified_name(&self) -> Option<&str> {
        self.class_qualified_name.as_deref().or_else(|| {
            self.package_path
                .as_ref()
                .map(|_| self.qualified_name.as_str())
        })
    }

    /// The callable identity for a parsed function or class-folder member file.
    pub fn function_qualified_name(&self) -> Option<&str> {
        if self.is_private {
            return None;
        }
        (self.package_path.is_some() || self.class_name.is_some())
            .then_some(self.qualified_name.as_str())
            .filter(|name| name.contains('.'))
    }
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

pub fn build_project_source_index(
    project_root: &Path,
    manifest: &ProjectManifest,
) -> Result<ProjectSourceIndex, ProjectSourceIndexError> {
    let mut index = ProjectSourceIndex::default();
    for source_root in &manifest.sources.roots {
        let absolute_root = project_root.join(source_root);
        if !absolute_root.is_dir() {
            return Err(ProjectSourceIndexError::InvalidSourceRoot {
                root: source_root.clone(),
            });
        }
        scan_source_dir(
            &absolute_root,
            &absolute_root,
            source_root,
            &ScanState::default(),
            &mut index,
            project_root,
        )?;
    }
    normalize_source_index(&mut index);
    Ok(index)
}

pub async fn build_project_source_index_async(
    project_root: &Path,
    manifest: &ProjectManifest,
) -> Result<ProjectSourceIndex, ProjectSourceIndexError> {
    let mut index = ProjectSourceIndex::default();
    for source_root in &manifest.sources.roots {
        let absolute_root = project_root.join(source_root);
        if !path_is_dir_async(&absolute_root).await {
            return Err(ProjectSourceIndexError::InvalidSourceRoot {
                root: source_root.clone(),
            });
        }
        scan_source_dir_async(
            &absolute_root,
            &absolute_root,
            source_root,
            &ScanState::default(),
            &mut index,
            project_root,
        )
        .await?;
    }
    normalize_source_index(&mut index);
    Ok(index)
}

/// Build the MATLAB lookup index for a folder that has no `runmat.toml`.
pub fn build_loose_source_index(
    root: &Path,
) -> Result<ProjectSourceIndex, ProjectSourceIndexError> {
    let mut index = ProjectSourceIndex::default();
    let entries = fs::read_dir(root).map_err(|source| ProjectSourceIndexError::ReadDir {
        path: root.to_path_buf(),
        source,
    })?;
    let mut entries = entries
        .map(|entry| {
            entry.map_err(|source| ProjectSourceIndexError::ReadEntry {
                path: root.to_path_buf(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| ProjectSourceIndexError::ReadEntry {
                path: root.to_path_buf(),
                source,
            })?;
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('+') || name.starts_with('@') || name == "private" {
                scan_source_dir(
                    &path,
                    root,
                    Path::new("."),
                    &ScanState::default(),
                    &mut index,
                    root,
                )?;
            }
            continue;
        }
        if let Some(source) = project_source_file_from_path(&path, root, Path::new(".")) {
            index.files.push(source);
        }
    }
    normalize_source_index(&mut index);
    Ok(index)
}

pub async fn build_loose_source_index_async(
    root: &Path,
) -> Result<ProjectSourceIndex, ProjectSourceIndexError> {
    let mut index = ProjectSourceIndex::default();
    let mut entries = runmat_filesystem::read_dir_async(root)
        .await
        .map_err(|source| ProjectSourceIndexError::ReadDir {
            path: root.to_path_buf(),
            source,
        })?;
    entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_string());

    for entry in entries {
        let path = entry.path().to_path_buf();
        if entry.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('+') || name.starts_with('@') || name == "private" {
                scan_source_dir_async(
                    &path,
                    root,
                    Path::new("."),
                    &ScanState::default(),
                    &mut index,
                    root,
                )
                .await?;
            }
            continue;
        }
        if let Some(source) = project_source_file_from_path(&path, root, Path::new(".")) {
            index.files.push(source);
        }
    }
    normalize_source_index(&mut index);
    Ok(index)
}

fn normalize_source_index(index: &mut ProjectSourceIndex) {
    index
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    index.package_dirs.sort();
    index.package_dirs.dedup();
    index.class_dirs.sort();
    index.class_dirs.dedup();
    index.private_dirs.sort();
    index.private_dirs.dedup();
}

#[derive(Debug, Clone, Default)]
struct ScanState {
    package_segments: Vec<String>,
    module_segments: Vec<String>,
    class_name: Option<String>,
    in_private: bool,
}

/// Normalize one MATLAB source path into the identity used by project
/// composition and loose-file discovery.
pub fn project_source_file_from_path(
    source_path: &Path,
    root_dir: &Path,
    source_root: &Path,
) -> Option<ProjectSourceFile> {
    let relative_path = source_path.strip_prefix(root_dir).ok()?.to_path_buf();
    if !source_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("m"))
    {
        return None;
    }
    let stem = source_path.file_stem()?.to_str()?.trim();
    if stem.is_empty() {
        return None;
    }

    let mut state = ScanState::default();
    if let Some(parent) = relative_path.parent() {
        for component in parent.components() {
            let segment = component.as_os_str().to_str()?;
            if let Some(package) = segment.strip_prefix('+') {
                if package.is_empty() {
                    return None;
                }
                state.package_segments.push(package.to_string());
            } else if let Some(class) = segment.strip_prefix('@') {
                if class.is_empty() {
                    return None;
                }
                state.class_name = Some(class.to_string());
            } else if segment == "private" {
                state.in_private = true;
            } else {
                state.module_segments.push(segment.to_string());
            }
        }
    }

    let mut qualified_segments = state.package_segments.clone();
    qualified_segments.extend(state.module_segments.iter().cloned());
    let class_qualified_name = state.class_name.as_ref().map(|class_name| {
        let mut class_segments = qualified_segments.clone();
        class_segments.push(class_name.clone());
        class_segments.join(".")
    });
    if let Some(class_name) = &state.class_name {
        qualified_segments.push(class_name.clone());
    }
    qualified_segments.push(stem.to_string());
    let qualified_name = qualified_segments.join(".");
    (!qualified_name.is_empty()).then_some(ProjectSourceFile {
        source_root: source_root.to_path_buf(),
        relative_path,
        qualified_name,
        package_path: (!state.package_segments.is_empty())
            .then(|| state.package_segments.join(".")),
        class_name: state.class_name,
        class_qualified_name,
        is_private: state.in_private,
    })
}

fn scan_source_dir(
    dir: &Path,
    root_absolute: &Path,
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
        sorted.push(entry.map_err(|source| ProjectSourceIndexError::ReadEntry {
            path: dir.to_path_buf(),
            source,
        })?);
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
            if let Some(package) = name.strip_prefix('+') {
                if !package.is_empty() {
                    next.package_segments.push(package.to_string());
                    if let Ok(relative) = path.strip_prefix(project_root) {
                        index.package_dirs.push(relative.to_path_buf());
                    }
                }
            } else if let Some(class) = name.strip_prefix('@') {
                if !class.is_empty() {
                    next.class_name = Some(class.to_string());
                    if let Ok(relative) = path.strip_prefix(project_root) {
                        index.class_dirs.push(relative.to_path_buf());
                    }
                }
            } else if name == "private" {
                next.in_private = true;
                if let Ok(relative) = path.strip_prefix(project_root) {
                    index.private_dirs.push(relative.to_path_buf());
                }
            } else {
                next.module_segments.push(name.to_string());
            }
            scan_source_dir(
                &path,
                root_absolute,
                source_root,
                &next,
                index,
                project_root,
            )?;
            continue;
        }
        if let Some(source) = project_source_file_from_path(&path, root_absolute, source_root) {
            index.files.push(source);
        }
    }
    Ok(())
}

async fn scan_source_dir_async(
    dir: &Path,
    root_absolute: &Path,
    source_root: &Path,
    state: &ScanState,
    index: &mut ProjectSourceIndex,
    project_root: &Path,
) -> Result<(), ProjectSourceIndexError> {
    let mut stack = vec![(dir.to_path_buf(), state.clone())];
    while let Some((current_dir, current_state)) = stack.pop() {
        let mut sorted = runmat_filesystem::read_dir_async(&current_dir)
            .await
            .map_err(|source| ProjectSourceIndexError::ReadDir {
                path: current_dir.clone(),
                source,
            })?;
        sorted.sort_by_key(|entry| entry.file_name().to_string_lossy().to_string());

        for entry in sorted {
            let path = entry.path().to_path_buf();
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.is_dir() {
                let mut next = current_state.clone();
                if let Some(package) = name.strip_prefix('+') {
                    if !package.is_empty() {
                        next.package_segments.push(package.to_string());
                        if let Ok(relative) = path.strip_prefix(project_root) {
                            index.package_dirs.push(relative.to_path_buf());
                        }
                    }
                } else if let Some(class) = name.strip_prefix('@') {
                    if !class.is_empty() {
                        next.class_name = Some(class.to_string());
                        if let Ok(relative) = path.strip_prefix(project_root) {
                            index.class_dirs.push(relative.to_path_buf());
                        }
                    }
                } else if name == "private" {
                    next.in_private = true;
                    if let Ok(relative) = path.strip_prefix(project_root) {
                        index.private_dirs.push(relative.to_path_buf());
                    }
                } else {
                    next.module_segments.push(name);
                }
                stack.push((path, next));
                continue;
            }
            if let Some(source) = project_source_file_from_path(&path, root_absolute, source_root) {
                index.files.push(source);
            }
        }
    }
    Ok(())
}
