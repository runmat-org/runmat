use super::manifest::{
    discover_project_manifest_from, discover_project_manifest_from_async, load_project_manifest,
    load_project_manifest_async, path_is_file_async, resolve_entrypoint_path,
    resolve_entrypoint_path_async, ProjectManifest, ProjectManifestLoadError,
};
use super::source_index::{
    build_project_source_index, build_project_source_index_async, ProjectSourceIndexError,
};
use std::path::{Path, PathBuf};
use thiserror::Error;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProjectEntrypoint {
    pub manifest_path: PathBuf,
    pub root_package: String,
    pub project_root: PathBuf,
    pub entrypoint: ResolvedProjectEntrypoint,
}

#[derive(Debug, Error)]
pub enum DiscoverProjectEntrypointError {
    #[error("failed to load discovered project manifest {manifest_path}: {source}")]
    ManifestLoad {
        manifest_path: PathBuf,
        #[source]
        source: Box<ProjectManifestLoadError>,
    },
    #[error("failed to resolve project entrypoint `{entrypoint}` from {manifest_path}: {source}")]
    Resolve {
        manifest_path: PathBuf,
        entrypoint: String,
        #[source]
        source: Box<ProjectEntrypointResolveError>,
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
        source: Box<DiscoverProjectEntrypointError>,
    },
}

pub fn resolve_project_entrypoint(
    project_root: &Path,
    manifest: &ProjectManifest,
    entrypoint_name: &str,
) -> Result<Option<ResolvedProjectEntrypoint>, ProjectEntrypointResolveError> {
    let Some(entrypoint) = manifest
        .entrypoints
        .iter()
        .find(|entrypoint| entrypoint.name == entrypoint_name)
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
        return Ok(Some(resolved_path_entrypoint(entrypoint_name, source_file)));
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
            return Err(missing_module_target(entrypoint_name, module, function));
        };
        return Ok(Some(resolved_module_entrypoint(
            entrypoint_name,
            source_file,
            module,
            function,
        )));
    }
    Ok(None)
}

pub async fn resolve_project_entrypoint_async(
    project_root: &Path,
    manifest: &ProjectManifest,
    entrypoint_name: &str,
) -> Result<Option<ResolvedProjectEntrypoint>, ProjectEntrypointResolveError> {
    let Some(entrypoint) = manifest
        .entrypoints
        .iter()
        .find(|entrypoint| entrypoint.name == entrypoint_name)
    else {
        return Ok(None);
    };

    if let Some(path) = &entrypoint.path {
        let Some(source_file) = resolve_entrypoint_path_async(project_root, path).await else {
            return Err(ProjectEntrypointResolveError::MissingPathTarget {
                entrypoint: entrypoint_name.to_string(),
                path: path.clone(),
            });
        };
        return Ok(Some(resolved_path_entrypoint(entrypoint_name, source_file)));
    }

    if let (Some(module), Some(function)) = (&entrypoint.module, &entrypoint.function) {
        let Some(source_file) =
            resolve_module_function_source_file_async(project_root, manifest, module, function)
                .await
                .map_err(|source| ProjectEntrypointResolveError::SourceIndex {
                    entrypoint: entrypoint_name.to_string(),
                    source,
                })?
        else {
            return Err(missing_module_target(entrypoint_name, module, function));
        };
        return Ok(Some(resolved_module_entrypoint(
            entrypoint_name,
            source_file,
            module,
            function,
        )));
    }
    Ok(None)
}

pub fn resolve_named_entrypoint_from(
    start: &Path,
    entrypoint_name: &str,
) -> Result<Option<DiscoveredProjectEntrypoint>, DiscoverProjectEntrypointError> {
    let Some(manifest_path) = discover_project_manifest_from(start) else {
        return Ok(None);
    };
    let manifest = load_project_manifest(&manifest_path).map_err(|source| {
        DiscoverProjectEntrypointError::ManifestLoad {
            manifest_path: manifest_path.clone(),
            source: Box::new(source),
        }
    })?;
    finish_discovered_entrypoint(
        manifest_path,
        manifest,
        entrypoint_name,
        resolve_project_entrypoint,
    )
}

pub async fn resolve_named_entrypoint_from_async(
    start: &Path,
    entrypoint_name: &str,
) -> Result<Option<DiscoveredProjectEntrypoint>, DiscoverProjectEntrypointError> {
    let Some(manifest_path) = discover_project_manifest_from_async(start).await else {
        return Ok(None);
    };
    let manifest = load_project_manifest_async(&manifest_path)
        .await
        .map_err(|source| DiscoverProjectEntrypointError::ManifestLoad {
            manifest_path: manifest_path.clone(),
            source: Box::new(source),
        })?;
    let project_root = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let root_package = manifest.package.name.clone();
    let Some(entrypoint) =
        resolve_project_entrypoint_async(&project_root, &manifest, entrypoint_name)
            .await
            .map_err(|source| DiscoverProjectEntrypointError::Resolve {
                manifest_path: manifest_path.clone(),
                entrypoint: entrypoint_name.to_string(),
                source: Box::new(source),
            })?
    else {
        return Ok(None);
    };
    Ok(Some(DiscoveredProjectEntrypoint {
        manifest_path,
        root_package,
        project_root,
        entrypoint,
    }))
}

pub fn resolve_project_source_input_from(
    cwd: &Path,
    source_input: &Path,
) -> Result<PathBuf, ResolveProjectSourceInputError> {
    let candidate = absolute_or_join(cwd, source_input);
    if candidate.is_file() {
        return Ok(source_input.to_path_buf());
    }
    if source_input.extension().is_none() {
        let inferred = candidate.with_extension("m");
        if inferred.is_file() {
            return Ok(source_input.with_extension("m"));
        }
    }
    let Some(entrypoint_name) = source_input_entrypoint_name_candidate(source_input) else {
        return Ok(source_input.to_path_buf());
    };
    let discovered = resolve_named_entrypoint_from(cwd, &entrypoint_name)
        .map_err(|source| source_input_error(cwd, &entrypoint_name, source))?;
    Ok(discovered
        .map(|discovered| discovered.entrypoint.source_file)
        .unwrap_or_else(|| source_input.to_path_buf()))
}

pub async fn resolve_project_source_input_from_async(
    cwd: &Path,
    source_input: &Path,
) -> Result<PathBuf, ResolveProjectSourceInputError> {
    let candidate = absolute_or_join(cwd, source_input);
    if path_is_file_async(&candidate).await {
        return Ok(source_input.to_path_buf());
    }
    if source_input.extension().is_none()
        && path_is_file_async(&candidate.with_extension("m")).await
    {
        return Ok(source_input.with_extension("m"));
    }
    let Some(entrypoint_name) = source_input_entrypoint_name_candidate(source_input) else {
        return Ok(source_input.to_path_buf());
    };
    let discovered = resolve_named_entrypoint_from_async(cwd, &entrypoint_name)
        .await
        .map_err(|source| source_input_error(cwd, &entrypoint_name, source))?;
    Ok(discovered
        .map(|discovered| discovered.entrypoint.source_file)
        .unwrap_or_else(|| source_input.to_path_buf()))
}

fn source_input_error(
    cwd: &Path,
    entrypoint: &str,
    source: DiscoverProjectEntrypointError,
) -> ResolveProjectSourceInputError {
    ResolveProjectSourceInputError::EntrypointResolve {
        cwd: cwd.to_path_buf(),
        entrypoint: entrypoint.to_string(),
        source: Box::new(source),
    }
}

fn source_input_entrypoint_name_candidate(path: &Path) -> Option<String> {
    if path.extension().is_some() || path.components().count() != 1 {
        return None;
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn absolute_or_join(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn finish_discovered_entrypoint(
    manifest_path: PathBuf,
    manifest: ProjectManifest,
    entrypoint_name: &str,
    resolver: impl Fn(
        &Path,
        &ProjectManifest,
        &str,
    ) -> Result<Option<ResolvedProjectEntrypoint>, ProjectEntrypointResolveError>,
) -> Result<Option<DiscoveredProjectEntrypoint>, DiscoverProjectEntrypointError> {
    let project_root = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let root_package = manifest.package.name.clone();
    let Some(entrypoint) =
        resolver(&project_root, &manifest, entrypoint_name).map_err(|source| {
            DiscoverProjectEntrypointError::Resolve {
                manifest_path: manifest_path.clone(),
                entrypoint: entrypoint_name.to_string(),
                source: Box::new(source),
            }
        })?
    else {
        return Ok(None);
    };
    Ok(Some(DiscoveredProjectEntrypoint {
        manifest_path,
        root_package,
        project_root,
        entrypoint,
    }))
}

fn resolved_path_entrypoint(name: &str, source_file: PathBuf) -> ResolvedProjectEntrypoint {
    ResolvedProjectEntrypoint {
        name: name.to_string(),
        source_file,
        module: None,
        function: None,
        target: ResolvedEntrypointTarget::Path,
    }
}

fn resolved_module_entrypoint(
    name: &str,
    source_file: PathBuf,
    module: &str,
    function: &str,
) -> ResolvedProjectEntrypoint {
    ResolvedProjectEntrypoint {
        name: name.to_string(),
        source_file,
        module: Some(module.to_string()),
        function: Some(function.to_string()),
        target: ResolvedEntrypointTarget::ModuleFunction,
    }
}

fn missing_module_target(
    entrypoint: &str,
    module: &str,
    function: &str,
) -> ProjectEntrypointResolveError {
    ProjectEntrypointResolveError::MissingModuleTarget {
        entrypoint: entrypoint.to_string(),
        module: module.to_string(),
        function: function.to_string(),
    }
}

fn resolve_module_function_source_file(
    project_root: &Path,
    manifest: &ProjectManifest,
    module: &str,
    function: &str,
) -> Result<Option<PathBuf>, ProjectSourceIndexError> {
    let index = build_project_source_index(project_root, manifest)?;
    Ok(module_source_path(project_root, &index, module, function))
}

async fn resolve_module_function_source_file_async(
    project_root: &Path,
    manifest: &ProjectManifest,
    module: &str,
    function: &str,
) -> Result<Option<PathBuf>, ProjectSourceIndexError> {
    let index = build_project_source_index_async(project_root, manifest).await?;
    Ok(module_source_path(project_root, &index, module, function))
}

fn module_source_path(
    project_root: &Path,
    index: &super::ProjectSourceIndex,
    module: &str,
    function: &str,
) -> Option<PathBuf> {
    let module_function = format!("{module}.{function}");
    index.files.iter().find_map(|file| {
        (file.qualified_name == module || file.qualified_name == module_function).then(|| {
            project_root
                .join(&file.source_root)
                .join(&file.relative_path)
        })
    })
}
