use super::manifest::{
    discover_project_manifest_from, discover_project_manifest_from_async,
    first_existing_manifest_path_async, load_project_manifest, load_project_manifest_async,
    path_is_file_async, ProjectManifest, ProjectManifestLoadError, PROJECT_MANIFEST_FILENAME,
    PROJECT_MANIFEST_FILENAMES,
};
use super::source_index::{
    build_project_source_index, build_project_source_index_async, ProjectSourceIndex,
    ProjectSourceIndexError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use thiserror::Error;

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
        source: Box<ProjectManifestLoadError>,
    },
    #[error("dependency `{dependency}` in package `{package}` points to missing manifest {path}")]
    MissingDependencyManifest {
        package: String,
        dependency: String,
        path: PathBuf,
    },
    #[error("dependency `{dependency}` in package `{package}` does not define a local `path` (version-only dependencies are not yet available to local composition)")]
    DependencyPathRequired { package: String, dependency: String },
    #[error(
        "failed to load dependency manifest {path} for dependency `{dependency}` of package `{package}`: {source}"
    )]
    DependencyManifestLoad {
        package: String,
        dependency: String,
        path: PathBuf,
        #[source]
        source: Box<ProjectManifestLoadError>,
    },
    #[error("failed to build source index for package `{package}`: {source}")]
    SourceIndex {
        package: String,
        #[source]
        source: Box<ProjectSourceIndexError>,
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

#[derive(Debug, Error)]
pub(super) enum DiscoverProjectCompositionError {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredProjectComposition {
    pub(super) manifest_path: PathBuf,
    pub(super) composition: ProjectCompositionGraph,
    pub(super) root_package: String,
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

pub async fn build_project_composition_graph_async(
    root_manifest_path: &Path,
) -> Result<ProjectCompositionGraph, ProjectCompositionError> {
    let mut loader = AsyncCompositionGraphLoader::default();
    let root_package = loader
        .load_package(
            root_manifest_path,
            None,
            true,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .await?;
    Ok(ProjectCompositionGraph {
        root_package,
        packages: loader.packages,
    })
}

pub(super) fn discover_project_composition_from(
    start: &Path,
) -> Result<Option<DiscoveredProjectComposition>, DiscoverProjectCompositionError> {
    let Some(manifest_path) = discover_project_manifest_from(start) else {
        return Ok(None);
    };
    finish_discovery(
        manifest_path.clone(),
        build_project_composition_graph(&manifest_path).map_err(|source| {
            DiscoverProjectCompositionError::Composition {
                manifest_path,
                source: Box::new(source),
            }
        })?,
    )
}

pub(super) async fn discover_project_composition_from_async(
    start: &Path,
) -> Result<Option<DiscoveredProjectComposition>, DiscoverProjectCompositionError> {
    let Some(manifest_path) = discover_project_manifest_from_async(start).await else {
        return Ok(None);
    };
    let composition = build_project_composition_graph_async(&manifest_path)
        .await
        .map_err(|source| DiscoverProjectCompositionError::Composition {
            manifest_path: manifest_path.clone(),
            source: Box::new(source),
        })?;
    finish_discovery(manifest_path, composition)
}

fn finish_discovery(
    manifest_path: PathBuf,
    composition: ProjectCompositionGraph,
) -> Result<Option<DiscoveredProjectComposition>, DiscoverProjectCompositionError> {
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
        if let Some(index) = active_paths.iter().position(|path| path == &manifest_path) {
            let mut cycle = active_package_names[index..].to_vec();
            if let Some(last) = active_package_names.last() {
                cycle.push(last.clone());
            }
            return Err(ProjectCompositionError::DependencyCycle {
                cycle: cycle.join(" -> "),
            });
        }

        let manifest = load_manifest(&manifest_path, from, is_root)?;
        let package_name = manifest.package.name.clone();
        let project_root = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let source_index =
            build_project_source_index(&project_root, &manifest).map_err(|source| {
                ProjectCompositionError::SourceIndex {
                    package: package_name.clone(),
                    source: Box::new(source),
                }
            })?;
        if let Some(existing) = self.packages.get(&package_name) {
            return existing_package_or_duplicate(existing, package_name, manifest_path);
        }

        active_paths.push(manifest_path.clone());
        active_package_names.push(package_name.clone());
        let mut dependencies = BTreeMap::new();
        for (dependency_name, dependency) in &manifest.dependencies {
            let Some(dependency_path) = dependency.path.as_ref() else {
                return Err(ProjectCompositionError::DependencyPathRequired {
                    package: package_name.clone(),
                    dependency: dependency_name.clone(),
                });
            };
            let dependency_root = project_root.join(dependency_path);
            let dependency_manifest_path = PROJECT_MANIFEST_FILENAMES
                .iter()
                .map(|filename| dependency_root.join(filename))
                .find(|candidate| candidate.is_file())
                .unwrap_or_else(|| dependency_root.join(PROJECT_MANIFEST_FILENAME));
            if !dependency_manifest_path.is_file() {
                return Err(ProjectCompositionError::MissingDependencyManifest {
                    package: package_name.clone(),
                    dependency: dependency_name.clone(),
                    path: dependency_manifest_path,
                });
            }
            let dependency_package = self.load_package(
                &dependency_manifest_path,
                Some((&package_name, dependency_name)),
                false,
                active_paths,
                active_package_names,
            )?;
            dependencies.insert(dependency_name.clone(), dependency_package);
        }
        active_paths.pop();
        active_package_names.pop();

        if let Some(existing) = self.packages.get(&package_name) {
            return existing_package_or_duplicate(existing, package_name, manifest_path);
        }
        self.insert_package(
            package_name.clone(),
            manifest_path,
            project_root,
            manifest,
            source_index,
            dependencies,
        );
        Ok(package_name)
    }

    fn insert_package(
        &mut self,
        package_name: String,
        manifest_path: PathBuf,
        project_root: PathBuf,
        manifest: ProjectManifest,
        source_index: ProjectSourceIndex,
        dependencies: BTreeMap<String, String>,
    ) {
        self.package_by_manifest
            .insert(manifest_path.clone(), package_name.clone());
        self.packages.insert(
            package_name.clone(),
            ProjectCompositionPackage {
                package_name,
                manifest_path,
                project_root,
                manifest,
                source_index,
                dependencies,
            },
        );
    }
}

fn load_manifest(
    manifest_path: &Path,
    from: Option<(&str, &str)>,
    is_root: bool,
) -> Result<ProjectManifest, ProjectCompositionError> {
    if is_root {
        load_project_manifest(manifest_path).map_err(|source| {
            ProjectCompositionError::RootManifestLoad {
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            }
        })
    } else {
        let (package, dependency) = from.expect("dependency context is required");
        load_project_manifest(manifest_path).map_err(|source| {
            ProjectCompositionError::DependencyManifestLoad {
                package: package.to_string(),
                dependency: dependency.to_string(),
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            }
        })
    }
}

fn existing_package_or_duplicate(
    existing: &ProjectCompositionPackage,
    package_name: String,
    manifest_path: PathBuf,
) -> Result<String, ProjectCompositionError> {
    if existing.manifest_path != manifest_path {
        Err(ProjectCompositionError::DuplicatePackageName {
            package: package_name,
            first_manifest: existing.manifest_path.clone(),
            second_manifest: manifest_path,
        })
    } else {
        Ok(existing.package_name.clone())
    }
}

type CompositionFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Default)]
struct AsyncCompositionGraphLoader {
    packages: BTreeMap<String, ProjectCompositionPackage>,
    package_by_manifest: BTreeMap<PathBuf, String>,
}

impl AsyncCompositionGraphLoader {
    fn load_package<'a>(
        &'a mut self,
        manifest_path: &'a Path,
        from: Option<(&'a str, &'a str)>,
        is_root: bool,
        active_paths: &'a mut Vec<PathBuf>,
        active_package_names: &'a mut Vec<String>,
    ) -> CompositionFuture<'a, Result<String, ProjectCompositionError>> {
        Box::pin(async move {
            let manifest_path = canonical_manifest_path_async(manifest_path).await;
            if let Some(existing) = self.package_by_manifest.get(&manifest_path) {
                return Ok(existing.clone());
            }
            if let Some(index) = active_paths.iter().position(|path| path == &manifest_path) {
                let mut cycle = active_package_names[index..].to_vec();
                if let Some(last) = active_package_names.last() {
                    cycle.push(last.clone());
                }
                return Err(ProjectCompositionError::DependencyCycle {
                    cycle: cycle.join(" -> "),
                });
            }

            let manifest = load_manifest_async(&manifest_path, from, is_root).await?;
            let package_name = manifest.package.name.clone();
            let project_root = manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let source_index = build_project_source_index_async(&project_root, &manifest)
                .await
                .map_err(|source| ProjectCompositionError::SourceIndex {
                    package: package_name.clone(),
                    source: Box::new(source),
                })?;
            if let Some(existing) = self.packages.get(&package_name) {
                return existing_package_or_duplicate(existing, package_name, manifest_path);
            }

            active_paths.push(manifest_path.clone());
            active_package_names.push(package_name.clone());
            let mut dependencies = BTreeMap::new();
            for (dependency_name, dependency) in &manifest.dependencies {
                let Some(dependency_path) = dependency.path.as_ref() else {
                    return Err(ProjectCompositionError::DependencyPathRequired {
                        package: package_name.clone(),
                        dependency: dependency_name.clone(),
                    });
                };
                let dependency_root = project_root.join(dependency_path);
                let dependency_manifest_path = first_existing_manifest_path_async(&dependency_root)
                    .await
                    .unwrap_or_else(|| dependency_root.join(PROJECT_MANIFEST_FILENAME));
                if !path_is_file_async(&dependency_manifest_path).await {
                    return Err(ProjectCompositionError::MissingDependencyManifest {
                        package: package_name.clone(),
                        dependency: dependency_name.clone(),
                        path: dependency_manifest_path,
                    });
                }
                let dependency_package = self
                    .load_package(
                        &dependency_manifest_path,
                        Some((&package_name, dependency_name)),
                        false,
                        active_paths,
                        active_package_names,
                    )
                    .await?;
                dependencies.insert(dependency_name.clone(), dependency_package);
            }
            active_paths.pop();
            active_package_names.pop();

            if let Some(existing) = self.packages.get(&package_name) {
                return existing_package_or_duplicate(existing, package_name, manifest_path);
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
                    dependencies,
                },
            );
            Ok(package_name)
        })
    }
}

async fn load_manifest_async(
    manifest_path: &Path,
    from: Option<(&str, &str)>,
    is_root: bool,
) -> Result<ProjectManifest, ProjectCompositionError> {
    if is_root {
        load_project_manifest_async(manifest_path)
            .await
            .map_err(|source| ProjectCompositionError::RootManifestLoad {
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            })
    } else {
        let (package, dependency) = from.expect("dependency context is required");
        load_project_manifest_async(manifest_path)
            .await
            .map_err(|source| ProjectCompositionError::DependencyManifestLoad {
                package: package.to_string(),
                dependency: dependency.to_string(),
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            })
    }
}

fn canonical_manifest_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

async fn canonical_manifest_path_async(path: &Path) -> PathBuf {
    runmat_filesystem::canonicalize_async(path)
        .await
        .unwrap_or_else(|_| path.to_path_buf())
}
