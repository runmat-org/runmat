use super::model::{LoadedPathPackage, LoadedPathProject, LoadedSource};
use super::FrozenProjectError;
use runmat_config::project::{
    build_project_source_index_async, load_project_manifest_async, ProjectManifest,
    PROJECT_MANIFEST_FILENAME, PROJECT_MANIFEST_FILENAMES,
};
use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

pub(super) async fn load_path_project(
    root_manifest_path: &Path,
) -> Result<LoadedPathProject, FrozenProjectError> {
    let root_manifest = canonical_manifest_path(root_manifest_path).await;
    let workspace_root = root_manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut loader = PathProjectLoader::default();
    loader
        .load_package(&root_manifest, None, true, &mut Vec::new())
        .await?;
    Ok(LoadedPathProject {
        root_manifest,
        workspace_root,
        packages: loader.packages,
    })
}

#[derive(Default)]
struct PathProjectLoader {
    packages: BTreeMap<PathBuf, LoadedPathPackage>,
}

type LoadFuture<'a> = Pin<Box<dyn Future<Output = Result<PathBuf, FrozenProjectError>> + 'a>>;

impl PathProjectLoader {
    fn load_package<'a>(
        &'a mut self,
        manifest_path: &'a Path,
        from: Option<(&'a str, &'a str)>,
        is_root: bool,
        active: &'a mut Vec<(PathBuf, String)>,
    ) -> LoadFuture<'a> {
        Box::pin(async move {
            let manifest_path = canonical_manifest_path(manifest_path).await;
            if let Some((index, _)) = active
                .iter()
                .enumerate()
                .find(|(_, (path, _))| path == &manifest_path)
            {
                let mut cycle = active[index..]
                    .iter()
                    .map(|(_, package)| package.clone())
                    .collect::<Vec<_>>();
                if let Some((_, package)) = active.get(index) {
                    cycle.push(package.clone());
                }
                return Err(FrozenProjectError::DependencyCycle {
                    cycle: cycle.join(" -> "),
                });
            }
            if self.packages.contains_key(&manifest_path) {
                return Ok(manifest_path);
            }

            let manifest = load_manifest(&manifest_path, from, is_root).await?;
            let package_name = manifest.package.name.clone();
            let project_root = manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let source_index = build_project_source_index_async(&project_root, &manifest)
                .await
                .map_err(|source| FrozenProjectError::SourceIndex {
                    package: package_name.clone(),
                    source: Box::new(source),
                })?;
            let mut sources = Vec::with_capacity(source_index.files.len());
            for descriptor in source_index.files {
                let path = project_root
                    .join(&descriptor.source_root)
                    .join(&descriptor.relative_path);
                let bytes = runmat_filesystem::read_async(&path)
                    .await
                    .map_err(|error| super::PathProjectError::ReadSource {
                        path: path.clone(),
                        reason: error.to_string(),
                    })?;
                sources.push(LoadedSource { descriptor, bytes });
            }

            active.push((manifest_path.clone(), package_name.clone()));
            let mut dependencies = BTreeMap::new();
            for (alias, dependency) in &manifest.dependencies {
                let Some(dependency_path) = dependency.path.as_ref() else {
                    return Err(FrozenProjectError::DependencyPathRequired {
                        package: package_name.clone(),
                        dependency: alias.clone(),
                    });
                };
                let dependency_root = project_root.join(dependency_path);
                let dependency_manifest = first_manifest_path(&dependency_root)
                    .await
                    .unwrap_or_else(|| dependency_root.join(PROJECT_MANIFEST_FILENAME));
                if !is_file(&dependency_manifest).await {
                    return Err(FrozenProjectError::MissingDependencyManifest {
                        package: package_name.clone(),
                        dependency: alias.clone(),
                        path: dependency_manifest,
                    });
                }
                let target = self
                    .load_package(
                        &dependency_manifest,
                        Some((&package_name, alias)),
                        false,
                        active,
                    )
                    .await?;
                dependencies.insert(alias.clone(), target);
            }
            active.pop();

            self.packages.insert(
                manifest_path.clone(),
                LoadedPathPackage {
                    manifest_path: manifest_path.clone(),
                    project_root,
                    manifest,
                    sources,
                    dependencies,
                },
            );
            Ok(manifest_path)
        })
    }
}

async fn load_manifest(
    manifest_path: &Path,
    from: Option<(&str, &str)>,
    is_root: bool,
) -> Result<ProjectManifest, FrozenProjectError> {
    if is_root {
        load_project_manifest_async(manifest_path)
            .await
            .map_err(|source| FrozenProjectError::RootManifestLoad {
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            })
    } else {
        let (package, dependency) = from.expect("dependency context");
        load_project_manifest_async(manifest_path)
            .await
            .map_err(|source| FrozenProjectError::DependencyManifestLoad {
                package: package.to_string(),
                dependency: dependency.to_string(),
                path: manifest_path.to_path_buf(),
                source: Box::new(source),
            })
    }
}

async fn first_manifest_path(root: &Path) -> Option<PathBuf> {
    for filename in PROJECT_MANIFEST_FILENAMES {
        let candidate = root.join(filename);
        if is_file(&candidate).await {
            return Some(candidate);
        }
    }
    None
}

async fn is_file(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
}

async fn canonical_manifest_path(path: &Path) -> PathBuf {
    runmat_filesystem::canonicalize_async(path)
        .await
        .unwrap_or_else(|_| path.to_path_buf())
}
