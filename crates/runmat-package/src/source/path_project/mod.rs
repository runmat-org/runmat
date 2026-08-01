mod freeze;
mod load;
mod model;

use crate::{GraphError, HostCapability};
#[cfg(not(target_arch = "wasm32"))]
use runmat_config::project::discover_project_manifest_from;
use runmat_config::project::{
    discover_project_manifest_from_async, ProjectManifestLoadError, ProjectSourceIndexError,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub use freeze::{build_frozen_project_async, PathProjectError};

use super::FrozenProject;

#[derive(Debug, Error)]
pub enum FrozenProjectError {
    #[error(transparent)]
    Project(#[from] PathProjectError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("failed to load root project manifest {path}: {source}")]
    RootManifestLoad {
        path: PathBuf,
        #[source]
        source: Box<ProjectManifestLoadError>,
    },
    #[error("failed to load dependency manifest {path} for dependency `{dependency}` of package `{package}`: {source}")]
    DependencyManifestLoad {
        package: String,
        dependency: String,
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
    #[error("dependency `{dependency}` in package `{package}` does not define a local `path`")]
    DependencyPathRequired { package: String, dependency: String },
    #[error("failed to build source index for package `{package}`: {source}")]
    SourceIndex {
        package: String,
        #[source]
        source: Box<ProjectSourceIndexError>,
    },
    #[error("dependency cycle detected while loading project composition: {cycle}")]
    DependencyCycle { cycle: String },
    #[error("synchronous project freezing is unavailable on WebAssembly; use the async API")]
    SyncUnavailable,
}

pub fn discover_frozen_project_from(
    start: &Path,
    host_capabilities: BTreeSet<HostCapability>,
) -> Result<Option<FrozenProject>, FrozenProjectError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(manifest_path) = discover_project_manifest_from(start) else {
            return Ok(None);
        };
        build_frozen_project(&manifest_path, host_capabilities).map(Some)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = start;
        let _ = host_capabilities;
        Ok(None)
    }
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
    #[cfg(not(target_arch = "wasm32"))]
    {
        futures::executor::block_on(build_frozen_project_async(manifest_path, host_capabilities))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = manifest_path;
        let _ = host_capabilities;
        Err(FrozenProjectError::SyncUnavailable)
    }
}
