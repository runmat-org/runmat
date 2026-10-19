use super::loader::{
    LoadedJavaArtifact, LoadedMexArtifact, LoadedNativeInterface, LoadedPythonArtifact,
    LoadedSource, PackageOrigin,
};
use super::ProjectResolveError;
use crate::{ContentDigest, NormalizedRelativePath, PathSourceId, SourceId};
use runmat_config::project::{
    build_project_source_index_async, ProjectManifest, PROJECT_MANIFEST_FILENAMES,
};
use std::path::{Path, PathBuf};

pub(super) async fn load_sources(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<
    (
        Vec<LoadedSource>,
        runmat_config::project::ProjectSourceIndex,
    ),
    ProjectResolveError,
> {
    let index = build_project_source_index_async(root, manifest)
        .await
        .map_err(|error| ProjectResolveError::SourceInventory {
            package: manifest.package.name.clone(),
            reason: error.to_string(),
        })?;
    let mut sources = Vec::with_capacity(index.files.len());
    for descriptor in &index.files {
        let path = root
            .join(&descriptor.source_root)
            .join(&descriptor.relative_path);
        let bytes = runmat_filesystem::read_async(&path)
            .await
            .map_err(|error| ProjectResolveError::SourceRead {
                path,
                reason: error.to_string(),
            })?;
        sources.push(LoadedSource {
            descriptor: descriptor.clone(),
            bytes,
        });
    }
    Ok((sources, index))
}

pub(super) async fn load_native_interfaces(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<Vec<LoadedNativeInterface>, ProjectResolveError> {
    let mut interfaces = Vec::with_capacity(manifest.native_interfaces.len());
    for (name, declaration) in &manifest.native_interfaces {
        let manifest_path = root.join(&declaration.manifest);
        let manifest_bytes = read_native_artifact(&manifest_path).await?;
        let library_path = root.join(&declaration.library);
        let library_bytes = read_native_artifact(&library_path).await?;
        interfaces.push(LoadedNativeInterface {
            name: name.clone(),
            manifest_path,
            manifest_bytes,
            library_path,
            library_bytes,
        });
    }
    Ok(interfaces)
}

pub(super) async fn load_java_artifacts(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<Vec<LoadedJavaArtifact>, ProjectResolveError> {
    let mut artifacts = Vec::with_capacity(manifest.java_artifacts.len());
    for (name, declaration) in &manifest.java_artifacts {
        let path = root.join(&declaration.path);
        let bytes = read_project_artifact(&path).await?;
        artifacts.push(LoadedJavaArtifact {
            name: name.clone(),
            path,
            bytes,
        });
    }
    Ok(artifacts)
}

pub(super) async fn load_python_artifacts(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<Vec<LoadedPythonArtifact>, ProjectResolveError> {
    let mut artifacts = Vec::with_capacity(manifest.python_artifacts.len());
    for (name, declaration) in &manifest.python_artifacts {
        let path = root.join(&declaration.path);
        let bytes = read_project_artifact(&path).await?;
        artifacts.push(LoadedPythonArtifact {
            name: name.clone(),
            module: declaration.module.clone(),
            path,
            bytes,
        });
    }
    Ok(artifacts)
}

pub(super) async fn load_mex_artifacts(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<Vec<LoadedMexArtifact>, ProjectResolveError> {
    let mut artifacts = Vec::with_capacity(manifest.mex_artifacts.len());
    for (name, declaration) in &manifest.mex_artifacts {
        let manifest_path = root.join(&declaration.manifest);
        let manifest_bytes = read_project_artifact(&manifest_path).await?;
        let module_path = root.join(&declaration.module);
        let module_bytes = read_project_artifact(&module_path).await?;
        artifacts.push(LoadedMexArtifact {
            name: name.clone(),
            manifest_path,
            manifest_bytes,
            module_path,
            module_bytes,
        });
    }
    Ok(artifacts)
}

pub(super) struct PackageContent<'a> {
    pub(super) sources: &'a [LoadedSource],
    pub(super) native_interfaces: &'a [LoadedNativeInterface],
    pub(super) mex_artifacts: &'a [LoadedMexArtifact],
    pub(super) java_artifacts: &'a [LoadedJavaArtifact],
    pub(super) python_artifacts: &'a [LoadedPythonArtifact],
}

async fn read_native_artifact(path: &Path) -> Result<Vec<u8>, ProjectResolveError> {
    runmat_filesystem::read_async(path)
        .await
        .map_err(|error| ProjectResolveError::SourceRead {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })
}

async fn read_project_artifact(path: &Path) -> Result<Vec<u8>, ProjectResolveError> {
    runmat_filesystem::read_async(path)
        .await
        .map_err(|error| ProjectResolveError::SourceRead {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })
}

pub(super) fn source_identity(
    workspace_root: &Path,
    manifest_path: &Path,
    root: &Path,
    manifest: &ProjectManifest,
    content: PackageContent<'_>,
    origin: &PackageOrigin,
) -> Result<SourceId, ProjectResolveError> {
    if let PackageOrigin::Git(source) = origin {
        return Ok(SourceId::Git(source.clone()));
    }
    if let PackageOrigin::ServerProject(source) = origin {
        return Ok(SourceId::ServerProject(source.clone()));
    }
    if let PackageOrigin::Registry(source) = origin {
        return Ok(SourceId::Registry(source.clone()));
    }
    if let PackageOrigin::Vendor(expected) = origin {
        let SourceId::Path(expected_path) = expected else {
            return Err(ProjectResolveError::Invalid(
                "vendor override currently requires a locked path source".to_string(),
            ));
        };
        let canonical_manifest = toml::to_string(manifest).map_err(|error| {
            ProjectResolveError::Invalid(format!(
                "cannot encode manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let manifest_digest = ContentDigest::sha256(canonical_manifest);
        let tree_digest = path_tree_digest(root, &content)?;
        if manifest_digest != expected_path.manifest_digest
            || tree_digest != expected_path.tree_digest
        {
            return Err(ProjectResolveError::Invalid(format!(
                "vendored package at {} does not match its locked manifest and tree digests",
                root.display()
            )));
        }
        return Ok(expected.clone());
    }
    let relative = root.strip_prefix(workspace_root).map_err(|_| {
        ProjectResolveError::Invalid(format!(
            "path package {} is outside workspace {}",
            root.display(),
            workspace_root.display()
        ))
    })?;
    let workspace_path = NormalizedRelativePath::new(relative)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    let canonical_manifest = toml::to_string(manifest).map_err(|error| {
        ProjectResolveError::Invalid(format!(
            "cannot encode manifest {}: {error}",
            manifest_path.display()
        ))
    })?;
    Ok(SourceId::Path(PathSourceId {
        workspace_path,
        manifest_digest: ContentDigest::sha256(canonical_manifest),
        tree_digest: path_tree_digest(root, &content)?,
    }))
}

fn path_tree_digest(
    root: &Path,
    content: &PackageContent<'_>,
) -> Result<ContentDigest, ProjectResolveError> {
    let mut input = Vec::new();
    for source in content.sources {
        let path = NormalizedRelativePath::new(
            source
                .descriptor
                .source_root
                .join(&source.descriptor.relative_path),
        )
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
        input.extend_from_slice(path.as_str().as_bytes());
        input.push(0);
        input.extend_from_slice(source.bytes.len().to_string().as_bytes());
        input.push(0);
        input.extend_from_slice(&source.bytes);
        input.push(0);
    }
    for interface in content.native_interfaces {
        append_native_artifact(
            &mut input,
            root,
            &interface.manifest_path,
            &interface.manifest_bytes,
        )?;
        append_native_artifact(
            &mut input,
            root,
            &interface.library_path,
            &interface.library_bytes,
        )?;
    }
    for artifact in content.mex_artifacts {
        append_project_artifact(
            &mut input,
            root,
            &artifact.manifest_path,
            &artifact.manifest_bytes,
        )?;
        append_project_artifact(
            &mut input,
            root,
            &artifact.module_path,
            &artifact.module_bytes,
        )?;
    }
    for artifact in content.java_artifacts {
        append_project_artifact(&mut input, root, &artifact.path, &artifact.bytes)?;
    }
    for artifact in content.python_artifacts {
        append_project_artifact(&mut input, root, &artifact.path, &artifact.bytes)?;
    }
    Ok(ContentDigest::sha256(input))
}

fn append_project_artifact(
    input: &mut Vec<u8>,
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> Result<(), ProjectResolveError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        ProjectResolveError::Invalid(format!(
            "project artifact {} is outside package root {}",
            path.display(),
            root.display()
        ))
    })?;
    let path = NormalizedRelativePath::new(relative)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    input.extend_from_slice(path.as_str().as_bytes());
    input.push(0);
    input.extend_from_slice(bytes.len().to_string().as_bytes());
    input.push(0);
    input.extend_from_slice(bytes);
    input.push(0);
    Ok(())
}

fn append_native_artifact(
    input: &mut Vec<u8>,
    root: &Path,
    path: &Path,
    bytes: &[u8],
) -> Result<(), ProjectResolveError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        ProjectResolveError::Invalid(format!(
            "native interface artifact {} is outside package root {}",
            path.display(),
            root.display()
        ))
    })?;
    let path = NormalizedRelativePath::new(relative)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    input.extend_from_slice(path.as_str().as_bytes());
    input.push(0);
    input.extend_from_slice(bytes.len().to_string().as_bytes());
    input.push(0);
    input.extend_from_slice(bytes);
    input.push(0);
    Ok(())
}

pub(super) async fn find_manifest(root: &Path) -> Option<PathBuf> {
    for filename in PROJECT_MANIFEST_FILENAMES {
        let candidate = root.join(filename);
        if is_file(&candidate).await {
            return Some(candidate);
        }
    }
    None
}

pub(super) async fn is_file(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
}

pub(super) async fn canonical_path(path: &Path) -> PathBuf {
    runmat_filesystem::canonicalize_async(path)
        .await
        .unwrap_or_else(|_| path.to_path_buf())
}
