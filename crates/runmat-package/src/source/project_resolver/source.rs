use super::loader::{LoadedSource, PackageOrigin};
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

pub(super) fn source_identity(
    workspace_root: &Path,
    manifest_path: &Path,
    root: &Path,
    manifest: &ProjectManifest,
    sources: &[LoadedSource],
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
        let tree_digest = path_tree_digest(sources)?;
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
        tree_digest: path_tree_digest(sources)?,
    }))
}

fn path_tree_digest(sources: &[LoadedSource]) -> Result<ContentDigest, ProjectResolveError> {
    let mut input = Vec::new();
    for source in sources {
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
    Ok(ContentDigest::sha256(input))
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
