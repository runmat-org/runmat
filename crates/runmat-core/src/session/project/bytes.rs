//! Freeze validated project foreign bytes before any adapter preparation.

use std::path::{Path, PathBuf};

use runmat_execution_artifact::PrivateArtifactRoot;
use runmat_package::{ContentDigest, FrozenProjectHandoff, FrozenProjectHandoffError};

use super::project_artifact_error;

pub(super) struct ProjectPrivateBytes {
    root: PrivateArtifactRoot,
}

impl ProjectPrivateBytes {
    pub(super) fn materialize(
        handoff: &mut FrozenProjectHandoff,
    ) -> Result<Option<Self>, FrozenProjectHandoffError> {
        if handoff.project.native_interfaces.is_empty()
            && handoff.project.mex_artifacts.is_empty()
            && handoff.project.java_artifacts.is_empty()
            && handoff.project.python_artifacts.is_empty()
        {
            return Ok(None);
        }
        let mut root =
            PrivateArtifactRoot::new("runmat-project-foreign-").map_err(project_artifact_error)?;
        for (index, artifact) in handoff.project.native_interfaces.iter_mut().enumerate() {
            freeze_pair(
                &mut root,
                "native",
                index,
                ArtifactFile {
                    path: &mut artifact.library_path,
                    digest: &artifact.library_digest,
                },
                ArtifactFile {
                    path: &mut artifact.manifest_path,
                    digest: &artifact.manifest_digest,
                },
                runmat_native_ffi::NativeInterfaceArtifactManifest::path_for_library,
            )?;
        }
        for (index, artifact) in handoff.project.mex_artifacts.iter_mut().enumerate() {
            freeze_pair(
                &mut root,
                "mex",
                index,
                ArtifactFile {
                    path: &mut artifact.module_path,
                    digest: &artifact.module_digest,
                },
                ArtifactFile {
                    path: &mut artifact.manifest_path,
                    digest: &artifact.manifest_digest,
                },
                runmat_mex::MexArtifactManifest::path_for_module,
            )?;
        }
        for (index, artifact) in handoff.project.java_artifacts.iter_mut().enumerate() {
            freeze_one(
                &mut root,
                "java",
                index,
                &mut artifact.path,
                &artifact.digest,
            )?;
        }
        for (index, artifact) in handoff.project.python_artifacts.iter_mut().enumerate() {
            freeze_one(
                &mut root,
                "python",
                index,
                &mut artifact.path,
                &artifact.digest,
            )?;
        }
        Ok(Some(Self { root }))
    }

    pub(super) fn lease(&self) -> runmat_execution_artifact::PrivateArtifactLease {
        self.root.lease()
    }

    pub(super) fn into_root(self) -> PrivateArtifactRoot {
        self.root
    }
}

struct ArtifactFile<'a> {
    path: &'a mut PathBuf,
    digest: &'a ContentDigest,
}

fn freeze_pair(
    root: &mut PrivateArtifactRoot,
    family: &str,
    index: usize,
    binary: ArtifactFile<'_>,
    sidecar: ArtifactFile<'_>,
    sidecar_for_binary: fn(&Path) -> PathBuf,
) -> Result<(), FrozenProjectHandoffError> {
    let binary_bytes = read_exact(binary.path, binary.digest)?;
    let sidecar_bytes = read_exact(sidecar.path, sidecar.digest)?;
    let filename = binary
        .path
        .file_name()
        .and_then(|filename| filename.to_str())
        .ok_or_else(|| project_artifact_error("foreign binary has no UTF-8 filename"))?;
    let relative_binary = format!("{family}/{index}/{filename}");
    let relative_sidecar = sidecar_for_binary(Path::new(&relative_binary));
    let relative_sidecar = relative_sidecar
        .to_str()
        .ok_or_else(|| project_artifact_error("foreign sidecar has no UTF-8 filename"))?;
    let issued_binary = root
        .write_verified(&relative_binary, &binary_bytes)
        .map_err(project_artifact_error)?;
    let issued_sidecar = root
        .write_verified(relative_sidecar, &sidecar_bytes)
        .map_err(project_artifact_error)?;
    *binary.path = issued_binary.as_path().to_path_buf();
    *sidecar.path = issued_sidecar.as_path().to_path_buf();
    Ok(())
}

fn freeze_one(
    root: &mut PrivateArtifactRoot,
    family: &str,
    index: usize,
    path: &mut PathBuf,
    expected: &ContentDigest,
) -> Result<(), FrozenProjectHandoffError> {
    let bytes = read_exact(path, expected)?;
    let filename = path
        .file_name()
        .and_then(|filename| filename.to_str())
        .ok_or_else(|| project_artifact_error("foreign artifact has no UTF-8 filename"))?;
    let relative = format!("{family}/{index}/{filename}");
    let issued = root
        .write_verified(&relative, &bytes)
        .map_err(project_artifact_error)?;
    *path = issued.as_path().to_path_buf();
    Ok(())
}

fn read_exact(path: &Path, expected: &ContentDigest) -> Result<Vec<u8>, FrozenProjectHandoffError> {
    let bytes = std::fs::read(path).map_err(project_artifact_error)?;
    let actual = ContentDigest::sha256(&bytes);
    if &actual != expected {
        return Err(project_artifact_error(format!(
            "foreign artifact {} changed after project resolution: expected {expected}, found {actual}",
            path.display()
        )));
    }
    Ok(bytes)
}
