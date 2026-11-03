use std::{fmt, path::Path};

use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

thread_local! {
    static SCOPED_INPUT: std::cell::RefCell<Option<(std::path::PathBuf, SolverMeshArtifact)>> = const { std::cell::RefCell::new(None) };
}

/// Own one original input for a numerical worker without persisting worker output.
pub(super) fn with_scoped_solver_mesh<T>(
    path: std::path::PathBuf,
    mesh: SolverMeshArtifact,
    work: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<(std::path::PathBuf, SolverMeshArtifact)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPED_INPUT.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(SCOPED_INPUT.with(|slot| slot.replace(Some((path, mesh)))));
    work()
}

#[derive(Debug)]
pub(super) enum SolverMeshArtifactLoadError {
    Read(std::io::Error),
    Invalid(runmat_meshing_core::MeshingContractError),
}

impl fmt::Display for SolverMeshArtifactLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "failed to read solver mesh artifact: {error}"),
            Self::Invalid(error) => write!(f, "solver mesh artifact is invalid: {error}"),
        }
    }
}

impl std::error::Error for SolverMeshArtifactLoadError {}

/// Read through the original asynchronous authority before entering numerical
/// work. Result-tree inputs use the captured artifact provider; authored inputs
/// outside that tree retain their original filesystem authority.
pub(super) async fn load_solver_mesh_artifact_async(
    path: &Path,
) -> Result<SolverMeshArtifact, SolverMeshArtifactLoadError> {
    let store = super::storage::capture_artifact_store()
        .map_err(|error| SolverMeshArtifactLoadError::Read(std::io::Error::other(error)))?;
    let (provider, path) = match store.filesystem_scope() {
        Some((root, provider))
            if path.starts_with(&root)
                && !path
                    .components()
                    .any(|part| matches!(part, std::path::Component::ParentDir)) =>
        {
            let resolved = if provider.path_namespace().is_absolute(path) {
                path.to_path_buf()
            } else {
                provider
                    .current_dir_override()
                    .map_or_else(runmat_filesystem::current_dir, Ok)
                    .map_err(SolverMeshArtifactLoadError::Read)?
                    .join(path)
            };
            (provider, resolved)
        }
        _ => runmat_filesystem::capture_provider_path(path)
            .map_err(SolverMeshArtifactLoadError::Read)?,
    };
    let maximum = SolverMeshArtifact::LIMITS.maximum_encoded_bytes as u64;
    let oversized = || {
        SolverMeshArtifactLoadError::Invalid(runmat_meshing_core::MeshingContractError::invalid(
            "solver mesh canonical bytes",
            format!("encoded input exceeds the canonical format limit of {maximum} bytes"),
        ))
    };
    // Preflight and verify the provider's returned extent against the original
    // canonical wire limit. Publication configuration is not input authority.
    let metadata = provider
        .metadata(&path)
        .await
        .map_err(SolverMeshArtifactLoadError::Read)?;
    if metadata.len() > maximum {
        return Err(oversized());
    }
    let bytes = provider
        .read(&path)
        .await
        .map_err(SolverMeshArtifactLoadError::Read)?;
    if bytes.len() as u64 > maximum {
        return Err(oversized());
    }
    // This is the cancellation-aware form of the synchronous loader's decoder:
    // the same format/materialization limits and independent sealed validation,
    // without inventing another allocation allowance or aggregate size cap.
    SolverMeshArtifact::canonical_decode_with_cancellation(&bytes, &MeshInputControl)
        .map_err(SolverMeshArtifactLoadError::Invalid)
}

struct MeshInputControl;
impl runmat_meshing_core::MeshingCancellationSignal for MeshInputControl {
    fn is_cancelled(&self) -> bool {
        crate::interrupt::is_cancelled()
    }
}

/// Reads and independently revalidates the canonical solver contract.
///
/// The canonical codec is the only accepted persisted representation. JSON payloads from
/// non-authoritative producers are rejected rather than inferred or upgraded.
pub(super) fn load_solver_mesh_artifact(
    path: &Path,
) -> Result<SolverMeshArtifact, SolverMeshArtifactLoadError> {
    if let Some(mesh) = SCOPED_INPUT.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|(original, _)| original == path)
            .map(|(_, mesh)| mesh.clone())
    }) {
        mesh.validate_canonical()
            .map_err(SolverMeshArtifactLoadError::Invalid)?;
        return Ok(mesh);
    }
    let bytes = runmat_filesystem::read(path).map_err(SolverMeshArtifactLoadError::Read)?;
    SolverMeshArtifact::canonical_decode(&bytes).map_err(SolverMeshArtifactLoadError::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_json_instead_of_guessing_an_obsolete_mesh_contract() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("mesh.json");
        runmat_filesystem::write(path.clone(), br#"{"mesh":{"nodes":[]}}"#).unwrap();

        assert!(matches!(
            load_solver_mesh_artifact(&path),
            Err(SolverMeshArtifactLoadError::Invalid(_))
        ));
    }
}
