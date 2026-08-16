use runmat_execution::Digest;

#[derive(Debug, thiserror::Error)]
pub enum MeshingExecutionError {
    #[error("meshing contract rejected execution artifact: {0}")]
    Meshing(#[from] runmat_meshing_core::MeshingContractError),
    #[error("shared execution artifact rejected meshing object: {0}")]
    Artifact(#[from] runmat_execution_artifact::ArtifactError),
    #[error("required meshing execution object is unavailable: {0}")]
    MissingObject(Digest),
    #[error("meshing execution object identity mismatch: {0}")]
    Identity(&'static str),
}

pub type MeshingExecutionResult<T> = Result<T, MeshingExecutionError>;
