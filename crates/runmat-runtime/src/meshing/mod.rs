//! Runtime composition boundary for canonical meshing.

mod persistence;
#[cfg(feature = "plot-core")]
mod visualization;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use runmat_geometry_core::LengthUnit;
use runmat_meshing_core::{MeshingEvidence, MeshingRequestSettings, SolverMeshArtifact};

pub use persistence::{write_canonical_artifact_pair, write_canonical_artifacts};
#[cfg(feature = "plot-core")]
pub use visualization::{solver_mesh_figure, SolverMeshFigureOptions};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingContactDefinition {
    pub side_a_face_selectors: Vec<String>,
    pub side_b_face_selectors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingExecutionRequest {
    pub source_path: PathBuf,
    pub source_units: Option<LengthUnit>,
    pub expected_source_sha256: [u8; 32],
    pub settings: MeshingRequestSettings,
    pub contact_definitions: Vec<MeshingContactDefinition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingExecutionOutput {
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshingExecutionError {
    #[error("{0}")]
    Admission(#[from] crate::analysis::fea_run_admission::FeaRunDenied),
    #[error("{0}")]
    Provider(String),
}

pub type MeshingExecutionProvider = Arc<
    dyn Fn(&MeshingExecutionRequest) -> Result<MeshingExecutionOutput, String>
        + Send
        + Sync
        + 'static,
>;

fn provider_slot() -> &'static RwLock<Option<MeshingExecutionProvider>> {
    static PROVIDER: OnceLock<RwLock<Option<MeshingExecutionProvider>>> = OnceLock::new();
    PROVIDER.get_or_init(|| RwLock::new(None))
}

pub struct MeshingExecutionProviderGuard {
    previous: Option<MeshingExecutionProvider>,
}

impl Drop for MeshingExecutionProviderGuard {
    fn drop(&mut self) {
        if let Ok(mut provider) = provider_slot().write() {
            *provider = self.previous.take();
        }
    }
}

pub fn replace_meshing_execution_provider(
    provider: Option<MeshingExecutionProvider>,
) -> Result<MeshingExecutionProviderGuard, String> {
    let mut slot = provider_slot()
        .write()
        .map_err(|_| "meshing execution provider lock poisoned".to_string())?;
    let previous = std::mem::replace(&mut *slot, provider);
    Ok(MeshingExecutionProviderGuard { previous })
}

pub fn execute_meshing(
    request: &MeshingExecutionRequest,
) -> Result<MeshingExecutionOutput, MeshingExecutionError> {
    use crate::analysis::fea_run_admission::{
        FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
    };

    // Native standalone meshing reserves this worker budget. Under a study (or a browser
    // host's scope), the provider must honor the already admitted parent allocation.
    let allocation = FeaRunScope::current_allocation().unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(runmat_cpu_execution::MAX_THREADS) as u64
            * 1000
    });
    let admission = FeaRunRequest::local(
        FeaRunKind::Meshing,
        allocation,
        runmat_execution::Digest::from_bytes(request.expected_source_sha256),
    )?;
    let scope = FeaRunScope::admit(admission)?;
    let result = scope
        .run_sync(|| execute_meshing_admitted(request))
        .map_err(MeshingExecutionError::Provider);
    scope.settle(if crate::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.is_ok() {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    });
    result
}

fn execute_meshing_admitted(
    request: &MeshingExecutionRequest,
) -> Result<MeshingExecutionOutput, String> {
    let provider = provider_slot()
        .read()
        .map_err(|_| "meshing execution provider lock poisoned".to_string())?
        .clone()
        .ok_or_else(|| {
            "this host cannot execute meshing; use a native RunMat host or configure compatible execution capacity".to_string()
        })?;
    provider(request)
}

pub fn parse_source_sha256(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err("geometry asset is missing its canonical source SHA-256".to_string());
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| "geometry asset has an invalid source SHA-256".to_string())?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::parse_source_sha256;

    #[test]
    fn source_digest_parser_requires_exact_sha256_hex() {
        assert_eq!(parse_source_sha256(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        assert!(parse_source_sha256("").is_err());
        assert!(parse_source_sha256(&"zz".repeat(32)).is_err());
    }
}
