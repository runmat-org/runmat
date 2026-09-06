//! Runtime composition boundary for canonical meshing.

mod persistence;
#[cfg(feature = "plot-core")]
mod visualization;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use runmat_geometry_core::{PersistentEntityId, UnitSystem};
use runmat_meshing_core::{MeshingEvidence, MeshingRequestSettings, SolverMeshArtifact};

pub use persistence::write_canonical_artifact_pair;
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
    pub source_units: UnitSystem,
    pub expected_source_sha256: [u8; 32],
    pub settings: MeshingRequestSettings,
    pub contact_definitions: Vec<MeshingContactDefinition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshingExecutionOutput {
    pub artifact: SolverMeshArtifact,
    pub evidence: MeshingEvidence,
    pub boundary_region_ids: BTreeMap<String, PersistentEntityId>,
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
