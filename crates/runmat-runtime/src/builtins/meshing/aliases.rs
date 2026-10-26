//! Region-alias sidecar written by `meshing.save` and read by `meshing.load`.
//!
//! Importer aliases (`face_000001`, `edge_000003`, `solid_000001`) bind model
//! region references to persistent CAD identities. They are not part of the
//! canonical mesh or evidence artifacts, so `meshing.save` publishes them in a
//! JSON sidecar bound to the mesh's canonical digest.

use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::error;
use crate::BuiltinResult;

const SIDECAR_SCHEMA: &str = "runmat.meshing.region-aliases/v1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AliasSidecar {
    schema: String,
    artifact_digest: String,
    aliases: BTreeMap<String, PersistentEntityId>,
}

/// The sidecar sits next to the mesh artifact: `<meshPath>.regions.json`.
pub(super) fn sidecar_path(mesh_path: &str) -> String {
    format!("{mesh_path}.regions.json")
}

pub(super) fn encode_sidecar(
    builtin: &'static str,
    artifact: &SolverMeshArtifact,
    aliases: &BTreeMap<String, PersistentEntityId>,
) -> BuiltinResult<Vec<u8>> {
    serde_json::to_vec_pretty(&AliasSidecar {
        schema: SIDECAR_SCHEMA.to_string(),
        artifact_digest: digest_hex(artifact),
        aliases: aliases.clone(),
    })
    .map_err(|cause| error(builtin, cause.to_string()))
}

/// Read the sidecar when present. A mesh written by `runmat mesh` has none and
/// loads without aliases; a sidecar that belongs to another mesh is rejected.
pub(super) fn read_sidecar(
    builtin: &'static str,
    mesh_path: &str,
    artifact: &SolverMeshArtifact,
) -> BuiltinResult<BTreeMap<String, PersistentEntityId>> {
    let path = sidecar_path(mesh_path);
    let bytes = match runmat_filesystem::read(&path) {
        Ok(bytes) => bytes,
        Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(cause) => return Err(error(builtin, format!("failed to read {path}: {cause}"))),
    };
    let sidecar: AliasSidecar = serde_json::from_slice(&bytes).map_err(|cause| {
        error(
            builtin,
            format!("invalid region-alias sidecar {path}: {cause}"),
        )
    })?;
    if sidecar.schema != SIDECAR_SCHEMA {
        return Err(error(
            builtin,
            format!("unsupported region-alias sidecar schema in {path}"),
        ));
    }
    if sidecar.artifact_digest != digest_hex(artifact) {
        return Err(error(
            builtin,
            format!("region-alias sidecar {path} belongs to a different solver mesh"),
        ));
    }
    Ok(sidecar.aliases)
}

fn digest_hex(artifact: &SolverMeshArtifact) -> String {
    artifact
        .canonical_digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
