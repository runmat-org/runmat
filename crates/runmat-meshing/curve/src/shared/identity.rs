use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

use super::SHARED_CURVE_MESH_SCHEMA_VERSION;

pub fn shared_curve_interior_node_id(edge_id: &PersistentEntityId, parameter: f64) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.shared-curve-interior-node\0");
    digest.update(SHARED_CURVE_MESH_SCHEMA_VERSION.to_be_bytes());
    write_bytes(&mut digest, edge_id.source_topology_id.as_bytes());
    digest.update((edge_id.assembly_path.len() as u64).to_be_bytes());
    for segment in &edge_id.assembly_path {
        write_bytes(&mut digest, segment.as_bytes());
    }
    digest.update(parameter.to_bits().to_be_bytes());
    StableDigest::from_bytes(digest.finalize().into())
}

pub fn shared_curve_vertex_node_id(vertex_id: &PersistentEntityId) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.shared-curve-vertex-node\0");
    digest.update(SHARED_CURVE_MESH_SCHEMA_VERSION.to_be_bytes());
    write_bytes(&mut digest, vertex_id.source_topology_id.as_bytes());
    digest.update((vertex_id.assembly_path.len() as u64).to_be_bytes());
    for segment in &vertex_id.assembly_path {
        write_bytes(&mut digest, segment.as_bytes());
    }
    StableDigest::from_bytes(digest.finalize().into())
}

fn write_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
