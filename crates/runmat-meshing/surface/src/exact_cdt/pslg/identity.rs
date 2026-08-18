use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

const EXACT_FACE_INTERIOR_IDENTITY_VERSION: u16 = 1;

pub fn exact_face_interior_node_id(face_id: &PersistentEntityId, uv: [f64; 2]) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-face-interior-node\0");
    digest.update(EXACT_FACE_INTERIOR_IDENTITY_VERSION.to_be_bytes());
    write_bytes(&mut digest, face_id.source_topology_id.as_bytes());
    digest.update((face_id.assembly_path.len() as u64).to_be_bytes());
    for segment in &face_id.assembly_path {
        write_bytes(&mut digest, segment.as_bytes());
    }
    for coordinate in uv {
        digest.update(coordinate.to_bits().to_be_bytes());
    }
    StableDigest::from_bytes(digest.finalize().into())
}

fn write_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}
