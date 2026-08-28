use sha2::{Digest, Sha256};

use super::StableDigest;

const MIDSIDE_NODE_DOMAIN: &[u8] = b"runmat/meshing/solver-midside-node/1\0";
const VOLUME_ELEMENT_DOMAIN: &[u8] = b"runmat/meshing/solver-volume-element/1\0";
const BOUNDARY_FACE_DOMAIN: &[u8] = b"runmat/meshing/solver-boundary-face/1\0";
const BOUNDARY_EDGE_DOMAIN: &[u8] = b"runmat/meshing/solver-boundary-edge/1\0";

pub fn solver_midside_node_identity(endpoints: [StableDigest; 2]) -> StableDigest {
    simplex_identity(MIDSIDE_NODE_DOMAIN, endpoints)
}

pub fn solver_volume_element_identity(corners: [StableDigest; 4]) -> StableDigest {
    simplex_identity(VOLUME_ELEMENT_DOMAIN, corners)
}

pub fn solver_boundary_face_identity(corners: [StableDigest; 3]) -> StableDigest {
    simplex_identity(BOUNDARY_FACE_DOMAIN, corners)
}

pub fn solver_boundary_edge_identity(endpoints: [StableDigest; 2]) -> StableDigest {
    simplex_identity(BOUNDARY_EDGE_DOMAIN, endpoints)
}

fn simplex_identity<const N: usize>(domain: &[u8], mut nodes: [StableDigest; N]) -> StableDigest {
    nodes.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for node in nodes {
        hasher.update(node.bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}
