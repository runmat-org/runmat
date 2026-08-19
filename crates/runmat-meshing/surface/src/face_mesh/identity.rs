use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

pub(crate) fn canonical_triangle<T: Ord + Copy>(values: [T; 3]) -> ([T; 3], usize) {
    let rotation = (0..3)
        .min_by_key(|rotation| rotate(values, *rotation))
        .unwrap();
    (rotate(values, rotation), rotation)
}

pub(super) fn rotate<T: Copy>(values: [T; 3], rotation: usize) -> [T; 3] {
    [
        values[rotation % 3],
        values[(rotation + 1) % 3],
        values[(rotation + 2) % 3],
    ]
}

pub(crate) fn exact_face_triangle_id(
    chart_id: StableDigest,
    node_ids: [StableDigest; 3],
) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-face-mesh-triangle\0");
    digest.update(1u16.to_be_bytes());
    digest.update(chart_id.bytes());
    for node_id in node_ids {
        digest.update(node_id.bytes());
    }
    StableDigest::from_bytes(digest.finalize().into())
}
