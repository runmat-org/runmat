mod build;
mod types;
mod validate;

use runmat_meshing_core::{PlanarPredicatePoint, StableDigest};

use crate::ExactFacePslgVertex;

pub use build::triangulate_exact_face_pslg;
pub use types::{
    ExactFaceDelaunay, ExactFaceDelaunayError, ExactFaceDelaunayErrorKind,
    ExactFaceDelaunayOptions, ExactFaceDelaunayTriangle,
};
pub use validate::validate_exact_face_delaunay;

#[cfg(test)]
mod tests;

/// Symbolic predicate identity is chart-local vertex order, not the shared 3D node identity.
/// This keeps distinct seam images independent while preserving deterministic tie-breaking.
fn predicate_point(vertex: ExactFacePslgVertex, index: u32) -> PlanarPredicatePoint {
    let mut bytes = [0x50; 32];
    bytes[..4].copy_from_slice(&index.to_be_bytes());
    PlanarPredicatePoint {
        identity: StableDigest::from_bytes(bytes),
        coordinates: vertex.uv,
    }
}
