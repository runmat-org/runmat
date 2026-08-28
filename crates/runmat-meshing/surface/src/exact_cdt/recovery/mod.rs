mod cavity;
mod ear;
mod planarity;
mod recover;
mod types;
mod validate;

pub use recover::recover_exact_face_segments;
pub(crate) use recover::recover_validated_face_segments;
pub use types::{ExactFaceConstrainedDelaunay, ExactFaceRecoveredSegment};
pub use validate::validate_exact_face_constrained_delaunay;
pub(crate) use validate::validate_face_constrained_topology;

#[cfg(test)]
mod tests;
