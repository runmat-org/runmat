mod cavity;
mod ear;
mod planarity;
mod recover;
mod topology;
mod types;
mod validate;

pub use recover::recover_exact_face_segments;
pub use types::{ExactFaceConstrainedDelaunay, ExactFaceRecoveredSegment};
pub use validate::validate_exact_face_constrained_delaunay;

#[cfg(test)]
mod tests;
