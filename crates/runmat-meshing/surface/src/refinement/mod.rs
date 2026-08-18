mod encroachment;
mod select;
mod types;

pub use encroachment::classify_exact_face_refinement_candidate;
pub use select::select_exact_face_refinement_candidate;
pub use types::{
    ExactFaceCandidateDisposition, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementReason, ExactProtectedSegmentSplit,
};

#[cfg(test)]
mod tests;
