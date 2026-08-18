mod encroachment;
mod insertion;
mod select;
mod types;

pub use encroachment::classify_exact_face_refinement_candidate;
pub use insertion::insert_exact_face_refinement_candidate;
pub use select::select_exact_face_refinement_candidate;
pub use types::{
    ExactFaceCandidateDisposition, ExactFaceRefinedTopology, ExactFaceRefinementCandidate,
    ExactFaceRefinementError, ExactFaceRefinementErrorKind, ExactFaceRefinementReason,
    ExactProtectedSegmentSplit,
};

#[cfg(test)]
mod tests;
