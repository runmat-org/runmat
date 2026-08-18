mod encroachment;
mod insertion;
mod refine;
mod select;
mod types;

pub use encroachment::classify_exact_face_refinement_candidate;
pub use insertion::insert_exact_face_refinement_candidate;
pub use refine::refine_exact_face_until_blocked;
pub use select::select_exact_face_refinement_candidate;
pub use types::{
    ExactFaceCandidateDisposition, ExactFaceRefinedMesh, ExactFaceRefinedTopology,
    ExactFaceRefinementCandidate, ExactFaceRefinementContext, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementOptions, ExactFaceRefinementOutcome,
    ExactFaceRefinementPolicy, ExactFaceRefinementReason, ExactProtectedSegmentSplit,
};

#[cfg(test)]
mod tests;
