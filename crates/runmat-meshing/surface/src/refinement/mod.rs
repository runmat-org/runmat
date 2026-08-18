mod chart_cut;
mod collar;
mod encroachment;
mod insertion;
mod refine;
mod select;
mod types;

pub use chart_cut::{split_exact_face_chart_cut, validate_exact_face_chart_cut_split_result};
pub use encroachment::classify_exact_face_refinement_candidate;
pub use insertion::insert_exact_face_refinement_candidate;
pub use refine::refine_exact_face_until_blocked;
pub use select::select_exact_face_refinement_candidate;
pub use types::{
    ExactChartCutSplit, ExactChartCutSplitImage, ExactFaceCandidateDisposition,
    ExactFaceFeatureCollar, ExactFaceFeatureCollars, ExactFaceRefinedMesh,
    ExactFaceRefinedTopology, ExactFaceRefinementCandidate, ExactFaceRefinementContext,
    ExactFaceRefinementError, ExactFaceRefinementErrorKind, ExactFaceRefinementOptions,
    ExactFaceRefinementOutcome, ExactFaceRefinementPolicy, ExactFaceRefinementReason,
    ExactProtectedSegmentSplit,
};

#[cfg(test)]
mod tests;
pub use collar::{derive_exact_face_feature_collars, validate_exact_face_feature_collars};
