mod domain;
mod pslg;
mod recovery;
mod topology;
mod triangulation;

pub use domain::{
    carve_exact_face_domain, validate_exact_face_trimmed_delaunay, ExactFaceTrimmedDelaunay,
};
pub(crate) use domain::{carve_validated_face_domain, validate_face_trimmed_topology};

pub(crate) use pslg::MAX_FACE_PSLG_ITEMS;
pub(crate) use pslg::{build_canonical_pslg, PslgLoopInput, PslgSegmentInput};
pub use pslg::{
    build_exact_face_pslg, exact_face_interior_node_id, validate_exact_face_pslg, ExactFacePslg,
    ExactFacePslgError, ExactFacePslgErrorKind, ExactFacePslgLoop, ExactFacePslgLoopSource,
    ExactFacePslgSegment, ExactFacePslgSegmentSource, ExactFacePslgVertex,
};
pub use recovery::{
    recover_exact_face_segments, validate_exact_face_constrained_delaunay,
    ExactFaceConstrainedDelaunay, ExactFaceRecoveredSegment,
};
pub(crate) use recovery::{recover_validated_face_segments, validate_face_constrained_topology};
pub use triangulation::{
    triangulate_exact_face_pslg, validate_exact_face_delaunay, ExactFaceDelaunay,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions,
    ExactFaceDelaunayTriangle,
};
pub(crate) use triangulation::{triangulate_validated_face_pslg, validate_face_delaunay_topology};
