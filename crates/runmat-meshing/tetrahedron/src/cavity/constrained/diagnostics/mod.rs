use super::*;

mod boundary_node_completion;
pub(crate) use boundary_node_completion::diagnostic_boundary_node_completion;
mod interior_star;
pub(crate) use interior_star::diagnostic_interior_star_quality;
mod local_cap_quality;
pub(crate) use local_cap_quality::diagnostic_missing_face_local_cap_quality;
mod missing_face_stitch;
pub(crate) use missing_face_stitch::{
    diagnostic_missing_face_edge_subpatch_cap_stitch,
    diagnostic_missing_face_hybrid_subpatch_cap_stitch, diagnostic_missing_face_local_cap_stitch,
    diagnostic_missing_face_shared_patch_cap_stitch,
};
