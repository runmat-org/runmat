use super::*;

#[cfg(test)]
mod exact_cover;
#[cfg(test)]
mod local_cap;
#[cfg(test)]
mod shared_cap;

#[cfg(test)]
pub(crate) use local_cap::diagnostic_missing_face_local_cap_stitch;
#[cfg(test)]
use shared_cap::diagnostic_missing_face_shared_cap_stitch_with_link;

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_shared_patch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Node,
        "incomplete_shared_patch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_edge_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_edge_subpatch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_hybrid_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_hybrid_subpatch_caps",
        true,
    )
}
