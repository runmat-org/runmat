use super::*;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryNodeCompletionDiagnostic {
    pub reason: &'static str,
    pub missing_face_count: usize,
    pub cap_candidate_count: usize,
    pub outside_candidate_count: usize,
    pub duplicate_candidate_count: usize,
    pub max_rejected_scaled_jacobian: f64,
    pub rejected_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub max_rejected_cap_height_ratio: f64,
    pub rejected_cap_height_ratio_bins: BTreeMap<String, usize>,
    pub rejected_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_cap_node_ids: BTreeMap<u32, usize>,
    pub split_cap_candidate_count: usize,
    pub split_cap_pass_count: usize,
    pub max_split_cap_scaled_jacobian: f64,
    pub split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub edge_split_cap_candidate_count: usize,
    pub edge_split_cap_pass_count: usize,
    pub max_edge_split_cap_scaled_jacobian: f64,
    pub edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub three_edge_split_cap_candidate_count: usize,
    pub three_edge_split_cap_pass_count: usize,
    pub max_three_edge_split_cap_scaled_jacobian: f64,
    pub three_edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub three_edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub solid_candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub zero_candidate_boundary_faces: Vec<[u32; 3]>,
    pub min_boundary_face_candidate_count: usize,
    pub min_candidate_boundary_faces: Vec<[u32; 3]>,
    pub max_boundary_face_candidate_count: usize,
    pub zero_solid_candidate_boundary_face_count: usize,
    pub zero_solid_candidate_boundary_faces: Vec<[u32; 3]>,
    pub min_solid_boundary_face_candidate_count: usize,
    pub min_solid_candidate_boundary_faces: Vec<[u32; 3]>,
    pub max_solid_boundary_face_candidate_count: usize,
    pub zero_addable_boundary_face_count: usize,
    pub zero_addable_boundary_faces: Vec<[u32; 3]>,
    pub min_addable_boundary_face_candidate_count: usize,
    pub min_addable_candidate_boundary_faces: Vec<[u32; 3]>,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_selected_tetrahedra: Vec<[u32; 4]>,
    pub dead_end_current_volume_m3: f64,
    pub dead_end_candidate_volume_m3: f64,
    pub dead_end_target_volume_m3: f64,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SupportNodeExactCoverDiagnostic {
    pub candidate_node_count: usize,
    pub candidate_count: usize,
    pub root_zero_raw_boundary_face_count: usize,
    pub root_zero_raw_boundary_faces: Vec<[u32; 3]>,
    pub root_min_raw_boundary_face_candidate_count: usize,
    pub root_min_raw_candidate_boundary_faces: Vec<[u32; 3]>,
    pub root_max_raw_boundary_face_candidate_count: usize,
    pub root_zero_addable_boundary_face_count: usize,
    pub root_zero_addable_boundary_faces: Vec<[u32; 3]>,
    pub root_min_addable_boundary_face_candidate_count: usize,
    pub root_min_addable_candidate_boundary_faces: Vec<[u32; 3]>,
    pub root_max_addable_boundary_face_candidate_count: usize,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub dead_end_faces_by_reason: BTreeMap<&'static str, Vec<[u32; 3]>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateDiagnostic {
    pub target_face: [u32; 3],
    pub candidate_count: usize,
    pub addable_count: usize,
    pub candidates: Vec<BoundaryExactCoverMateCandidateDiagnostic>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateCandidateDiagnostic {
    pub node_ids: [u32; 4],
    pub exact_scaled_jacobian: f64,
    pub addable: bool,
    pub conflicting_faces: Vec<[u32; 3]>,
    pub missing_future_mate_faces: Vec<[u32; 3]>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverInteriorMateClosureDiagnostic {
    pub initial_candidate_count: usize,
    pub candidate_count: usize,
    pub injected_candidate_count: usize,
    pub found_cover: bool,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_selected_tetrahedra: Vec<[u32; 4]>,
    pub dead_end_current_volume_m3: f64,
    pub dead_end_candidate_volume_m3: f64,
    pub dead_end_target_volume_m3: f64,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub dead_end_faces_by_reason: BTreeMap<&'static str, Vec<[u32; 3]>>,
    pub dead_end_selected_tetrahedra_by_reason: BTreeMap<&'static str, Vec<[u32; 4]>>,
    pub dead_end_selected_roles_by_reason: BTreeMap<&'static str, Vec<&'static str>>,
    pub unforced_found_cover: bool,
    pub unforced_selected_tetrahedron_count: usize,
    pub unforced_search_attempt_count: usize,
    pub unforced_dead_end_reason_histogram: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverFaceCandidateSourceDiagnostic {
    pub target_face: [u32; 3],
    pub fourth_node_count: usize,
    pub centroid_inside_count: usize,
    pub solid_pass_count: usize,
    pub relaxed_pass_count: usize,
    pub outside_surface_count: usize,
    pub solid_rejected_by_reason: BTreeMap<&'static str, usize>,
    pub relaxed_rejected_by_reason: BTreeMap<&'static str, usize>,
    pub relaxed_candidate_node_ids: Vec<[u32; 4]>,
}
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundarySteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryPatchSteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub missing_face_count: usize,
    pub patch_count: usize,
    pub steiner_node_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapQualityDiagnostic {
    pub missing_face_count: usize,
    pub pass_face_count: usize,
    pub failed_face_count: usize,
    pub candidate_count: usize,
    pub candidate_source_bins: BTreeMap<&'static str, usize>,
    pub max_scaled_jacobian: f64,
    pub max_failed_face_scaled_jacobian: f64,
    pub failed_face_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub failed_face_source_bins: BTreeMap<&'static str, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapStitchDiagnostic {
    pub missing_face_count: usize,
    pub missing_faces: Vec<[u32; 3]>,
    pub patch_count: usize,
    pub patch_size_histogram: BTreeMap<usize, usize>,
    pub patch_capped_face_count_histogram: BTreeMap<usize, usize>,
    pub incomplete_patch_size_histogram: BTreeMap<usize, usize>,
    pub uncapped_faces: Vec<[u32; 3]>,
    pub capped_face_count: usize,
    pub inserted_node_count: usize,
    pub side_connector_candidate_count: usize,
    pub candidate_tetrahedron_count: usize,
    pub cap_side_face_count: usize,
    pub zero_mate_cap_side_face_count: usize,
    pub min_cap_side_face_mate_count: usize,
    pub max_cap_side_face_mate_count: usize,
    pub open_interior_face_count: usize,
    pub open_interior_component_count: usize,
    pub open_interior_component_size_histogram: BTreeMap<usize, usize>,
    pub candidate_with_orphan_interior_face_count: usize,
    pub candidate_without_orphan_interior_face_count: usize,
    pub root_boundary_zero_raw_candidate_face_count: usize,
    pub root_boundary_zero_addable_candidate_face_count: usize,
    pub root_boundary_min_raw_candidate_count: usize,
    pub root_boundary_min_addable_candidate_count: usize,
    pub root_boundary_max_addable_candidate_count: usize,
    pub cover_dead_end_reason: &'static str,
    pub cover_dead_end_depth: usize,
    pub cover_dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub selected_tetrahedron_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryMissingFaceClusterDiagnostic {
    pub missing_face_count: usize,
    pub edge_component_count: usize,
    pub edge_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_count: usize,
    pub node_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_count_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_ids: BTreeMap<u32, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InteriorStarQualityDiagnostic {
    pub candidate_count: usize,
    pub pass_count: usize,
    pub scaled_worst_face_candidate_count: usize,
    pub scaled_worst_face_pass_count: usize,
    pub max_min_scaled_jacobian: f64,
    pub max_scaled_worst_face_min_scaled_jacobian: f64,
    pub min_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub min_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}
