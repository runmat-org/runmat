use std::collections::BTreeMap;

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
