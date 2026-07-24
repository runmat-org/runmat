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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateDiagnostic {
    pub target_face: [u32; 3],
    pub candidate_count: usize,
    pub addable_count: usize,
    pub candidates: Vec<BoundaryExactCoverMateCandidateDiagnostic>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateCandidateDiagnostic {
    pub node_ids: [u32; 4],
    pub exact_scaled_jacobian: f64,
    pub addable: bool,
    pub conflicting_faces: Vec<[u32; 3]>,
    pub missing_future_mate_faces: Vec<[u32; 3]>,
}

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
