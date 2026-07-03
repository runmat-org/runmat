use std::collections::BTreeMap;

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
