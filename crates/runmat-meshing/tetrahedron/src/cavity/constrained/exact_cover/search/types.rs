use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(in super::super::super) struct BoundaryExactCoverSolution {
    pub(in super::super::super) selected_indices: Vec<usize>,
    pub(in super::super::super) min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super::super) struct BoundaryExactCoverRootAvailability {
    pub(in super::super::super) zero_raw_candidate_face_count: usize,
    pub(in super::super::super) zero_addable_candidate_face_count: usize,
    pub(in super::super::super) min_raw_candidate_count: usize,
    pub(in super::super::super) min_addable_candidate_count: usize,
    pub(in super::super::super) max_addable_candidate_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(in super::super::super) struct BoundaryExactCoverDeadEnd {
    pub(in super::super::super) reason: &'static str,
    pub(in super::super::super) face: Option<[u32; 3]>,
    pub(in super::super::super) depth: usize,
    pub(in super::super::super) selected_tetrahedra: Vec<[u32; 4]>,
    pub(in super::super::super) selected_roles: Vec<&'static str>,
    pub(in super::super::super) current_volume_m3: f64,
    pub(in super::super::super) candidate_volume_m3: f64,
    pub(in super::super::super) target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(in super::super::super) struct BoundaryExactCoverTrace {
    pub(in super::super::super) dead_end: Option<BoundaryExactCoverDeadEnd>,
    pub(in super::super::super) dead_ends: Vec<BoundaryExactCoverDeadEnd>,
    pub(in super::super::super) dead_end_reason_counts: BTreeMap<&'static str, usize>,
    pub(in super::super::super) dead_end_faces_by_reason:
        BTreeMap<&'static str, BTreeSet<[u32; 3]>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in super::super::super) enum ForcedInteriorMateFailure {
    NoAddableMate {
        face: Option<[u32; 3]>,
        reason: ForcedInteriorMateNoAddableReason,
    },
    VolumeOverflow {
        current_volume_m3: f64,
        candidate_volume_m3: f64,
        target_volume_m3: f64,
    },
}

impl ForcedInteriorMateFailure {
    pub(in super::super::super) fn reason(self) -> &'static str {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { reason, .. } => reason.as_str(),
            ForcedInteriorMateFailure::VolumeOverflow { .. } => {
                "forced_interior_mate_volume_overflow"
            }
        }
    }

    pub(in super::super::super) fn face(self) -> Option<[u32; 3]> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { face, .. } => face,
            ForcedInteriorMateFailure::VolumeOverflow { .. } => None,
        }
    }

    pub(in super::super::super) fn volume(self) -> Option<(f64, f64, f64)> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { .. } => None,
            ForcedInteriorMateFailure::VolumeOverflow {
                current_volume_m3,
                candidate_volume_m3,
                target_volume_m3,
            } => Some((current_volume_m3, candidate_volume_m3, target_volume_m3)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super::super) enum ForcedInteriorMateNoAddableReason {
    NoCandidateContainsFace,
    FaceCountConflict,
    FutureMateConflict,
}

impl ForcedInteriorMateNoAddableReason {
    pub(in super::super::super) fn as_str(self) -> &'static str {
        match self {
            ForcedInteriorMateNoAddableReason::NoCandidateContainsFace => {
                "forced_interior_mate_no_candidate_contains_face"
            }
            ForcedInteriorMateNoAddableReason::FaceCountConflict => {
                "forced_interior_mate_face_count_conflict"
            }
            ForcedInteriorMateNoAddableReason::FutureMateConflict => {
                "forced_interior_mate_future_mate_conflict"
            }
        }
    }
}
