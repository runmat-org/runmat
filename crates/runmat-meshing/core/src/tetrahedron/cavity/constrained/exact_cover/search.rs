use super::*;

mod traced;

pub(in super::super) struct BoundaryExactCoverSearch<'a> {
    pub(in super::super) candidates: &'a [ConstrainedCavityRefillTetrahedron],
    pub(in super::super) candidate_faces: Vec<[[u32; 3]; 4]>,
    pub(in super::super) boundary_faces: BTreeSet<[u32; 3]>,
    pub(in super::super) target_volume_m3: f64,
    pub(in super::super) volume_tolerance_m3: f64,
    pub(in super::super) max_attempt_count: usize,
    pub(in super::super) attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(in super::super) struct BoundaryExactCoverSolution {
    pub(in super::super) selected_indices: Vec<usize>,
    pub(in super::super) min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct BoundaryExactCoverRootAvailability {
    pub(in super::super) zero_raw_candidate_face_count: usize,
    pub(in super::super) zero_addable_candidate_face_count: usize,
    pub(in super::super) min_raw_candidate_count: usize,
    pub(in super::super) min_addable_candidate_count: usize,
    pub(in super::super) max_addable_candidate_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(in super::super) struct BoundaryExactCoverDeadEnd {
    pub(in super::super) reason: &'static str,
    pub(in super::super) face: Option<[u32; 3]>,
    pub(in super::super) depth: usize,
    pub(in super::super) selected_tetrahedra: Vec<[u32; 4]>,
    pub(in super::super) selected_roles: Vec<&'static str>,
    pub(in super::super) current_volume_m3: f64,
    pub(in super::super) candidate_volume_m3: f64,
    pub(in super::super) target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(in super::super) struct BoundaryExactCoverTrace {
    pub(in super::super) dead_end: Option<BoundaryExactCoverDeadEnd>,
    pub(in super::super) dead_ends: Vec<BoundaryExactCoverDeadEnd>,
    pub(in super::super) dead_end_reason_counts: BTreeMap<&'static str, usize>,
    pub(in super::super) dead_end_faces_by_reason: BTreeMap<&'static str, BTreeSet<[u32; 3]>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in super::super) enum ForcedInteriorMateFailure {
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
    pub(in super::super) fn reason(self) -> &'static str {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { reason, .. } => reason.as_str(),
            ForcedInteriorMateFailure::VolumeOverflow { .. } => {
                "forced_interior_mate_volume_overflow"
            }
        }
    }

    pub(in super::super) fn face(self) -> Option<[u32; 3]> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { face, .. } => face,
            ForcedInteriorMateFailure::VolumeOverflow { .. } => None,
        }
    }

    pub(in super::super) fn volume(self) -> Option<(f64, f64, f64)> {
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
pub(in super::super) enum ForcedInteriorMateNoAddableReason {
    NoCandidateContainsFace,
    FaceCountConflict,
    FutureMateConflict,
}

impl ForcedInteriorMateNoAddableReason {
    pub(in super::super) fn as_str(self) -> &'static str {
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

impl<'a> BoundaryExactCoverSearch<'a> {
    pub(in super::super) fn new(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTetrahedron],
        volume_relative_tolerance: f64,
    ) -> Self {
        Self::with_attempt_limit(cavity, candidates, volume_relative_tolerance, 5_000)
    }

    pub(in super::super) fn with_attempt_limit(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTetrahedron],
        volume_relative_tolerance: f64,
        max_attempt_count: usize,
    ) -> Self {
        Self {
            candidates,
            candidate_faces: candidates
                .iter()
                .map(|candidate| tetrahedron_faces(candidate.node_ids).map(sorted_face))
                .collect(),
            boundary_faces: cavity
                .boundary_faces
                .iter()
                .map(|face| sorted_face(face.node_ids))
                .collect(),
            target_volume_m3: cavity.target_volume_m3,
            volume_tolerance_m3: cavity.target_volume_m3.max(1.0e-18) * volume_relative_tolerance,
            max_attempt_count,
            attempts: 0,
        }
    }

    pub(in super::super) fn search_best(&mut self) -> Option<Vec<usize>> {
        let mut best = None::<BoundaryExactCoverSolution>;
        self.search_best_from(
            0.0,
            f64::INFINITY,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut best,
        );
        best.map(|solution| solution.selected_indices)
    }

    #[cfg(test)]
    pub(in super::super) fn search(&mut self) -> Option<Vec<usize>> {
        self.search_from(0.0, &mut BTreeMap::new(), &mut Vec::new())
    }

    #[cfg(test)]
    pub(in super::super) fn search_from(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count
            || current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3
        {
            return None;
        }
        let Some((forced_volume_m3, forced_indices)) =
            self.propagate_forced_interior_mates(current_volume_m3, face_counts, selected)
        else {
            return None;
        };
        let current_volume_m3 = current_volume_m3 + forced_volume_m3;
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                return Some(selected.clone());
            }
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return None;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            if let Some(result) = self.search_from(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
            ) {
                return Some(result);
            }
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.rollback_selected_candidates(&forced_indices, face_counts, selected);
        None
    }

    pub(in super::super) fn search_best_from(
        &mut self,
        current_volume_m3: f64,
        current_min_scaled_jacobian: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        best: &mut Option<BoundaryExactCoverSolution>,
    ) {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count
            || current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3
        {
            return;
        }
        if best
            .as_ref()
            .is_some_and(|solution| current_min_scaled_jacobian <= solution.min_scaled_jacobian)
        {
            return;
        }
        let Some((forced_volume_m3, forced_indices)) =
            self.propagate_forced_interior_mates(current_volume_m3, face_counts, selected)
        else {
            return;
        };
        let current_volume_m3 = current_volume_m3 + forced_volume_m3;
        let current_min_scaled_jacobian = forced_indices
            .iter()
            .map(|index| self.candidates[*index].exact_scaled_jacobian)
            .fold(current_min_scaled_jacobian, f64::min);
        if best
            .as_ref()
            .is_some_and(|solution| current_min_scaled_jacobian <= solution.min_scaled_jacobian)
        {
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return;
        }
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                *best = Some(BoundaryExactCoverSolution {
                    selected_indices: selected.clone(),
                    min_scaled_jacobian: current_min_scaled_jacobian,
                });
            }
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            self.search_best_from(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                current_min_scaled_jacobian
                    .min(self.candidates[candidate_index].exact_scaled_jacobian),
                face_counts,
                selected,
                best,
            );
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.rollback_selected_candidates(&forced_indices, face_counts, selected);
    }
}
