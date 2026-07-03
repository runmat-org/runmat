use super::*;

pub(super) struct BoundaryExactCoverSearch<'a> {
    pub(super) candidates: &'a [ConstrainedCavityRefillTetrahedron],
    pub(super) candidate_faces: Vec<[[u32; 3]; 4]>,
    pub(super) boundary_faces: BTreeSet<[u32; 3]>,
    pub(super) target_volume_m3: f64,
    pub(super) volume_tolerance_m3: f64,
    pub(super) max_attempt_count: usize,
    pub(super) attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundaryExactCoverSolution {
    pub(super) selected_indices: Vec<usize>,
    pub(super) min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BoundaryExactCoverRootAvailability {
    pub(super) zero_raw_candidate_face_count: usize,
    pub(super) zero_addable_candidate_face_count: usize,
    pub(super) min_raw_candidate_count: usize,
    pub(super) min_addable_candidate_count: usize,
    pub(super) max_addable_candidate_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundaryExactCoverDeadEnd {
    pub(super) reason: &'static str,
    pub(super) face: Option<[u32; 3]>,
    pub(super) depth: usize,
    pub(super) selected_tetrahedra: Vec<[u32; 4]>,
    pub(super) selected_roles: Vec<&'static str>,
    pub(super) current_volume_m3: f64,
    pub(super) candidate_volume_m3: f64,
    pub(super) target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundaryExactCoverTrace {
    pub(super) dead_end: Option<BoundaryExactCoverDeadEnd>,
    pub(super) dead_ends: Vec<BoundaryExactCoverDeadEnd>,
    pub(super) dead_end_reason_counts: BTreeMap<&'static str, usize>,
    pub(super) dead_end_faces_by_reason: BTreeMap<&'static str, BTreeSet<[u32; 3]>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ForcedInteriorMateFailure {
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
    pub(super) fn reason(self) -> &'static str {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { reason, .. } => reason.as_str(),
            ForcedInteriorMateFailure::VolumeOverflow { .. } => {
                "forced_interior_mate_volume_overflow"
            }
        }
    }

    pub(super) fn face(self) -> Option<[u32; 3]> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { face, .. } => face,
            ForcedInteriorMateFailure::VolumeOverflow { .. } => None,
        }
    }

    pub(super) fn volume(self) -> Option<(f64, f64, f64)> {
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
pub(super) enum ForcedInteriorMateNoAddableReason {
    NoCandidateContainsFace,
    FaceCountConflict,
    FutureMateConflict,
}

impl ForcedInteriorMateNoAddableReason {
    pub(super) fn as_str(self) -> &'static str {
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
    pub(super) fn new(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTetrahedron],
        volume_relative_tolerance: f64,
    ) -> Self {
        Self::with_attempt_limit(cavity, candidates, volume_relative_tolerance, 5_000)
    }

    pub(super) fn with_attempt_limit(
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

    pub(super) fn search_best(&mut self) -> Option<Vec<usize>> {
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

    pub(super) fn search_with_trace(&mut self) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
        let mut trace = BoundaryExactCoverTrace {
            dead_end: None,
            dead_ends: Vec::new(),
            dead_end_reason_counts: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        };
        let result = self.search_from_traced(
            0.0,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut trace,
        );
        (result, trace)
    }

    #[cfg(test)]
    pub(super) fn search_without_forced_with_trace(
        &mut self,
    ) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
        let mut trace = BoundaryExactCoverTrace {
            dead_end: None,
            dead_ends: Vec::new(),
            dead_end_reason_counts: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        };
        let result = self.search_from_without_forced_traced(
            0.0,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut trace,
        );
        (result, trace)
    }

    pub(super) fn record_dead_end(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
    ) {
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, None, None);
    }

    pub(super) fn record_dead_end_for_face(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
        face: Option<[u32; 3]>,
        volume: Option<(f64, f64, f64)>,
    ) {
        *trace.dead_end_reason_counts.entry(reason).or_default() += 1;
        if let Some(face) = face {
            trace
                .dead_end_faces_by_reason
                .entry(reason)
                .or_default()
                .insert(face);
        }
        let (current_volume_m3, candidate_volume_m3, target_volume_m3) =
            volume.unwrap_or((0.0, 0.0, self.target_volume_m3));
        let dead_end = BoundaryExactCoverDeadEnd {
            reason,
            face,
            depth: selected.len(),
            selected_tetrahedra: selected
                .iter()
                .map(|candidate_index| self.candidates[*candidate_index].node_ids)
                .collect(),
            selected_roles: selected_roles.to_vec(),
            current_volume_m3,
            candidate_volume_m3,
            target_volume_m3,
        };
        if trace.dead_end.is_none() {
            trace.dead_end = Some(dead_end.clone());
        }
        if trace.dead_ends.len() < 128 {
            trace.dead_ends.push(dead_end);
        }
    }

    #[cfg(test)]
    pub(super) fn root_boundary_availability(&self) -> BoundaryExactCoverRootAvailability {
        let face_counts = BTreeMap::<[u32; 3], usize>::new();
        let selected = Vec::<usize>::new();
        let mut zero_raw = 0_usize;
        let mut zero_addable = 0_usize;
        let mut min_raw = usize::MAX;
        let mut min_addable = usize::MAX;
        let mut max_addable = 0_usize;
        for face in &self.boundary_faces {
            let raw_count = self
                .candidate_faces
                .iter()
                .filter(|candidate_faces| candidate_faces.contains(face))
                .count();
            let addable_count = (0..self.candidates.len())
                .filter(|candidate_index| {
                    self.candidate_can_be_added_for_face(
                        *candidate_index,
                        *face,
                        &face_counts,
                        &selected,
                    )
                })
                .count();
            zero_raw += usize::from(raw_count == 0);
            zero_addable += usize::from(addable_count == 0);
            min_raw = min_raw.min(raw_count);
            min_addable = min_addable.min(addable_count);
            max_addable = max_addable.max(addable_count);
        }
        if self.boundary_faces.is_empty() {
            min_raw = 0;
            min_addable = 0;
        }
        BoundaryExactCoverRootAvailability {
            zero_raw_candidate_face_count: zero_raw,
            zero_addable_candidate_face_count: zero_addable,
            min_raw_candidate_count: min_raw,
            min_addable_candidate_count: min_addable,
            max_addable_candidate_count: max_addable,
        }
    }

    pub(super) fn search_from_traced(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
        trace: &mut BoundaryExactCoverTrace,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count {
            self.record_dead_end(trace, selected, selected_roles, "attempt_limit");
            return None;
        }
        if current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
            self.record_dead_end(trace, selected, selected_roles, "volume_overflow");
            return None;
        }
        let (forced_volume_m3, forced_indices) = match self.propagate_forced_interior_mates_traced(
            current_volume_m3,
            face_counts,
            selected,
            selected_roles,
        ) {
            Ok(forced) => forced,
            Err(reason) => {
                self.record_dead_end_for_face(
                    trace,
                    selected,
                    selected_roles,
                    reason.reason(),
                    reason.face(),
                    reason.volume(),
                );
                return None;
            }
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
            let reason = if !boundary_ok {
                "boundary_incomplete"
            } else if !interior_ok {
                "interior_incomplete"
            } else {
                "volume_mismatch"
            };
            self.record_dead_end(trace, selected, selected_roles, reason);
            self.rollback_selected_candidates_with_roles(
                &forced_indices,
                face_counts,
                selected,
                selected_roles,
            );
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
            selected_roles.push("branch");
            if let Some(result) = self.search_from_traced(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
                selected_roles,
                trace,
            ) {
                return Some(result);
            }
            selected_roles.pop();
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        let (reason, face) = self.candidates_exhausted_reason_and_face(face_counts, selected);
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, face, None);
        self.rollback_selected_candidates_with_roles(
            &forced_indices,
            face_counts,
            selected,
            selected_roles,
        );
        None
    }

    #[cfg(test)]
    pub(super) fn search_from_without_forced_traced(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
        trace: &mut BoundaryExactCoverTrace,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count {
            self.record_dead_end(trace, selected, selected_roles, "attempt_limit");
            return None;
        }
        if current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
            self.record_dead_end(trace, selected, selected_roles, "volume_overflow");
            return None;
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
                return Some(selected.clone());
            }
            let reason = if !boundary_ok {
                "boundary_incomplete"
            } else if !interior_ok {
                "interior_incomplete"
            } else {
                "volume_mismatch"
            };
            self.record_dead_end(trace, selected, selected_roles, reason);
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
            selected_roles.push("branch");
            if let Some(result) = self.search_from_without_forced_traced(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
                selected_roles,
                trace,
            ) {
                return Some(result);
            }
            selected_roles.pop();
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.record_dead_end(
            trace,
            selected,
            selected_roles,
            self.candidates_exhausted_reason(face_counts, selected),
        );
        None
    }

    #[cfg(test)]
    pub(super) fn search(&mut self) -> Option<Vec<usize>> {
        self.search_from(0.0, &mut BTreeMap::new(), &mut Vec::new())
    }

    #[cfg(test)]
    pub(super) fn search_from(
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

    pub(super) fn search_best_from(
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

    pub(super) fn propagate_forced_interior_mates(
        &self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) -> Option<(f64, Vec<usize>)> {
        let mut forced_indices = Vec::<usize>::new();
        let mut forced_volume_m3 = 0.0;
        loop {
            let forced_candidate = self.forced_interior_mate(face_counts, selected)?;
            let Some(candidate_index) = forced_candidate else {
                return Some((forced_volume_m3, forced_indices));
            };
            let next_volume_m3 =
                current_volume_m3 + forced_volume_m3 + self.candidates[candidate_index].volume_m3;
            if next_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
                self.rollback_selected_candidates(&forced_indices, face_counts, selected);
                return None;
            }
            self.add_candidate_faces(candidate_index, face_counts);
            selected.push(candidate_index);
            forced_indices.push(candidate_index);
            forced_volume_m3 += self.candidates[candidate_index].volume_m3;
        }
    }

    pub(super) fn propagate_forced_interior_mates_traced(
        &self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
    ) -> Result<(f64, Vec<usize>), ForcedInteriorMateFailure> {
        let mut forced_indices = Vec::<usize>::new();
        let mut forced_volume_m3 = 0.0;
        loop {
            let forced_candidate = self
                .forced_interior_mate_traced(face_counts, selected)
                .ok_or_else(|| {
                    let (face, reason) =
                        self.forced_interior_mate_no_addable_reason(face_counts, selected);
                    ForcedInteriorMateFailure::NoAddableMate { face, reason }
                })?;
            let Some(candidate_index) = forced_candidate else {
                return Ok((forced_volume_m3, forced_indices));
            };
            let next_volume_m3 =
                current_volume_m3 + forced_volume_m3 + self.candidates[candidate_index].volume_m3;
            if next_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
                self.rollback_selected_candidates_with_roles(
                    &forced_indices,
                    face_counts,
                    selected,
                    selected_roles,
                );
                return Err(ForcedInteriorMateFailure::VolumeOverflow {
                    current_volume_m3: current_volume_m3 + forced_volume_m3,
                    candidate_volume_m3: self.candidates[candidate_index].volume_m3,
                    target_volume_m3: self.target_volume_m3,
                });
            }
            self.add_candidate_faces(candidate_index, face_counts);
            selected.push(candidate_index);
            selected_roles.push("forced");
            forced_indices.push(candidate_index);
            forced_volume_m3 += self.candidates[candidate_index].volume_m3;
        }
    }

    pub(super) fn forced_interior_mate(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        self.forced_interior_mate_traced(face_counts, selected)
    }

    pub(super) fn forced_interior_mate_traced(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        let mut forced = None::<usize>;
        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            match candidates.as_slice() {
                [] => return None,
                [candidate] => {
                    if forced.is_none() {
                        forced = Some(*candidate);
                    }
                }
                _ => {}
            }
        }
        Some(forced)
    }

    pub(super) fn forced_interior_mate_no_addable_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> (Option<[u32; 3]>, ForcedInteriorMateNoAddableReason) {
        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let mate_indices = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_faces[*candidate_index].contains(&face)
                })
                .collect::<Vec<_>>();
            if mate_indices.is_empty() {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::NoCandidateContainsFace,
                );
            }
            if mate_indices.iter().all(|candidate_index| {
                self.candidate_faces[*candidate_index]
                    .iter()
                    .any(|candidate_face| {
                        let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                        if self.boundary_faces.contains(candidate_face) {
                            count != 0
                        } else {
                            count >= 2
                        }
                    })
            }) {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::FaceCountConflict,
                );
            }
            if mate_indices.iter().all(|candidate_index| {
                !self.candidate_faces[*candidate_index]
                    .iter()
                    .all(|candidate_face| {
                        let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                        self.boundary_faces.contains(candidate_face)
                            || count == 1
                            || self.interior_face_has_future_mate(
                                *candidate_index,
                                *candidate_face,
                                face_counts,
                                selected,
                            )
                    })
            }) {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::FutureMateConflict,
                );
            }
        }
        (
            None,
            ForcedInteriorMateNoAddableReason::NoCandidateContainsFace,
        )
    }

    pub(super) fn rollback_selected_candidates(
        &self,
        indices: &[usize],
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) {
        for candidate_index in indices.iter().rev() {
            let Some(position) = selected
                .iter()
                .rposition(|selected_index| selected_index == candidate_index)
            else {
                continue;
            };
            selected.remove(position);
            self.remove_candidate_faces(*candidate_index, face_counts);
        }
    }

    pub(super) fn rollback_selected_candidates_with_roles(
        &self,
        indices: &[usize],
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
    ) {
        for candidate_index in indices.iter().rev() {
            let Some(position) = selected
                .iter()
                .rposition(|selected_index| selected_index == candidate_index)
            else {
                continue;
            };
            selected.remove(position);
            if position < selected_roles.len() {
                selected_roles.remove(position);
            }
            self.remove_candidate_faces(*candidate_index, face_counts);
        }
    }

    pub(super) fn add_candidate_faces(
        &self,
        candidate_index: usize,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
    ) {
        for face in self.candidate_faces[candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    pub(super) fn remove_candidate_faces(
        &self,
        candidate_index: usize,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
    ) {
        for face in self.candidate_faces[candidate_index] {
            if let Some(count) = face_counts.get_mut(&face) {
                *count -= 1;
                if *count == 0 {
                    face_counts.remove(&face);
                }
            }
        }
    }

    pub(super) fn next_cover_candidates(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Vec<usize>> {
        let mut best = None::<Vec<usize>>;
        for face in self
            .boundary_faces
            .iter()
            .filter(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            *face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        if best.is_some() {
            return best;
        }

        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        best
    }

    #[cfg(test)]
    pub(super) fn candidates_exhausted_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> &'static str {
        self.candidates_exhausted_reason_and_face(face_counts, selected)
            .0
    }

    pub(super) fn candidates_exhausted_reason_and_face(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> (&'static str, Option<[u32; 3]>) {
        for face in self
            .boundary_faces
            .iter()
            .filter(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        {
            let raw_count = self.raw_candidate_count_for_face(*face, selected);
            if raw_count == 0 {
                return ("boundary_face_no_raw_candidate", Some(*face));
            }
            let addable_count = self.addable_candidate_count_for_face(*face, face_counts, selected);
            if addable_count == 0 {
                return ("boundary_face_no_addable_candidate", Some(*face));
            }
            return ("boundary_face_candidates_exhausted", Some(*face));
        }

        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let raw_count = self.raw_candidate_count_for_face(face, selected);
            if raw_count == 0 {
                return ("interior_face_no_raw_candidate", Some(face));
            }
            let addable_count = self.addable_candidate_count_for_face(face, face_counts, selected);
            if addable_count == 0 {
                return ("interior_face_no_addable_candidate", Some(face));
            }
            return ("interior_face_candidates_exhausted", Some(face));
        }

        ("candidates_exhausted", None)
    }

    pub(super) fn raw_candidate_count_for_face(&self, face: [u32; 3], selected: &[usize]) -> usize {
        (0..self.candidates.len())
            .filter(|candidate_index| {
                !selected.contains(candidate_index)
                    && self.candidate_faces[*candidate_index].contains(&face)
            })
            .count()
    }

    pub(super) fn addable_candidate_count_for_face(
        &self,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> usize {
        (0..self.candidates.len())
            .filter(|candidate_index| {
                !selected.contains(candidate_index)
                    && self.candidate_can_be_added_for_face(
                        *candidate_index,
                        face,
                        face_counts,
                        selected,
                    )
            })
            .count()
    }

    pub(super) fn candidate_can_be_added_for_face(
        &self,
        candidate_index: usize,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> bool {
        self.candidate_faces[candidate_index].contains(&face)
            && self.candidate_faces[candidate_index]
                .iter()
                .all(|candidate_face| {
                    let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                    if self.boundary_faces.contains(candidate_face) {
                        count == 0
                    } else {
                        count < 2
                    }
                })
            && self.candidate_faces[candidate_index]
                .iter()
                .all(|candidate_face| {
                    let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                    self.boundary_faces.contains(candidate_face)
                        || count == 1
                        || self.interior_face_has_future_mate(
                            candidate_index,
                            *candidate_face,
                            face_counts,
                            selected,
                        )
                })
    }

    pub(super) fn interior_face_has_future_mate(
        &self,
        candidate_index: usize,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> bool {
        (0..self.candidates.len()).any(|mate_index| {
            !selected.contains(&mate_index)
                && mate_index != candidate_index
                && self.candidate_faces[mate_index].contains(&face)
                && self.candidate_faces[mate_index].iter().all(|mate_face| {
                    let count = face_counts.get(mate_face).copied().unwrap_or(0)
                        + usize::from(self.candidate_faces[candidate_index].contains(mate_face));
                    if self.boundary_faces.contains(mate_face) {
                        count == 0
                    } else {
                        count < 2
                    }
                })
        })
    }

    pub(super) fn sort_cover_candidates(&self, candidates: &mut [usize]) {
        candidates.sort_by(|left, right| {
            self.candidates[*right]
                .exact_scaled_jacobian
                .total_cmp(&self.candidates[*left].exact_scaled_jacobian)
        });
    }
}

pub(super) fn boundary_node_exact_cover_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() < 4
        || node_ids.len() > MAX_BOUNDARY_EXACT_COVER_NODES
        || cavity.boundary_faces.len() > MAX_BOUNDARY_EXACT_COVER_FACES
    {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let touches_boundary = tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        if touches_boundary
                            && candidate_keys.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids))
                        {
                            candidates.push(tetrahedron.clone());
                        }
                        all_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() || candidates.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
        return Ok(None);
    }
    if let Some(refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &candidates, options)?
    {
        return Ok(Some(refill));
    }
    exact_cover_refill_from_on_demand_interior_mates(cavity, candidates, all_candidates, options)
}

pub(super) fn exact_cover_refill_from_candidate_tetrahedra(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let Some(selected_indices) = search.search_best() else {
        return Ok(None);
    };
    let selected_tetrahedra = selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    refill_from_tetrahedra(
        cavity,
        selected_tetrahedra,
        options.volume_relative_tolerance,
    )
    .map(Some)
}

pub(super) fn exact_cover_refill_from_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    mut candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    all_candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    if candidates.is_empty() || candidates.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidate_keys = candidates
        .iter()
        .map(|candidate| sorted_tetrahedron_nodes(candidate.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }

    for _ in 0..64 {
        let (selected, trace) = {
            let mut search = BoundaryExactCoverSearch::new(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            search.search_with_trace()
        };
        if let Some(selected) = selected {
            let selected_tetrahedra = selected
                .into_iter()
                .map(|index| candidates[index].clone())
                .collect::<Vec<_>>();
            return refill_from_tetrahedra(
                cavity,
                selected_tetrahedra,
                options.volume_relative_tolerance,
            )
            .map(Some);
        }

        let future_mate_dead_ends = trace
            .dead_ends
            .iter()
            .filter(|dead_end| dead_end.reason == "forced_interior_mate_future_mate_conflict")
            .cloned()
            .collect::<Vec<_>>();
        let no_candidate_dead_end_faces = trace
            .dead_ends
            .iter()
            .filter_map(|dead_end| {
                (dead_end.reason == "forced_interior_mate_no_candidate_contains_face")
                    .then_some(dead_end.face)
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        let open_interior_dead_end_faces = trace
            .dead_ends
            .iter()
            .filter(|dead_end| {
                matches!(
                    dead_end.reason,
                    "interior_face_no_raw_candidate"
                        | "interior_face_no_addable_candidate"
                        | "interior_face_candidates_exhausted"
                        | "interior_incomplete"
                )
            })
            .flat_map(|dead_end| {
                open_interior_faces_from_tetrahedron_node_ids(&dead_end.selected_tetrahedra)
            })
            .filter(|face| !boundary_faces.contains(face))
            .collect::<BTreeSet<_>>();
        let root_blocked_boundary_mate_faces = trace
            .dead_ends
            .iter()
            .any(|dead_end| dead_end.reason == "boundary_face_no_addable_candidate")
            .then(|| {
                root_boundary_future_mate_faces(
                    cavity,
                    &candidates,
                    options.volume_relative_tolerance,
                )
            })
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
        if future_mate_dead_ends.is_empty()
            && no_candidate_dead_end_faces.is_empty()
            && open_interior_dead_end_faces.is_empty()
            && root_blocked_boundary_mate_faces.is_empty()
        {
            return Ok(None);
        }

        let search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let mut mate_faces = BTreeSet::<[u32; 3]>::new();
        mate_faces.extend(no_candidate_dead_end_faces);
        mate_faces.extend(open_interior_dead_end_faces);
        mate_faces.extend(root_blocked_boundary_mate_faces);
        for dead_end in &future_mate_dead_ends {
            let Some(face) = dead_end.face else {
                continue;
            };
            let selected_indices = dead_end
                .selected_tetrahedra
                .iter()
                .filter_map(|selected_tetrahedron| {
                    candidates.iter().position(|candidate| {
                        sorted_tetrahedron_nodes(candidate.node_ids)
                            == sorted_tetrahedron_nodes(*selected_tetrahedron)
                    })
                })
                .collect::<Vec<_>>();
            let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
            for selected_index in &selected_indices {
                for selected_face in
                    tetrahedron_faces(candidates[*selected_index].node_ids).map(sorted_face)
                {
                    *face_counts.entry(selected_face).or_default() += 1;
                }
            }
            for candidate_index in (0..candidates.len()).filter(|candidate_index| {
                !selected_indices.contains(candidate_index)
                    && search.candidate_faces[*candidate_index].contains(&face)
            }) {
                for candidate_face in search.candidate_faces[candidate_index] {
                    if !boundary_faces.contains(&candidate_face)
                        && face_counts.get(&candidate_face).copied().unwrap_or(0) == 0
                        && !search.interior_face_has_future_mate(
                            candidate_index,
                            candidate_face,
                            &face_counts,
                            &selected_indices,
                        )
                    {
                        mate_faces.insert(candidate_face);
                    }
                }
            }
        }

        let mut added = false;
        for mate_face in mate_faces {
            let Some(indices) = all_candidates_by_face.get(&mate_face) else {
                continue;
            };
            let mut indices = indices.clone();
            indices.sort_by(|left, right| {
                all_candidates[*right]
                    .exact_scaled_jacobian
                    .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
            });
            for index in indices {
                if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                    break;
                }
                let candidate = &all_candidates[index];
                if candidate_keys.insert(sorted_tetrahedron_nodes(candidate.node_ids)) {
                    candidates.push(candidate.clone());
                    added = true;
                    break;
                }
            }
        }
        if !added {
            return Ok(None);
        }
    }

    Ok(None)
}

pub(super) fn open_interior_faces_from_tetrahedron_node_ids(
    tetrahedra: &[[u32; 4]],
) -> Vec<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(*tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

pub(super) fn root_boundary_future_mate_faces(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    volume_relative_tolerance: f64,
) -> Vec<[u32; 3]> {
    let search = BoundaryExactCoverSearch::new(cavity, candidates, volume_relative_tolerance);
    let face_counts = BTreeMap::<[u32; 3], usize>::new();
    let selected = Vec::<usize>::new();
    let mut mate_faces = BTreeSet::<[u32; 3]>::new();
    for boundary_face in &search.boundary_faces {
        for candidate_index in 0..candidates.len() {
            if !search.candidate_faces[candidate_index].contains(boundary_face) {
                continue;
            }
            for candidate_face in search.candidate_faces[candidate_index] {
                if search.boundary_faces.contains(&candidate_face) {
                    continue;
                }
                if !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                ) {
                    mate_faces.insert(candidate_face);
                }
            }
        }
    }
    mate_faces.into_iter().collect()
}

#[cfg(test)]
pub(super) fn exact_cover_trace_faces_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 3]>> {
    trace
        .dead_end_faces_by_reason
        .iter()
        .map(|(reason, faces)| (*reason, faces.iter().copied().collect::<Vec<_>>()))
        .collect()
}

#[cfg(test)]
pub(super) fn exact_cover_trace_selected_tetrahedra_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 4]>> {
    let mut selected_tetrahedra_by_reason = BTreeMap::<&'static str, Vec<[u32; 4]>>::new();
    for dead_end in &trace.dead_ends {
        selected_tetrahedra_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_tetrahedra.clone());
    }
    selected_tetrahedra_by_reason
}

#[cfg(test)]
pub(super) fn exact_cover_trace_selected_roles_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut selected_roles_by_reason = BTreeMap::<&'static str, Vec<&'static str>>::new();
    for dead_end in &trace.dead_ends {
        selected_roles_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_roles.clone());
    }
    selected_roles_by_reason
}

#[cfg(test)]
pub(super) fn diagnostic_unforced_exact_cover_for_candidates(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    volume_relative_tolerance: f64,
) -> (bool, usize, usize, BTreeMap<&'static str, usize>) {
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        candidates,
        volume_relative_tolerance,
        250,
    );
    let (selected, trace) = search.search_without_forced_with_trace();
    (
        selected.is_some(),
        selected.map(|selected| selected.len()).unwrap_or(0),
        search.attempts,
        trace.dead_end_reason_counts,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundaryExactCoverDiagnostic {
        boundary_node_count: node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        solid_candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        zero_candidate_boundary_faces: Vec::new(),
        min_boundary_face_candidate_count: 0,
        min_candidate_boundary_faces: Vec::new(),
        max_boundary_face_candidate_count: 0,
        zero_solid_candidate_boundary_face_count: 0,
        zero_solid_candidate_boundary_faces: Vec::new(),
        min_solid_boundary_face_candidate_count: 0,
        min_solid_candidate_boundary_faces: Vec::new(),
        max_solid_boundary_face_candidate_count: 0,
        zero_addable_boundary_face_count: 0,
        zero_addable_boundary_faces: Vec::new(),
        min_addable_boundary_face_candidate_count: 0,
        min_addable_candidate_boundary_faces: Vec::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
    };
    if node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut solid_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        solid_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    diagnostic.solid_candidate_count = solid_candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    let solid_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            solid_candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.min_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    diagnostic.zero_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_solid_candidate_boundary_face_count = solid_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_solid_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    diagnostic.zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_addable_boundary_face_count = addable_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_addable_boundary_face_candidate_count = addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.dead_end_reason = dead_end.reason;
        diagnostic.dead_end_face = dead_end.face;
        diagnostic.dead_end_depth = dead_end.depth;
        diagnostic.dead_end_selected_tetrahedra = dead_end.selected_tetrahedra;
        diagnostic.dead_end_current_volume_m3 = dead_end.current_volume_m3;
        diagnostic.dead_end_candidate_volume_m3 = dead_end.candidate_volume_m3;
        diagnostic.dead_end_target_volume_m3 = dead_end.target_volume_m3;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }

    let search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_keys
                .contains(&sorted_tetrahedron_nodes(candidate.node_ids))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for candidate_index in &selected {
        for face in search.candidate_faces[*candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let target_face = sorted_face(target_face);
    let mut diagnostics = Vec::<BoundaryExactCoverMateCandidateDiagnostic>::new();
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !search.candidate_faces[candidate_index].contains(&target_face)
        {
            continue;
        }
        let mut conflicting_faces = Vec::<[u32; 3]>::new();
        let mut missing_future_mate_faces = Vec::<[u32; 3]>::new();
        for candidate_face in search.candidate_faces[candidate_index] {
            let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
            if if boundary_faces.contains(&candidate_face) {
                count != 0
            } else {
                count >= 2
            } {
                conflicting_faces.push(candidate_face);
            }
            if !boundary_faces.contains(&candidate_face)
                && count == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                )
            {
                missing_future_mate_faces.push(candidate_face);
            }
        }
        let addable = search.candidate_can_be_added_for_face(
            candidate_index,
            target_face,
            &face_counts,
            &selected,
        );
        diagnostics.push(BoundaryExactCoverMateCandidateDiagnostic {
            node_ids: candidates[candidate_index].node_ids,
            exact_scaled_jacobian: candidates[candidate_index].exact_scaled_jacobian,
            addable,
            conflicting_faces,
            missing_future_mate_faces,
        });
    }
    diagnostics.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverMateDiagnostic {
        target_face,
        candidate_count: diagnostics.len(),
        addable_count: diagnostics
            .iter()
            .filter(|candidate| candidate.addable)
            .count(),
        candidates: diagnostics,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    let search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_keys
                .contains(&sorted_tetrahedron_nodes(candidate.node_ids))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for candidate_index in &selected {
        for face in search.candidate_faces[*candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let target_face = sorted_face(target_face);
    let mut diagnostics = Vec::<BoundaryExactCoverMateCandidateDiagnostic>::new();
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !search.candidate_faces[candidate_index].contains(&target_face)
        {
            continue;
        }
        let mut conflicting_faces = Vec::<[u32; 3]>::new();
        let mut missing_future_mate_faces = Vec::<[u32; 3]>::new();
        for candidate_face in search.candidate_faces[candidate_index] {
            let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
            if if boundary_faces.contains(&candidate_face) {
                count != 0
            } else {
                count >= 2
            } {
                conflicting_faces.push(candidate_face);
            }
            if !boundary_faces.contains(&candidate_face)
                && count == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                )
            {
                missing_future_mate_faces.push(candidate_face);
            }
        }
        let addable = search.candidate_can_be_added_for_face(
            candidate_index,
            target_face,
            &face_counts,
            &selected,
        );
        diagnostics.push(BoundaryExactCoverMateCandidateDiagnostic {
            node_ids: candidates[candidate_index].node_ids,
            exact_scaled_jacobian: candidates[candidate_index].exact_scaled_jacobian,
            addable,
            conflicting_faces,
            missing_future_mate_faces,
        });
    }
    diagnostics.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverMateDiagnostic {
        target_face,
        candidate_count: diagnostics.len(),
        addable_count: diagnostics
            .iter()
            .filter(|candidate| candidate.addable)
            .count(),
        candidates: diagnostics,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_face_candidate_sources(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCandidateSourceDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let target_face = sorted_face(target_face);
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = BoundaryExactCoverFaceCandidateSourceDiagnostic {
        target_face,
        fourth_node_count: 0,
        centroid_inside_count: 0,
        solid_pass_count: 0,
        relaxed_pass_count: 0,
        outside_surface_count: 0,
        solid_rejected_by_reason: BTreeMap::new(),
        relaxed_rejected_by_reason: BTreeMap::new(),
        relaxed_candidate_node_ids: Vec::new(),
    };
    let face_nodes = target_face
        .map(|node_id| boundary_node_map.get(&node_id).copied())
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(ConstrainedCavityRefillError::MissingBoundaryNode {
            node_id: target_face[0],
        })?;
    for fourth_node_id in cavity_boundary_node_ids(cavity) {
        if target_face.contains(&fourth_node_id) {
            continue;
        }
        let Some(fourth_point) = boundary_node_map.get(&fourth_node_id).copied() else {
            return Err(ConstrainedCavityRefillError::MissingBoundaryNode {
                node_id: fourth_node_id,
            });
        };
        diagnostic.fourth_node_count += 1;
        let node_ids = [
            target_face[0],
            target_face[1],
            target_face[2],
            fourth_node_id,
        ];
        let points = [face_nodes[0], face_nodes[1], face_nodes[2], fourth_point];
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            diagnostic.outside_surface_count += 1;
            continue;
        }
        diagnostic.centroid_inside_count += 1;
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(_) => diagnostic.solid_pass_count += 1,
            Err(reason) => {
                *diagnostic
                    .solid_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, relaxed_options) {
            Ok(tetrahedron) => {
                diagnostic.relaxed_pass_count += 1;
                diagnostic
                    .relaxed_candidate_node_ids
                    .push(sorted_tetrahedron_nodes(tetrahedron.node_ids));
            }
            Err(reason) => {
                *diagnostic
                    .relaxed_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
    }
    diagnostic.relaxed_candidate_node_ids.sort();
    diagnostic.relaxed_candidate_node_ids.dedup();
    Ok(diagnostic)
}

pub fn selected_exact_cover_face_count_blockers(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCountBlockers, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let target_face = sorted_face(target_face);
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }

    let mut blockers = Vec::<BoundaryExactCoverFaceCountBlocker>::new();
    let mut candidate_count = 0_usize;
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .contains(&target_face)
                    {
                        continue;
                    }
                    if selected_keys.contains(&sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    candidate_count += 1;
                    let mut conflicting_faces = Vec::<[u32; 3]>::new();
                    let mut blocking_selected_tetrahedra = Vec::<[u32; 4]>::new();
                    for candidate_face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
                        let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
                        let conflicts = if boundary_faces.contains(&candidate_face) {
                            count != 0
                        } else {
                            count >= 2
                        };
                        if conflicts {
                            conflicting_faces.push(candidate_face);
                            if let Some(selected_tetrahedra) =
                                selected_tetrahedra_by_face.get(&candidate_face)
                            {
                                blocking_selected_tetrahedra
                                    .extend(selected_tetrahedra.iter().copied());
                            }
                        }
                    }
                    if !conflicting_faces.is_empty() {
                        blocking_selected_tetrahedra.sort();
                        blocking_selected_tetrahedra.dedup();
                        blockers.push(BoundaryExactCoverFaceCountBlocker {
                            node_ids: tetrahedron.node_ids,
                            exact_scaled_jacobian: tetrahedron.exact_scaled_jacobian,
                            conflicting_faces,
                            blocking_selected_tetrahedra,
                        });
                    }
                }
            }
        }
    }
    blockers.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverFaceCountBlockers {
        target_face,
        selected_tetrahedron_count: selected_tetrahedron_node_ids.len(),
        candidate_count,
        blocker_count: blockers.len(),
        blockers,
    })
}

pub fn selected_exact_cover_saturated_component(
    cavity: &ConstrainedCavity,
    selected_tetrahedron_node_ids: &[[u32; 4]],
    seed_face: [u32; 3],
) -> BoundaryExactCoverSaturatedComponent {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let seed_face = sorted_face(seed_face);
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }
    let saturated_faces = selected_tetrahedra_by_face
        .iter()
        .filter_map(|(face, selected_tetrahedra)| {
            (!boundary_faces.contains(face) && selected_tetrahedra.len() >= 2).then_some(*face)
        })
        .collect::<BTreeSet<_>>();
    let mut component_faces = BTreeSet::<[u32; 3]>::new();
    let mut component_tetrahedra = BTreeSet::<[u32; 4]>::new();
    let mut pending = Vec::<[u32; 3]>::new();
    if saturated_faces.contains(&seed_face) {
        pending.push(seed_face);
    }
    while let Some(face) = pending.pop() {
        if !component_faces.insert(face) {
            continue;
        }
        let Some(selected_tetrahedra) = selected_tetrahedra_by_face.get(&face) else {
            continue;
        };
        for selected_tetrahedron in selected_tetrahedra {
            if component_tetrahedra.insert(*selected_tetrahedron) {
                for adjacent_face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
                    if saturated_faces.contains(&adjacent_face)
                        && !component_faces.contains(&adjacent_face)
                    {
                        pending.push(adjacent_face);
                    }
                }
            }
        }
    }
    BoundaryExactCoverSaturatedComponent {
        seed_face,
        saturated_face_count: saturated_faces.len(),
        component_face_count: component_faces.len(),
        component_tetrahedron_count: component_tetrahedra.len(),
        component_faces: component_faces.into_iter().collect(),
        component_tetrahedra: component_tetrahedra.into_iter().collect(),
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_interior_mate_closure(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    if touches_boundary
                        && candidate_keys.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids))
                    {
                        candidates.push(tetrahedron.clone());
                    }
                    all_candidates.push(tetrahedron);
                }
            }
        }
    }
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    for _ in 0..4 {
        let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
        for candidate in &candidates {
            for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
                *face_counts.entry(face).or_default() += 1;
            }
        }
        let missing_faces = face_counts
            .iter()
            .filter_map(|(face, count)| {
                (!boundary_faces.contains(face) && *count == 1).then_some(*face)
            })
            .collect::<Vec<_>>();
        if missing_faces.is_empty() {
            break;
        }
        let mut added = false;
        for face in missing_faces {
            if let Some(indices) = all_candidates_by_face.get(&face) {
                let mut indices = indices.clone();
                indices.sort_by(|left, right| {
                    all_candidates[*right]
                        .exact_scaled_jacobian
                        .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
                });
                for index in indices {
                    if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                        break;
                    }
                    let candidate = &all_candidates[index];
                    if candidate_keys.insert(sorted_tetrahedron_nodes(candidate.node_ids)) {
                        candidates.push(candidate.clone());
                        added = true;
                        break;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let (selected, trace) = search.search_with_trace();
    let Some(selected) = selected else {
        let dead_end = trace.dead_end.clone();
        return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
            initial_candidate_count,
            candidate_count: candidates.len(),
            injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
            found_cover: false,
            selected_tetrahedron_count: 0,
            search_attempt_count: search.attempts,
            reason: if search.attempts > 5_000 {
                "search_exhausted"
            } else {
                "cover_not_found"
            },
            dead_end_reason: dead_end
                .as_ref()
                .map(|dead_end| dead_end.reason)
                .unwrap_or("not_evaluated"),
            dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
            dead_end_depth: dead_end
                .as_ref()
                .map(|dead_end| dead_end.depth)
                .unwrap_or(0),
            dead_end_selected_tetrahedra: dead_end
                .as_ref()
                .map(|dead_end| dead_end.selected_tetrahedra.clone())
                .unwrap_or_default(),
            dead_end_current_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.current_volume_m3)
                .unwrap_or(0.0),
            dead_end_candidate_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.candidate_volume_m3)
                .unwrap_or(0.0),
            dead_end_target_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.target_volume_m3)
                .unwrap_or(0.0),
            dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
            dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
            dead_end_selected_tetrahedra_by_reason: exact_cover_trace_selected_tetrahedra_by_reason(
                &trace,
            ),
            dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(&trace),
            unforced_found_cover: false,
            unforced_selected_tetrahedron_count: 0,
            unforced_search_attempt_count: 0,
            unforced_dead_end_reason_histogram: BTreeMap::new(),
        });
    };
    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: true,
        selected_tetrahedron_count: selected.len(),
        search_attempt_count: search.attempts,
        reason: "cover_found",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
        cavity,
        boundary_nodes,
        &[],
        options,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    excluded_tetrahedron_node_ids: &[[u32; 4]],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    let excluded_keys = excluded_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let candidate_key = sorted_tetrahedron_nodes(tetrahedron.node_ids);
                    if touches_boundary
                        && !excluded_keys.contains(&candidate_key)
                        && candidate_keys.insert(candidate_key)
                    {
                        candidates.push(tetrahedron.clone());
                    }
                    all_candidates.push(tetrahedron);
                }
            }
        }
    }
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    let mut total_attempts = 0_usize;
    for _ in 0..64 {
        let mut search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let (selected, trace) = search.search_with_trace();
        total_attempts += search.attempts;
        if let Some(selected) = selected {
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: true,
                selected_tetrahedron_count: selected.len(),
                search_attempt_count: total_attempts,
                reason: "cover_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover: false,
                unforced_selected_tetrahedron_count: 0,
                unforced_search_attempt_count: 0,
                unforced_dead_end_reason_histogram: BTreeMap::new(),
            });
        }
        let Some(dead_end) = trace.dead_end.clone() else {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: "cover_not_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        };
        let future_mate_dead_ends = trace
            .dead_ends
            .iter()
            .filter(|dead_end| dead_end.reason == "forced_interior_mate_future_mate_conflict")
            .cloned()
            .collect::<Vec<_>>();
        let no_candidate_dead_end_faces = trace
            .dead_ends
            .iter()
            .filter_map(|dead_end| {
                (dead_end.reason == "forced_interior_mate_no_candidate_contains_face")
                    .then_some(dead_end.face)
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        if future_mate_dead_ends.is_empty() && no_candidate_dead_end_faces.is_empty() {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
        let search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let mut mate_faces = BTreeSet::<[u32; 3]>::new();
        mate_faces.extend(no_candidate_dead_end_faces);
        for future_dead_end in &future_mate_dead_ends {
            let Some(face) = future_dead_end.face else {
                continue;
            };
            let selected_indices = future_dead_end
                .selected_tetrahedra
                .iter()
                .filter_map(|selected_tetrahedron| {
                    candidates.iter().position(|candidate| {
                        sorted_tetrahedron_nodes(candidate.node_ids)
                            == sorted_tetrahedron_nodes(*selected_tetrahedron)
                    })
                })
                .collect::<Vec<_>>();
            let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
            for selected_index in &selected_indices {
                for selected_face in
                    tetrahedron_faces(candidates[*selected_index].node_ids).map(sorted_face)
                {
                    *face_counts.entry(selected_face).or_default() += 1;
                }
            }
            for candidate_index in (0..candidates.len()).filter(|candidate_index| {
                !selected_indices.contains(candidate_index)
                    && search.candidate_faces[*candidate_index].contains(&face)
            }) {
                for candidate_face in search.candidate_faces[candidate_index] {
                    if !boundary_faces.contains(&candidate_face)
                        && face_counts.get(&candidate_face).copied().unwrap_or(0) == 0
                        && !search.interior_face_has_future_mate(
                            candidate_index,
                            candidate_face,
                            &face_counts,
                            &selected_indices,
                        )
                    {
                        mate_faces.insert(candidate_face);
                    }
                }
            }
        }
        let mut added = false;
        for mate_face in mate_faces {
            let Some(indices) = all_candidates_by_face.get(&mate_face) else {
                continue;
            };
            let mut indices = indices.clone();
            indices.sort_by(|left, right| {
                all_candidates[*right]
                    .exact_scaled_jacobian
                    .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
            });
            for index in indices {
                if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                    break;
                }
                let candidate = &all_candidates[index];
                let candidate_key = sorted_tetrahedron_nodes(candidate.node_ids);
                if !excluded_keys.contains(&candidate_key) && candidate_keys.insert(candidate_key) {
                    candidates.push(candidate.clone());
                    added = true;
                    break;
                }
            }
        }
        if !added {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
    }

    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: false,
        selected_tetrahedron_count: 0,
        search_attempt_count: total_attempts,
        reason: "iteration_limit",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundarySteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundarySteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let Some(centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };
    if point_in_closed_triangle_surface(centroid, &boundary_triangles, MeshingTolerance::default())
        != PointInClosedSurface::Inside
    {
        diagnostic.reason = "steiner_point_outside_cavity";
        return Ok(diagnostic);
    }
    let steiner_node_id = next_cavity_node_id(cavity);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    node_points.insert(steiner_node_id, centroid);
    let mut node_ids = boundary_node_ids.clone();
    node_ids.push(steiner_node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_patch_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryPatchSteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut diagnostic = BoundaryPatchSteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        missing_face_count: 0,
        patch_count: 0,
        steiner_node_count: 0,
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }

    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    diagnostic.missing_face_count = missing_faces.len();
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    diagnostic.patch_count = components.len();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut node_ids = boundary_node_ids.clone();
    let mut next_node_id = next_cavity_node_id(cavity);
    for component in components {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in component {
            patch_node_ids.extend(missing_faces[face_index]);
        }
        let Some(surface_point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) else {
            continue;
        };
        let Some(point) =
            patch_steiner_point_inside_cavity(surface_point, cavity_centroid, &boundary_triangles)
        else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, point);
        node_ids.push(next_node_id);
        diagnostic.steiner_node_count += 1;
        next_node_id = next_node_id.saturating_add(1);
    }
    if diagnostic.steiner_node_count == 0 {
        diagnostic.reason = "no_valid_patch_steiner_points";
        return Ok(diagnostic);
    }

    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 1_024 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<SupportNodeExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return Ok(SupportNodeExactCoverDiagnostic {
            candidate_node_count: candidate_nodes.len(),
            candidate_count: 0,
            root_zero_raw_boundary_face_count: 0,
            root_zero_raw_boundary_faces: Vec::new(),
            root_min_raw_boundary_face_candidate_count: 0,
            root_min_raw_candidate_boundary_faces: Vec::new(),
            root_max_raw_boundary_face_candidate_count: 0,
            root_zero_addable_boundary_face_count: 0,
            root_zero_addable_boundary_faces: Vec::new(),
            root_min_addable_boundary_face_candidate_count: 0,
            root_min_addable_candidate_boundary_faces: Vec::new(),
            root_max_addable_boundary_face_candidate_count: 0,
            selected_tetrahedron_count: 0,
            search_attempt_count: 0,
            found_cover: false,
            reason: "no_candidate_tetrahedra",
            dead_end_reason: "not_evaluated",
            dead_end_face: None,
            dead_end_depth: 0,
            dead_end_reason_histogram: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        });
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let root_raw_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    let root_zero_raw_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_raw_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_raw_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let root_addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    let root_zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let (selected, trace) = search.search_with_trace();
    let dead_end = trace.dead_end.clone();
    let dead_end_faces_by_reason = exact_cover_trace_faces_by_reason(&trace);
    Ok(SupportNodeExactCoverDiagnostic {
        candidate_node_count: candidate_nodes.len(),
        candidate_count: candidates.len(),
        root_zero_raw_boundary_face_count: root_zero_raw_boundary_faces.len(),
        root_zero_raw_boundary_faces,
        root_min_raw_boundary_face_candidate_count,
        root_min_raw_candidate_boundary_faces,
        root_max_raw_boundary_face_candidate_count,
        root_zero_addable_boundary_face_count: root_zero_addable_boundary_faces.len(),
        root_zero_addable_boundary_faces,
        root_min_addable_boundary_face_candidate_count,
        root_min_addable_candidate_boundary_faces,
        root_max_addable_boundary_face_candidate_count,
        selected_tetrahedron_count: selected.as_ref().map(Vec::len).unwrap_or(0),
        search_attempt_count: search.attempts,
        found_cover: selected.is_some(),
        reason: if selected.is_some() {
            "cover_found"
        } else if search.attempts > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        },
        dead_end_reason: dead_end
            .as_ref()
            .map(|dead_end| dead_end.reason)
            .unwrap_or("not_evaluated"),
        dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
        dead_end_depth: dead_end.map(|dead_end| dead_end.depth).unwrap_or(0),
        dead_end_reason_histogram: trace.dead_end_reason_counts,
        dead_end_faces_by_reason,
    })
}
