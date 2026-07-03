use super::*;

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

    pub(in super::super) fn search_with_trace(
        &mut self,
    ) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
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
    pub(in super::super) fn search_without_forced_with_trace(
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

    pub(in super::super) fn record_dead_end(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
    ) {
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, None, None);
    }

    pub(in super::super) fn record_dead_end_for_face(
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
    pub(in super::super) fn root_boundary_availability(
        &self,
    ) -> BoundaryExactCoverRootAvailability {
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

    pub(in super::super) fn search_from_traced(
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
    pub(in super::super) fn search_from_without_forced_traced(
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

    pub(in super::super) fn propagate_forced_interior_mates(
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

    pub(in super::super) fn propagate_forced_interior_mates_traced(
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

    pub(in super::super) fn forced_interior_mate(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        self.forced_interior_mate_traced(face_counts, selected)
    }

    pub(in super::super) fn forced_interior_mate_traced(
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

    pub(in super::super) fn forced_interior_mate_no_addable_reason(
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

    pub(in super::super) fn rollback_selected_candidates(
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

    pub(in super::super) fn rollback_selected_candidates_with_roles(
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

    pub(in super::super) fn add_candidate_faces(
        &self,
        candidate_index: usize,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
    ) {
        for face in self.candidate_faces[candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    pub(in super::super) fn remove_candidate_faces(
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

    pub(in super::super) fn next_cover_candidates(
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
    pub(in super::super) fn candidates_exhausted_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> &'static str {
        self.candidates_exhausted_reason_and_face(face_counts, selected)
            .0
    }

    pub(in super::super) fn candidates_exhausted_reason_and_face(
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

    pub(in super::super) fn raw_candidate_count_for_face(
        &self,
        face: [u32; 3],
        selected: &[usize],
    ) -> usize {
        (0..self.candidates.len())
            .filter(|candidate_index| {
                !selected.contains(candidate_index)
                    && self.candidate_faces[*candidate_index].contains(&face)
            })
            .count()
    }

    pub(in super::super) fn addable_candidate_count_for_face(
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

    pub(in super::super) fn candidate_can_be_added_for_face(
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

    pub(in super::super) fn interior_face_has_future_mate(
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

    pub(in super::super) fn sort_cover_candidates(&self, candidates: &mut [usize]) {
        candidates.sort_by(|left, right| {
            self.candidates[*right]
                .exact_scaled_jacobian
                .total_cmp(&self.candidates[*left].exact_scaled_jacobian)
        });
    }
}
