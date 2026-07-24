use super::*;

mod recording;

impl<'a> BoundaryExactCoverSearch<'a> {
    pub(in super::super::super) fn search_from_traced(
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
            if self.cover_is_complete(face_counts, current_volume_m3) {
                return Some(selected.clone());
            }
            self.record_dead_end(
                trace,
                selected,
                selected_roles,
                self.incomplete_cover_reason(face_counts, current_volume_m3),
            );
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
    pub(in super::super::super) fn search_from_without_forced_traced(
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
            if self.cover_is_complete(face_counts, current_volume_m3) {
                return Some(selected.clone());
            }
            self.record_dead_end(
                trace,
                selected,
                selected_roles,
                self.incomplete_cover_reason(face_counts, current_volume_m3),
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
}
