use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
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
