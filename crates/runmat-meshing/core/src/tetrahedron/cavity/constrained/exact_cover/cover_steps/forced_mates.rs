use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
    pub(in super::super::super) fn propagate_forced_interior_mates(
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

    pub(in super::super::super) fn propagate_forced_interior_mates_traced(
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

    pub(in super::super::super) fn forced_interior_mate(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        self.forced_interior_mate_traced(face_counts, selected)
    }

    pub(in super::super::super) fn forced_interior_mate_traced(
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

    pub(in super::super::super) fn forced_interior_mate_no_addable_reason(
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

    pub(in super::super::super) fn rollback_selected_candidates(
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

    pub(in super::super::super) fn rollback_selected_candidates_with_roles(
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
}
