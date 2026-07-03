use super::*;

mod forced_mates;

impl<'a> BoundaryExactCoverSearch<'a> {
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
