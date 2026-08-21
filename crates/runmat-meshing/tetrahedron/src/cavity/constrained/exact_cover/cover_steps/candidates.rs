use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
    pub(in super::super::super) fn candidates_exhausted_reason_and_face(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> (&'static str, Option<[u32; 3]>) {
        if let Some(face) = self
            .boundary_faces
            .iter()
            .find(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
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

        if let Some(face) = face_counts
            .iter()
            .filter_map(|(face, count)| {
                (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
            })
            .next()
        {
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

    pub(in super::super::super) fn raw_candidate_count_for_face(
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

    pub(in super::super::super) fn addable_candidate_count_for_face(
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

    pub(in super::super::super) fn candidate_can_be_added_for_face(
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

    pub(in super::super::super) fn interior_face_has_future_mate(
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
}
