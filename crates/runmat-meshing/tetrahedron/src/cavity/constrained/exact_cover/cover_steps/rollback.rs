use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
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
