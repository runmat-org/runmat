use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
    pub(in super::super) fn cover_is_complete(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        current_volume_m3: f64,
    ) -> bool {
        self.boundary_faces_are_covered(face_counts)
            && self.interior_faces_are_paired(face_counts)
            && self.volume_matches_target(current_volume_m3)
    }

    pub(in super::super) fn incomplete_cover_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        current_volume_m3: f64,
    ) -> &'static str {
        if !self.boundary_faces_are_covered(face_counts) {
            "boundary_incomplete"
        } else if !self.interior_faces_are_paired(face_counts) {
            "interior_incomplete"
        } else if !self.volume_matches_target(current_volume_m3) {
            "volume_mismatch"
        } else {
            "cover_complete"
        }
    }

    fn boundary_faces_are_covered(&self, face_counts: &BTreeMap<[u32; 3], usize>) -> bool {
        self.boundary_faces
            .iter()
            .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1)
    }

    fn interior_faces_are_paired(&self, face_counts: &BTreeMap<[u32; 3], usize>) -> bool {
        face_counts
            .iter()
            .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2)
    }

    fn volume_matches_target(&self, current_volume_m3: f64) -> bool {
        (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
    }
}
