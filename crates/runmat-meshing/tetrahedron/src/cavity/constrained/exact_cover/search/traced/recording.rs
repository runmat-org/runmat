use super::*;

impl<'a> BoundaryExactCoverSearch<'a> {
    #[cfg(test)]
    pub(in super::super::super::super) fn search_with_trace(
        &mut self,
    ) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
        self.search_with_trace_controlled(&runmat_meshing_core::NeverCancelled, u64::MAX)
            .expect("the never-cancelled exact-cover search cannot cancel")
    }

    pub(in crate::cavity::constrained) fn search_with_trace_controlled(
        &mut self,
        cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
        cancellation_check_interval: u64,
    ) -> Result<(Option<Vec<usize>>, BoundaryExactCoverTrace), ()> {
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
            ExactCoverSearchCancellation {
                signal: cancellation,
                interval: cancellation_check_interval,
            },
        )?;
        Ok((result, trace))
    }

    pub(in super::super::super::super) fn record_dead_end(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
    ) {
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, None, None);
    }

    pub(in super::super::super::super) fn record_dead_end_for_face(
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
    pub(in super::super::super::super) fn root_boundary_availability(
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
}
