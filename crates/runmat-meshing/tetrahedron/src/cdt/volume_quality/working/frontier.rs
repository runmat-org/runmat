//! Bounded reads of the maintained work ranking; no complete scan or sort.
use super::*;
use std::ops::Bound::{Excluded, Unbounded};

#[derive(Clone, Copy)]
pub(in crate::cdt) struct QualityCursor(ranking::Rank);

impl WorkingVolumeQuality {
    pub(in crate::cdt) fn refinement_window(
        &self,
        after: Option<QualityCursor>,
        maximum: usize,
    ) -> Vec<(QualityCursor, &DelaunayTetrahedronQuality)> {
        let upper = after.map_or(Unbounded, |position| Excluded(position.0));
        self.refinement
            .priorities
            .range((Unbounded, upper))
            .rev()
            .take(maximum)
            .map(|(rank, key)| (QualityCursor(*rank), &self.cells[key].quality))
            .collect()
    }

    /// Only new keys in the active comparison delta can introduce a new fully
    /// constrained cell. The unchanged baseline does not need another traversal.
    pub(in crate::cdt) fn creates_all_boundary_refinement_violation(&self) -> bool {
        self.baseline_originals.iter().any(|(key, original)| {
            original.is_none()
                && self.cells.get(key).is_some_and(|entry| {
                    entry.quality.boundary_node_count == 4 && entry.quality.requires_refinement()
                })
        })
    }
}
