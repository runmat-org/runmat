//! Incremental quality owned beside a working mesh. Trials retain only changed
//! cells; the original spectrum ordering is represented by exact bin counts.
use super::*;
use crate::cdt::topology::VolumeTopologyRead;
use std::{collections::BTreeMap, sync::Arc};
mod context;
mod frontier;
mod grading;
mod output;
mod ranking;
mod trial;
pub(in crate::cdt) use context::PreparedLocalQuality;
pub(in crate::cdt) use frontier::QualityCursor;
use ranking::Ranking;
pub(in crate::cdt) use ranking::SpectrumDelta;
pub(in crate::cdt) use trial::QualityTrial;
type CellKey = [StableDigest; 4];
#[cfg(test)]
mod tests;

#[derive(Clone, PartialEq)]
struct Entry {
    quality: DelaunayTetrahedronQuality,
    geometry: [[u64; 3]; 4],
    grading: Option<grading::Dependency>,
}

pub(in crate::cdt) struct WorkingVolumeQuality {
    cells: BTreeMap<CellKey, Entry>,
    shape: Ranking,
    refinement: Ranking,
    journals: Vec<BTreeMap<CellKey, Option<Arc<Entry>>>>,
    baseline_originals: BTreeMap<CellKey, Option<Arc<Entry>>>,
    comparison_ready: bool,
    repair_floor: Option<u64>,
    last_scoped_update: Option<(usize, usize)>,
}

impl WorkingVolumeQuality {
    pub(in crate::cdt) fn new(
        quality: DelaunayVolumeQuality,
        topology: &(impl VolumeTopologyRead + ?Sized),
        options: DelaunayVolumeQualityOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayVolumeQualityError> {
        checkpoint(0, options, cancellation)?;
        if topology.nodes().len() as u64 > options.maximum_nodes
            || quality.tetrahedra.len() as u64 > options.maximum_tetrahedra
        {
            return Err(error(
                DelaunayVolumeQualityErrorKind::ResourceLimit,
                None,
                "working quality exceeds its current inventory limit",
            ));
        }
        let mut result = Self {
            cells: BTreeMap::new(),
            shape: Ranking::default(),
            refinement: Ranking::default(),
            journals: Vec::new(),
            baseline_originals: BTreeMap::new(),
            comparison_ready: false,
            repair_floor: None,
            last_scoped_update: None,
        };
        for (index, quality) in quality.tetrahedra.into_iter().enumerate() {
            checkpoint(index as u64, options, cancellation)?;
            let geometry = geometry_key(topology, quality.node_identities)?;
            let key = cell_key(quality.node_identities);
            if result.cells.contains_key(&key) {
                return Err(error(
                    DelaunayVolumeQualityErrorKind::InvalidQuality,
                    Some(index),
                    "working quality contains duplicate cells",
                ));
            }
            result.replace(
                key,
                Some(Entry {
                    quality,
                    geometry,
                    grading: None,
                }),
            );
        }
        result.reset_comparison();
        Ok(result)
    }

    /// Set the current accepted state as the target once, outside all trials.
    pub(in crate::cdt) fn reset_comparison(&mut self) {
        assert!(
            self.journals.is_empty(),
            "comparison target cannot change inside a trial"
        );
        self.repair_floor = self
            .shape
            .priorities
            .last_key_value()
            .map(|(rank, _)| rank.0)
            .into_iter()
            .chain(
                self.refinement
                    .priorities
                    .last_key_value()
                    .map(|(rank, _)| rank.0),
            )
            .max();
        self.shape.delta.clear();
        self.refinement.delta.clear();
        self.baseline_originals.clear();
        self.comparison_ready = true;
    }

    pub(in crate::cdt) fn compare_baseline(&self, objective: QualityObjective) -> Ordering {
        self.ranking(objective).compare_baseline()
    }

    pub(in crate::cdt) fn worst(
        &self,
        objective: QualityObjective,
    ) -> Option<&DelaunayTetrahedronQuality> {
        self.ranking(objective)
            .priorities
            .last_key_value()
            .map(|(_, key)| &self.cells[key].quality)
    }

    /// Remaining (shape, refinement) violations; a constant-time diagnostic read.
    pub(in crate::cdt) fn violation_counts(&self) -> (usize, usize) {
        (
            self.shape.priorities.len(),
            self.refinement.priorities.len(),
        )
    }

    pub(in crate::cdt) fn cell_count(&self) -> usize {
        self.cells.len()
    }
    /// Mutable slots are not canonical quality ordinals: resolve their corners.
    pub(in crate::cdt) fn quality_for_cell(
        &self,
        identities: CellKey,
    ) -> Option<&DelaunayTetrahedronQuality> {
        self.cells
            .get(&cell_key(identities))
            .map(|entry| &entry.quality)
    }
    #[cfg(test)]
    pub(in crate::cdt) fn retained_trial_entry_count(&self) -> usize {
        self.journals.iter().map(BTreeMap::len).sum()
    }

    #[cfg(test)]
    pub(in crate::cdt) fn last_scoped_update_work(&self) -> Option<(usize, usize)> {
        self.last_scoped_update
    }

    /// Bounded output for a terminal diagnostic. Search ranking uses maintained
    /// deltas; this complete read occurs only after candidate search has failed.
    pub(in crate::cdt) fn diagnostic_spectrum_prefix(
        &self,
        objective: QualityObjective,
        limit: usize,
    ) -> Vec<f64> {
        let mut result = Vec::with_capacity(limit + 1);
        for entry in self.cells.values() {
            let ratio = match objective {
                QualityObjective::Shape => entry.quality.optimization_violation_ratio,
                QualityObjective::Refinement => entry.quality.refinement_violation_ratio,
            };
            if ratio <= 1.0 {
                continue;
            }
            let position = result.partition_point(|value: &f64| value.total_cmp(&ratio).is_gt());
            if position < limit {
                result.insert(position, ratio);
                result.truncate(limit);
            }
        }
        result
    }

    /// Counts are meaningful only against the same unchanged comparison target;
    /// the search owner binds retained deltas to that target's lifetime/revision.
    pub(in crate::cdt) fn capture_spectrum(&self) -> SpectrumDelta {
        SpectrumDelta::capture(self)
    }
    pub(in crate::cdt) fn compare_target(
        &self,
        target: &SpectrumDelta,
        objective: QualityObjective,
    ) -> Ordering {
        target.compare_owner(self, objective)
    }

    #[cfg(test)]
    pub(in crate::cdt) fn into_canonical(
        self,
        topology: &DelaunayVolumeTopology,
        options: DelaunayVolumeQualityOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
        self.into_cells(options, cancellation)?
            .summarize(topology, options, cancellation)
    }

    fn ranking(&self, objective: QualityObjective) -> &Ranking {
        match objective {
            QualityObjective::Shape => &self.shape,
            QualityObjective::Refinement => &self.refinement,
        }
    }

    fn replace(&mut self, key: CellKey, entry: Option<Entry>) {
        let old = self.cells.remove(&key);
        if let Some(old) = &old {
            self.shape.update(
                old,
                key,
                QualityObjective::Shape,
                false,
                self.comparison_ready,
            );
            self.refinement.update(
                old,
                key,
                QualityObjective::Refinement,
                false,
                self.comparison_ready,
            );
        }
        let retain_baseline = self.comparison_ready && !self.baseline_originals.contains_key(&key);
        let retain_undo = self
            .journals
            .iter()
            .any(|journal| !journal.contains_key(&key));
        if retain_baseline || retain_undo {
            // Most accepted cells never need a shared allocation. Move an old
            // record once when it becomes reversible; all live undo owners share it.
            let original = old.map(Arc::new);
            if retain_baseline {
                self.baseline_originals.insert(key, original.clone());
            }
            for journal in &mut self.journals {
                journal.entry(key).or_insert_with(|| original.clone());
            }
        }
        if let Some(entry) = entry {
            self.shape.update(
                &entry,
                key,
                QualityObjective::Shape,
                true,
                self.comparison_ready,
            );
            self.refinement.update(
                &entry,
                key,
                QualityObjective::Refinement,
                true,
                self.comparison_ready,
            );
            self.cells.insert(key, entry);
        }
        // Undo represents the difference from each live parent's start, not
        // every transient cell ever visited by speculative search.
        for journal in &mut self.journals {
            if journal
                .get(&key)
                .is_some_and(|old| old.as_deref() == self.cells.get(&key))
            {
                journal.remove(&key);
            }
        }
        if self.comparison_ready
            && self
                .baseline_originals
                .get(&key)
                .is_some_and(|old| old.as_deref() == self.cells.get(&key))
        {
            self.baseline_originals.remove(&key);
        }
    }

    pub(in crate::cdt) fn baseline_contains(&self, key: &CellKey) -> bool {
        self.baseline_originals
            .get(key)
            .map_or_else(|| self.cells.contains_key(key), Option::is_some)
    }

    pub(in crate::cdt) fn baseline_successor(&self, key: &CellKey) -> Option<CellKey> {
        use std::ops::Bound::{Excluded, Unbounded};
        let current = self
            .cells
            .range((Excluded(*key), Unbounded))
            .find(|(key, _)| !self.baseline_originals.contains_key(*key))
            .map(|(key, _)| *key);
        let original = self
            .baseline_originals
            .range((Excluded(*key), Unbounded))
            .find(|(_, entry)| entry.is_some())
            .map(|(key, _)| *key);
        match (current, original) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}

fn cell_key(mut identities: CellKey) -> CellKey {
    identities.sort_unstable();
    identities
}

fn geometry_key(
    topology: &(impl VolumeTopologyRead + ?Sized),
    identities: CellKey,
) -> Result<[[u64; 3]; 4], DelaunayVolumeQualityError> {
    let mut result = [[0; 3]; 4];
    for (slot, identity) in identities.into_iter().enumerate() {
        let index = topology.node_index(identity).ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                None,
                "working quality corner is absent",
            )
        })?;
        result[slot] = topology
            .node(index)
            .coordinates_m
            .canonical_value()
            .map(ordered_float_key);
    }
    result.sort_unstable();
    Ok(result)
}
