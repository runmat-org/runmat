//! Local net edge-loss observation without retaining spatial-index snapshots.
//! Physical rollback remains the cavity trial's responsibility. Nested rejected
//! edits coalesce by comparing first-written cell corners to the current owner.
use super::TopologyMutation;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
};

pub(in crate::cdt) struct EdgeChangeObserver<'a> {
    topology: &'a mut TopologyMutation,
}

impl TopologyMutation {
    pub(in crate::cdt) fn observe_edge_changes(&mut self) -> EdgeChangeObserver<'_> {
        self.edge_observers.push(BTreeMap::new());
        EdgeChangeObserver { topology: self }
    }
}

/// Net cell change since an observer started: the corners of every observed slot whose
/// content changed, and the active slots that now hold a different cell.
pub(in crate::cdt) struct ObservedCellChanges {
    pub(in crate::cdt) removed: Vec<[u32; 4]>,
    pub(in crate::cdt) created: Vec<u32>,
}

impl EdgeChangeObserver<'_> {
    pub(in crate::cdt) fn changed_cells(&self) -> ObservedCellChanges {
        let observer = self
            .topology
            .edge_observers
            .last()
            .expect("live edge observer");
        let sorted = |mut vertices: [u32; 4]| {
            vertices.sort_unstable();
            vertices
        };
        let mut changes = ObservedCellChanges {
            removed: Vec::new(),
            created: Vec::new(),
        };
        for (&index, previous) in observer {
            let current = self
                .topology
                .tetrahedra
                .get(index as usize)
                .and_then(Option::as_ref)
                .map(|cell| cell.vertex_indices);
            if previous.map(sorted) == current.map(sorted) {
                continue;
            }
            if let Some(previous) = previous {
                changes.removed.push(*previous);
            }
            if current.is_some() {
                changes.created.push(index);
            }
        }
        changes
    }

    pub(in crate::cdt) fn removed_edge_candidates<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<BTreeSet<[u32; 2]>, E> {
        let observer = self
            .topology
            .edge_observers
            .last()
            .expect("live edge observer");
        let mut edges = BTreeSet::new();
        for (&index, previous) in observer {
            checkpoint()?;
            let Some(mut before) = *previous else {
                continue;
            };
            before.sort_unstable();
            let after = self
                .topology
                .tetrahedra
                .get(index as usize)
                .and_then(Option::as_ref)
                .map(|cell| {
                    let mut vertices = cell.vertex_indices;
                    vertices.sort_unstable();
                    vertices
                });
            if after == Some(before) {
                continue;
            }
            for first in 0..4 {
                for second in first + 1..4 {
                    checkpoint()?;
                    edges.insert([before[first], before[second]]);
                }
            }
        }
        Ok(edges)
    }
}

impl Deref for EdgeChangeObserver<'_> {
    type Target = TopologyMutation;
    fn deref(&self) -> &Self::Target {
        self.topology
    }
}
impl DerefMut for EdgeChangeObserver<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.topology
    }
}
impl Drop for EdgeChangeObserver<'_> {
    fn drop(&mut self) {
        self.topology
            .edge_observers
            .pop()
            .expect("live edge observer");
    }
}
