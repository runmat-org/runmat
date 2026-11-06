//! Nested local trials retain first-write undo entries, never a whole mesh copy.
//! Dropping a trial restores cells, incidence, point additions and cached authority;
//! committing an inner trial still leaves its changes reversible by its parent.
use super::{
    spatial_cells::{CellSpatialCache, CellSpatialCheckpoint},
    storage::MutableTetrahedron,
    TopologyMutation,
};
use crate::cdt::topology::ConnectivitySearchState;
use runmat_meshing_core::{PointSpatialIndex, StableDigest};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    sync::Arc,
};
mod compaction;
mod record;
#[cfg(test)]
mod tests;

type Face = [u32; 3];
type FaceUses = Vec<(u32, usize)>;
#[derive(Clone, Debug)]
pub(super) struct TrialJournal {
    trace_len: usize,
    node_len: usize,
    cell_len: usize,
    coordinate_scale_m: f64,
    point_index: Option<Arc<PointSpatialIndex<StableDigest>>>,
    cell_index: Option<CellSpatialCheckpoint>,
    connectivity: Option<ConnectivitySearchState>,
    cells: BTreeMap<u32, Option<MutableTetrahedron>>,
    nodes: BTreeMap<u32, crate::cdt::DelaunayVolumeNode>,
    cell_keys: BTreeMap<[u32; 4], Option<u32>>,
    stars: BTreeMap<(u32, u32), bool>,
    faces: BTreeMap<Face, Option<FaceUses>>,
    boundary: BTreeMap<Face, Option<(u32, usize)>>,
    free: BTreeMap<u32, bool>,
    dirty: BTreeMap<Face, bool>,
}

/// Whole-owner adoption must happen outside a trial. Local providers edit through
/// the ordinary checked cavity/point APIs; unwinding also rolls their edits back.
pub(in crate::cdt) struct CavityTrial<'a> {
    topology: &'a mut TopologyMutation,
    committed: bool,
}
impl TopologyMutation {
    pub(in crate::cdt) fn cavity_trial(&mut self) -> CavityTrial<'_> {
        let journal = TrialJournal {
            trace_len: self.edit_trace.len(),
            node_len: self.nodes.len(),
            cell_len: self.tetrahedra.len(),
            coordinate_scale_m: self.coordinate_scale_m,
            point_index: self.spatial_index.borrow().clone(),
            cell_index: self
                .cell_spatial_index
                .borrow_mut()
                .as_mut()
                .map(CellSpatialCache::checkpoint),
            connectivity: *self.connectivity_state.borrow(),
            cells: BTreeMap::new(),
            nodes: BTreeMap::new(),
            cell_keys: BTreeMap::new(),
            stars: BTreeMap::new(),
            faces: BTreeMap::new(),
            boundary: BTreeMap::new(),
            free: BTreeMap::new(),
            dirty: BTreeMap::new(),
        };
        self.trial_journals.push(journal);
        CavityTrial {
            topology: self,
            committed: false,
        }
    }
}
impl TopologyMutation {
    /// Opens a trial that no borrow holds, for an owner that edits through its own API while the
    /// trial is open. Every path must close it with [`Self::finish_detached_trial`]; trials opened
    /// inside it nest as usual. Whole-owner adoption must not happen while it is open.
    pub(in crate::cdt) fn begin_detached_trial(&mut self) {
        std::mem::forget(self.cavity_trial());
    }

    /// Closes the innermost trial opened by [`Self::begin_detached_trial`]: commits it, or rolls
    /// back every edit made since it opened.
    pub(in crate::cdt) fn finish_detached_trial(&mut self, commit: bool) {
        let trial = CavityTrial {
            topology: self,
            committed: false,
        };
        if commit {
            trial.commit();
        }
    }
}

impl CavityTrial<'_> {
    pub(in crate::cdt) fn commit(mut self) {
        let journal = self
            .topology
            .trial_journals
            .pop()
            .expect("live local trial");
        if let Some(checkpoint) = journal.cell_index {
            self.topology
                .cell_spatial_index
                .get_mut()
                .as_mut()
                .expect("retained cell cache")
                .commit_checkpoint(checkpoint);
        }
        self.committed = true;
    }
}
impl Deref for CavityTrial<'_> {
    type Target = TopologyMutation;
    fn deref(&self) -> &Self::Target {
        self.topology
    }
}
impl DerefMut for CavityTrial<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.topology
    }
}
impl Drop for CavityTrial<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let journal = self
                .topology
                .trial_journals
                .pop()
                .expect("live local trial");
            journal.rollback(self.topology);
        }
    }
}
impl TrialJournal {
    fn rollback(self, topology: &mut TopologyMutation) {
        // Capture only keys this child touched. Parent compaction after restore
        // must not scan unrelated accepted mesh or accumulated parent history.
        let journal_keys = compaction::CavityJournalKeys::from_journal(&self);
        journal_keys.record_parents(topology);
        topology.edit_trace.truncate(self.trace_len);
        // Remove all transient identities/positions before restoring old ones:
        // a later edit can legitimately reuse a position vacated by an earlier
        // edit. Sequential remove/restore would lose that original membership.
        for node in topology.nodes.drain(self.node_len..) {
            topology.node_indices.remove(&node.identity);
            topology.positions.remove(
                &node
                    .coordinates_m
                    .canonical_value()
                    .map(super::storage::coordinate_bits),
            );
        }
        for index in self
            .nodes
            .keys()
            .filter(|index| (**index as usize) < self.node_len)
        {
            let current = topology.nodes[*index as usize];
            topology.node_indices.remove(&current.identity);
            topology.positions.remove(
                &current
                    .coordinates_m
                    .canonical_value()
                    .map(super::storage::coordinate_bits),
            );
        }
        for (index, node) in self.nodes {
            if (index as usize) < self.node_len {
                topology.nodes[index as usize] = node;
                topology.node_indices.insert(node.identity, index);
                topology.positions.insert(
                    node.coordinates_m
                        .canonical_value()
                        .map(super::storage::coordinate_bits),
                );
            }
        }
        for (index, cell) in self.cells {
            if (index as usize) < self.cell_len {
                topology.tetrahedra[index as usize] = cell;
            }
        }
        topology.tetrahedra.truncate(self.cell_len);
        restore_map(&mut topology.tetrahedron_indices, self.cell_keys);
        for ((vertex, cell), present) in self.stars {
            if (vertex as usize) < self.node_len {
                if present {
                    topology.vertex_stars[vertex as usize].insert(cell);
                } else {
                    topology.vertex_stars[vertex as usize].remove(&cell);
                }
            }
        }
        restore_map(&mut topology.face_uses, self.faces);
        restore_map(&mut topology.boundary_faces, self.boundary);
        for (index, present) in self.free {
            restore_set(&mut topology.free_tetrahedra, index, present);
        }
        if let Some(dirty) = &mut topology.physical_dirty_faces {
            for (face, present) in self.dirty {
                restore_set(dirty, face, present);
            }
        }
        topology.vertex_stars.truncate(self.node_len);
        topology.coordinate_scale_m = self.coordinate_scale_m;
        *topology.spatial_index.get_mut() = self.point_index;
        if let Some(checkpoint) = self.cell_index {
            topology
                .cell_spatial_index
                .get_mut()
                .as_mut()
                .expect("retained cell cache")
                .rollback_checkpoint(checkpoint);
        } else {
            *topology.cell_spatial_index.get_mut() = None;
        }
        *topology.connectivity_state.get_mut() = self.connectivity;
        journal_keys.compact(topology);
    }
}
fn restore_map<K: Ord, V>(target: &mut BTreeMap<K, V>, journal: BTreeMap<K, Option<V>>) {
    for (key, value) in journal {
        if let Some(value) = value {
            target.insert(key, value);
        } else {
            target.remove(&key);
        }
    }
}
fn restore_set<K: Ord>(target: &mut BTreeSet<K>, key: K, present: bool) {
    if present {
        target.insert(key);
    } else {
        target.remove(&key);
    }
}
