//! Retain first-write undo only while current values differ from a trial's start.
//! Work is proportional to the touched cavity, never the full mesh or journal.
use super::*;

pub(in crate::cdt::mutation) struct CavityJournalKeys {
    pub(super) slots: BTreeSet<u32>,
    pub(super) keys: BTreeSet<[u32; 4]>,
    pub(super) faces: BTreeSet<Face>,
    pub(super) stars: BTreeSet<(u32, u32)>,
    pub(super) nodes: BTreeSet<u32>,
}
impl CavityJournalKeys {
    pub(super) fn from_journal(journal: &TrialJournal) -> Self {
        Self {
            slots: journal
                .cells
                .keys()
                .chain(journal.free.keys())
                .copied()
                .collect(),
            keys: journal.cell_keys.keys().copied().collect(),
            faces: journal
                .faces
                .keys()
                .chain(journal.boundary.keys())
                .chain(journal.dirty.keys())
                .copied()
                .collect(),
            stars: journal.stars.keys().copied().collect(),
            nodes: journal.nodes.keys().copied().collect(),
        }
    }
    /// A child can temporarily restore an ancestor's baseline, removing that
    /// ancestor's undo. Before restoring the child's start, capture the current
    /// baseline again for any ancestor whose undo was compacted away.
    pub(super) fn record_parents(&self, topology: &mut TopologyMutation) {
        for journal in &mut topology.trial_journals {
            for index in &self.slots {
                journal
                    .cells
                    .entry(*index)
                    .or_insert_with(|| topology.tetrahedra.get(*index as usize).cloned().flatten());
                journal
                    .free
                    .entry(*index)
                    .or_insert_with(|| topology.free_tetrahedra.contains(index));
            }
            for key in &self.keys {
                journal
                    .cell_keys
                    .entry(*key)
                    .or_insert_with(|| topology.tetrahedron_indices.get(key).copied());
            }
            for key in &self.stars {
                journal.stars.entry(*key).or_insert_with(|| {
                    topology
                        .vertex_stars
                        .get(key.0 as usize)
                        .is_some_and(|star| star.contains(&key.1))
                });
            }
            for face in &self.faces {
                journal
                    .faces
                    .entry(*face)
                    .or_insert_with(|| topology.face_uses.get(face).cloned());
                journal
                    .boundary
                    .entry(*face)
                    .or_insert_with(|| topology.boundary_faces.get(face).copied());
                journal.dirty.entry(*face).or_insert_with(|| {
                    topology
                        .physical_dirty_faces
                        .as_ref()
                        .is_some_and(|set| set.contains(face))
                });
            }
            for index in &self.nodes {
                if let Some(node) = topology.nodes.get(*index as usize) {
                    journal.nodes.entry(*index).or_insert(*node);
                }
            }
        }
    }
    pub(in crate::cdt::mutation) fn compact(self, topology: &mut TopologyMutation) {
        for journal in &mut topology.trial_journals {
            for index in &self.slots {
                if journal.cells.get(index).is_some_and(|old| {
                    old.as_ref()
                        == topology
                            .tetrahedra
                            .get(*index as usize)
                            .and_then(Option::as_ref)
                }) {
                    journal.cells.remove(index);
                }
                if journal
                    .free
                    .get(index)
                    .is_some_and(|old| *old == topology.free_tetrahedra.contains(index))
                {
                    journal.free.remove(index);
                }
            }
            for key in &self.keys {
                if journal
                    .cell_keys
                    .get(key)
                    .is_some_and(|old| old.as_ref() == topology.tetrahedron_indices.get(key))
                {
                    journal.cell_keys.remove(key);
                }
            }
            for key in &self.stars {
                let present = topology
                    .vertex_stars
                    .get(key.0 as usize)
                    .is_some_and(|star| star.contains(&key.1));
                if journal.stars.get(key).is_some_and(|old| *old == present) {
                    journal.stars.remove(key);
                }
            }
            for face in &self.faces {
                if journal
                    .faces
                    .get(face)
                    .is_some_and(|old| old.as_ref() == topology.face_uses.get(face))
                {
                    journal.faces.remove(face);
                }
                if journal
                    .boundary
                    .get(face)
                    .is_some_and(|old| old.as_ref() == topology.boundary_faces.get(face))
                {
                    journal.boundary.remove(face);
                }
                let dirty = topology
                    .physical_dirty_faces
                    .as_ref()
                    .is_some_and(|set| set.contains(face));
                if journal.dirty.get(face).is_some_and(|old| *old == dirty) {
                    journal.dirty.remove(face);
                }
            }
            for index in &self.nodes {
                if journal
                    .nodes
                    .get(index)
                    .is_some_and(|old| Some(old) == topology.nodes.get(*index as usize))
                {
                    journal.nodes.remove(index);
                }
            }
        }
    }
}
