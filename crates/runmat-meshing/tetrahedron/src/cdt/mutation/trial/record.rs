//! Snapshot only entries a prepared atomic cavity will change, in every live trial.
use super::*;
impl TopologyMutation {
    pub(in crate::cdt::mutation) fn record_trial_point(&mut self, index: u32, cells: &[u32]) {
        let node = self.nodes[index as usize];
        let faces = cells
            .iter()
            .flat_map(|index| {
                let vertices = self
                    .tetrahedron(*index)
                    .expect("live point star")
                    .vertex_indices;
                (0..4).map(move |opposite| Self::face(vertices, opposite))
            })
            .collect::<BTreeSet<_>>();
        for journal in &mut self.trial_journals {
            journal.nodes.entry(index).or_insert(node);
            if let Some(dirty) = &self.physical_dirty_faces {
                for face in &faces {
                    journal
                        .dirty
                        .entry(*face)
                        .or_insert_with(|| dirty.contains(face));
                }
            }
        }
    }
    pub(in crate::cdt::mutation) fn record_trial_cavity(
        &mut self,
        removed: &BTreeSet<u32>,
        created: &[(u32, MutableTetrahedron)],
        neighbors: &[(u32, usize, u32)],
    ) -> Option<super::compaction::CavityJournalKeys> {
        self.record_cavity_trace(removed, created);
        for observer in &mut self.edge_observers {
            for index in removed
                .iter()
                .copied()
                .chain(created.iter().map(|(index, _)| *index))
            {
                observer.entry(index).or_insert_with(|| {
                    self.tetrahedra
                        .get(index as usize)
                        .and_then(Option::as_ref)
                        .map(|cell| cell.vertex_indices)
                });
            }
        }
        if self.trial_journals.is_empty() {
            return None;
        }
        let mut slots = removed.clone();
        slots.extend(created.iter().map(|(index, _)| *index));
        slots.extend(neighbors.iter().map(|(index, _, _)| *index));
        let cells = removed
            .iter()
            .map(|index| {
                self.tetrahedron(*index)
                    .expect("prepared cell")
                    .vertex_indices
            })
            .chain(created.iter().map(|(_, cell)| cell.vertex_indices))
            .collect::<Vec<_>>();
        let mut keys = BTreeSet::new();
        let mut faces = BTreeSet::new();
        let star_changes = removed
            .iter()
            .flat_map(|index| {
                self.tetrahedron(*index)
                    .expect("prepared cell")
                    .vertex_indices
                    .map(|vertex| (vertex, *index))
            })
            .chain(
                created
                    .iter()
                    .flat_map(|(index, cell)| cell.vertex_indices.map(|vertex| (vertex, *index))),
            )
            .collect::<BTreeSet<_>>();
        for cell in cells {
            let mut key = cell;
            key.sort_unstable();
            keys.insert(key);
            for opposite in 0..4 {
                faces.insert(Self::face(cell, opposite));
            }
        }
        for journal in &mut self.trial_journals {
            for index in &slots {
                journal
                    .cells
                    .entry(*index)
                    .or_insert_with(|| self.tetrahedra.get(*index as usize).cloned().flatten());
            }
            for key in &keys {
                journal
                    .cell_keys
                    .entry(*key)
                    .or_insert_with(|| self.tetrahedron_indices.get(key).copied());
            }
            for (vertex, cell) in &star_changes {
                journal
                    .stars
                    .entry((*vertex, *cell))
                    .or_insert_with(|| self.vertex_stars[*vertex as usize].contains(cell));
            }
            for face in &faces {
                journal
                    .faces
                    .entry(*face)
                    .or_insert_with(|| self.face_uses.get(face).cloned());
                journal
                    .boundary
                    .entry(*face)
                    .or_insert_with(|| self.boundary_faces.get(face).copied());
                if let Some(dirty) = &self.physical_dirty_faces {
                    journal
                        .dirty
                        .entry(*face)
                        .or_insert_with(|| dirty.contains(face));
                }
            }
            for index in removed
                .iter()
                .copied()
                .chain(created.iter().map(|(index, _)| *index))
            {
                journal
                    .free
                    .entry(index)
                    .or_insert_with(|| self.free_tetrahedra.contains(&index));
            }
        }
        Some(super::compaction::CavityJournalKeys {
            slots,
            keys,
            faces,
            stars: star_changes,
            nodes: BTreeSet::new(),
        })
    }
    pub(in crate::cdt::mutation) fn record_trial_dirty_discharge(&mut self) {
        if let Some(dirty) = &self.physical_dirty_faces {
            for journal in &mut self.trial_journals {
                for face in dirty {
                    journal.dirty.entry(*face).or_insert(true);
                }
            }
        }
    }
}
