//! Physical cell bounds use stable corners across canonical slot reordering.
//! Persistent branches are shared by cavity trials; a cancellable change journal
//! updates only replaced cells. The index never supplies geometric admission.
use super::TopologyMutation;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use runmat_meshing_core::{Aabb3, BoundsSpatialIndex, SpatialEntry, StableDigest};
use std::collections::BTreeMap;

type CellKey = [StableDigest; 4];
type Entry = SpatialEntry<CellKey>;
mod pending;
use pending::PendingChanges;
mod checkpoint;
pub(super) use checkpoint::CellSpatialCheckpoint;
#[derive(Clone, Debug)]
pub(super) struct CellSpatialCache {
    index: BoundsSpatialIndex<CellKey>,
    pending: PendingChanges,
}

impl CellSpatialCache {
    // Queries observe only the current topology, never intermediate cells that
    // were created and removed before a query. Retain the first indexed bounds
    // and final live bounds per stable corner key; exact net-zero changes vanish.
    fn record(&mut self, entry: Entry, present: bool) {
        let next = match self.pending.get(&entry.payload) {
            None => Some(if present {
                (None, Some(entry.bounds))
            } else {
                (Some(entry.bounds), None)
            }),
            Some((original, _)) => {
                let current = present.then_some(entry.bounds);
                (!pending::bounds_equal(*original, current)).then_some((*original, current))
            }
        };
        self.pending.set(entry.payload, next);
    }
}

impl TopologyMutation {
    fn cell_entry(&self, index: u32) -> Entry {
        let cell = self.tetrahedron(index).expect("live indexed cell");
        let mut payload = cell
            .vertex_indices
            .map(|vertex| self.nodes[vertex as usize].identity);
        payload.sort_unstable();
        let points = cell
            .vertex_indices
            .into_iter()
            .flat_map(|vertex| {
                let bounds = self.node_bounds(vertex);
                [bounds.min_m, bounds.max_m]
            })
            .collect::<Vec<_>>();
        Entry {
            bounds: Aabb3::from_points(&points).expect("canonical finite cell bounds"),
            payload,
        }
    }

    pub(super) fn record_cell_removal(&mut self, index: u32) {
        if self.cell_spatial_index.get_mut().is_some() {
            let entry = self.cell_entry(index);
            self.cell_spatial_index
                .get_mut()
                .as_mut()
                .expect("prepared cell index")
                .record(entry, false);
        }
    }
    pub(super) fn record_cell_insertion(&mut self, index: u32) {
        if self.cell_spatial_index.get_mut().is_some() {
            let entry = self.cell_entry(index);
            self.cell_spatial_index
                .get_mut()
                .as_mut()
                .expect("prepared cell index")
                .record(entry, true);
        }
    }

    pub(in crate::cdt) fn spatial_tetrahedron_candidates<E>(
        &self,
        bounds: Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.prepare_cell_spatial_index(checkpoint)?;
        let retained = self.cell_spatial_index.borrow();
        let cache = retained.as_ref().expect("prepared cell index");
        let mut result = Vec::new();
        cache.index.visit_bounds_with_checkpoint(
            bounds,
            &mut |entry| {
                if let Some(index) = self.slot_for_key(entry.payload) {
                    result.push(index);
                }
            },
            checkpoint,
        )?;
        result.sort_unstable();
        result.dedup();
        Ok(result)
    }

    pub(in crate::cdt) fn prepare_cell_spatial_index<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let mut retained = self.cell_spatial_index.borrow_mut();
        if retained.is_none() {
            let mut entries = Vec::with_capacity(self.active_tetrahedron_count());
            for (index, cell) in self.tetrahedra.iter().enumerate() {
                checkpoint()?;
                if cell.is_some() {
                    entries.push(self.cell_entry(index as u32));
                }
            }
            let index = BoundsSpatialIndex::from_entries_with_checkpoint(entries, checkpoint)?
                .expect("canonical finite cell bounds");
            *retained = Some(CellSpatialCache {
                index,
                pending: PendingChanges::default(),
            });
        }
        let cache = retained.as_mut().expect("prepared cell index");
        while let Some((key, (removed, created))) = cache.pending.first_key_value() {
            // The root and journal advance together. A rejected change leaves a
            // complete retained prefix; no query sees a partially updated root.
            let mut next = cache.index.clone();
            if let Some(entry) = removed {
                assert!(
                    next.remove_with_checkpoint(
                        &Entry {
                            bounds: *entry,
                            payload: *key
                        },
                        checkpoint
                    )?,
                    "retained original cell bounds"
                );
            }
            if let Some(entry) = created {
                assert!(
                    next.insert_with_checkpoint(
                        Entry {
                            bounds: *entry,
                            payload: *key
                        },
                        checkpoint
                    )?,
                    "canonical finite cell bounds"
                );
            }
            checkpoint()?;
            cache.index = next;
            cache.pending.remove_first();
        }
        Ok(())
    }

    fn slot_for_key(&self, key: CellKey) -> Option<u32> {
        let mut vertices = [0; 4];
        for (slot, identity) in key.into_iter().enumerate() {
            vertices[slot] = *self.node_indices.get(&identity)?;
        }
        vertices.sort_unstable();
        self.tetrahedron_indices.get(&vertices).copied()
    }

    pub(super) fn rebase_cell_index<E>(
        &self,
        next: &mut Self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let retained = self.cell_spatial_index.borrow();
        let Some(cache) = retained.as_ref() else {
            return Ok(());
        };
        // Bounds can be reused only under the original physical point inventory.
        for node in &self.nodes {
            checkpoint()?;
            if !next.node_indices.get(&node.identity).is_some_and(|index| {
                next.nodes[*index as usize].coordinates_m.canonical_value()
                    == node.coordinates_m.canonical_value()
                    && next.nodes[*index as usize].construction == node.construction
            }) {
                return Ok(());
            }
        }
        let mut cache = cache.clone();
        for (index, cell) in self.tetrahedra.iter().enumerate() {
            checkpoint()?;
            if cell.is_some() {
                let entry = self.cell_entry(index as u32);
                if next.slot_for_key(entry.payload).is_none() {
                    cache.record(entry, false);
                }
            }
        }
        for (index, cell) in next.tetrahedra.iter().enumerate() {
            checkpoint()?;
            if cell.is_some() {
                let entry = next.cell_entry(index as u32);
                if self.slot_for_key(entry.payload).is_none() {
                    cache.record(entry, true);
                }
            }
        }
        next.cell_spatial_index = std::cell::RefCell::new(Some(cache));
        Ok(())
    }
}

#[cfg(test)]
mod tests;
