//! Lazy node broad phase for recovery. Canonical coordinates and predicates remain
//! authoritative; candidate order preserves the original node-inventory order.
use super::TopologyMutation;
use crate::cdt::DelaunayVolumeTopology;
use runmat_meshing_core::{Aabb3, PointSpatialIndex, SpatialPoint, StableDigest};

impl TopologyMutation {
    pub(in crate::cdt) fn prepare_point_spatial_index<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let mut retained = self.spatial_index.borrow_mut();
        if retained.is_none() {
            let mut entries = Vec::with_capacity(self.nodes.len());
            for node in &self.nodes {
                checkpoint()?;
                entries.push(entry(*node));
            }
            *retained = Some(std::sync::Arc::new(
                PointSpatialIndex::from_entries_with_checkpoint(entries, checkpoint)?,
            ));
        }
        synchronize_point_index(
            retained.as_mut().expect("prepared point index"),
            &self.nodes,
            checkpoint,
        )
    }
    pub(super) fn point_index_after_replacement<E>(
        &self,
        slot: u32,
        replacement: crate::cdt::DelaunayVolumeNode,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Option<std::sync::Arc<PointSpatialIndex<StableDigest>>>, E> {
        let retained = self.spatial_index.borrow();
        let Some(index) = retained.as_ref() else {
            return Ok(None);
        };
        if slot as usize >= index.len() {
            return Ok(Some(index.clone()));
        }
        let mut next = index.clone_with_checkpoint(checkpoint)?;
        assert!(
            next.replace_with_checkpoint(slot as usize, entry(replacement), checkpoint)?,
            "replacement belongs to retained point prefix"
        );
        Ok(Some(std::sync::Arc::new(next)))
    }
    pub(in crate::cdt) fn node_coordinate_scale_m(&self) -> f64 {
        self.coordinate_scale_m
    }
    pub(in crate::cdt) fn spatial_node_candidates<E>(
        &self,
        bounds: Option<Aabb3>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.spatial_node_candidates_in_region(bounds, &|_| true, checkpoint)
    }
    pub(in crate::cdt) fn spatial_node_candidates_in_region<E>(
        &self,
        bounds: Option<Aabb3>,
        intersects: &impl Fn(Aabb3) -> bool,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        let Some(bounds) = bounds.filter(|bounds| {
            bounds
                .min_m
                .into_iter()
                .chain(bounds.max_m)
                .all(f64::is_finite)
        }) else {
            let mut result = Vec::with_capacity(self.nodes.len());
            for index in 0..self.nodes.len() {
                checkpoint()?;
                result.push(index as u32);
            }
            return Ok(result);
        };
        self.prepare_point_spatial_index(checkpoint)?;
        let mut retained = self.spatial_index.borrow_mut();
        let index = retained.as_mut().expect("spatial index was prepared");
        // Nodes are append-only. A cancelled append preserves the index and the
        // next query resumes synchronization from its complete retained length.
        let mut candidates = Vec::new();
        index.visit_region_with_checkpoint(
            bounds,
            intersects,
            &mut |entry| {
                candidates.push(self.node_indices[&entry.payload]);
            },
            checkpoint,
        )?;
        // A retained index may have been built before canonical node reordering.
        // Preserve the original authoritative inventory order at every query.
        candidates.sort_unstable();
        Ok(candidates)
    }

    /// Canonical admission can reorder node slots without changing their geometry.
    /// Retain a complete or append-only prefix broad phase after checking original
    /// stable identities and coordinates against the admitted inventory. Cancelled
    /// synchronization preserves topology and a valid original point prefix. Changed inventories
    /// safely take the ordinary lazy rebuild path.
    pub(in crate::cdt) fn adopt_canonical_with_spatial_index<E>(
        &mut self,
        canonical: DelaunayVolumeTopology,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        assert!(
            self.trial_journals.is_empty() && self.edge_observers.is_empty(),
            "whole-owner adoption outside local trials"
        );
        checkpoint()?;
        let mut next = Self::from_canonical(canonical);
        let retained = self.spatial_index.borrow();
        let mut reusable = retained.as_ref().is_some_and(|index| {
            index.len() <= self.nodes.len() && self.nodes.len() == next.nodes.len()
        });
        if reusable {
            for node in &self.nodes {
                checkpoint()?;
                if !next.node_indices.get(&node.identity).is_some_and(|index| {
                    next.nodes[*index as usize].coordinates_m.canonical_value()
                        == node.coordinates_m.canonical_value()
                        && next.nodes[*index as usize].construction == node.construction
                }) {
                    reusable = false;
                    break;
                }
            }
        }
        drop(retained);
        self.rebase_cell_index(&mut next, checkpoint)?;
        if reusable {
            // Complete the original append-only prefix before canonical slots
            // are reordered. Each successful cache append still describes this
            // original topology if later synchronization is cancelled.
            let mut retained = self.spatial_index.borrow_mut();
            synchronize_point_index(
                retained.as_mut().expect("retained original prefix"),
                &self.nodes,
                checkpoint,
            )?;
            drop(retained);
            next.spatial_index = std::cell::RefCell::new(self.spatial_index.borrow_mut().take());
        }
        *self = next;
        Ok(())
    }
}

fn entry(node: crate::cdt::DelaunayVolumeNode) -> SpatialPoint<StableDigest> {
    let bounds = runmat_meshing_core::quality::predicate::constructed_point_bounds(
        runmat_meshing_core::quality::predicate::SpatialPredicatePoint {
            identity: node.identity,
            coordinates: node.coordinates_m.canonical_value(),
            construction: node.construction,
        },
    )
    .expect("canonical finite point construction");
    SpatialPoint::with_bounds(node.coordinates_m.canonical_value(), bounds, node.identity)
        .expect("canonical construction enclosure")
}

fn synchronize_point_index<E>(
    index: &mut std::sync::Arc<PointSpatialIndex<StableDigest>>,
    nodes: &[crate::cdt::DelaunayVolumeNode],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<(), E> {
    while index.len() < nodes.len() {
        let next = index.len();
        // Cavity trials share immutable geometry. Only a trial that actually
        // appends points needs independent index storage for its transaction.
        if std::sync::Arc::get_mut(index).is_none() {
            let copied = index.clone_with_checkpoint(checkpoint)?;
            *index = std::sync::Arc::new(copied);
        }
        std::sync::Arc::get_mut(index)
            .expect("detached point index")
            .insert_with_checkpoint(entry(nodes[next]), checkpoint)?;
    }
    Ok(())
}
