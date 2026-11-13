//! Atomic same-connectivity relocation. Exact materialized orientation is
//! checked on the complete affected star before point/index state is changed.
use super::storage::{
    coordinate_bits, error, execution_error, TopologyMutation, TopologyMutationError,
    TopologyMutationErrorKind,
};
use super::trace::TopologyEdit;
use crate::cdt::{
    topology::orient_materialized_volume_nodes, DelaunayTopologyOptions, DelaunayVolumeNode,
};
use runmat_meshing_core::{
    quality::predicate::{constructed_point_bounds, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};
use std::collections::BTreeSet;

pub(super) struct PointMove<'a> {
    pub(super) index: u32,
    pub(super) replacement: DelaunayVolumeNode,
    pub(super) replaced_cells: Option<&'a BTreeSet<u32>>,
}

impl TopologyMutation {
    pub(in crate::cdt) fn replace_node(
        &mut self,
        index: u32,
        replacement: DelaunayVolumeNode,
        protected: &BTreeSet<StableDigest>,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u32>, TopologyMutationError> {
        self.move_point(
            PointMove {
                index,
                replacement,
                replaced_cells: None,
            },
            protected,
            options,
            cancellation,
        )
    }
    pub(super) fn move_point(
        &mut self,
        change: PointMove<'_>,
        protected: &BTreeSet<StableDigest>,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u32>, TopologyMutationError> {
        let PointMove {
            index,
            replacement,
            replaced_cells,
        } = change;
        checkpoint(cancellation)?;
        let old = self
            .nodes
            .get(index as usize)
            .copied()
            .ok_or_else(|| error("relocation node is absent"))?;
        if options.cancellation_check_interval == 0 {
            return Err(error("relocation polling interval must be nonzero"));
        }
        if protected.contains(&old.identity) {
            return Err(error("relocation changes a protected node"));
        }
        if replacement.identity == StableDigest::ZERO
            || self
                .node_indices
                .get(&replacement.identity)
                .is_some_and(|existing| *existing != index)
            || (replacement.coordinates_m.canonical_value() != old.coordinates_m.canonical_value()
                && self.contains_position(replacement.coordinates_m.canonical_value()))
        {
            return Err(error("relocation node identity or position collides"));
        }
        constructed_point_bounds(replacement.predicate_point()).map_err(|failure| {
            error(format!(
                "relocation point construction is invalid: {failure:?}"
            ))
        })?;
        if self.nodes.len() as u64 > options.maximum_nodes
            || self.tetrahedron_indices.len() as u64 > options.maximum_tetrahedra
        {
            return Err(execution_error(
                TopologyMutationErrorKind::ResourceLimit,
                "relocation exceeds current topology inventory limits",
            ));
        }
        let cells = self.vertex_stars[index as usize]
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if cells.is_empty() {
            return Err(error("relocation requires a complete nonempty vertex star"));
        }
        if replaced_cells.is_some_and(|removed| !cells.iter().all(|cell| removed.contains(cell))) {
            return Err(error(
                "joint relocation must replace the complete point star",
            ));
        }
        for (position, cell) in cells.iter().copied().enumerate() {
            if replaced_cells.is_some() {
                continue;
            }

            if (position as u64).is_multiple_of(options.cancellation_check_interval) {
                checkpoint(cancellation)?;
            }
            let nodes = self
                .tetrahedron(cell)
                .expect("live point star")
                .vertex_indices
                .map(|vertex| {
                    if vertex == index {
                        replacement
                    } else {
                        self.nodes[vertex as usize]
                    }
                });
            if orient_materialized_volume_nodes(nodes)
                .map_err(|failure| error(format!("relocation orientation failed: {failure:?}")))?
                != PredicateSign::Positive
            {
                return Err(error("relocation degenerates or inverts its vertex star"));
            }
        }
        let point_index = self
            .point_index_after_replacement(index, replacement, &mut || checkpoint(cancellation))?;
        checkpoint(cancellation)?;
        let affected = if self.trace_enabled && !self.trial_journals.is_empty() {
            cells
                .iter()
                .map(|cell| {
                    self.tetrahedron(*cell)
                        .expect("live point star")
                        .vertex_indices
                        .map(|vertex| self.nodes[vertex as usize].identity)
                })
                .collect()
        } else {
            Vec::new()
        };
        self.record_trial_point(index, &cells);
        for cell in &cells {
            self.record_cell_removal(*cell);
        }
        self.node_indices.remove(&old.identity);
        self.positions
            .remove(&old.coordinates_m.canonical_value().map(coordinate_bits));
        self.nodes[index as usize] = replacement;
        self.node_indices.insert(replacement.identity, index);
        self.positions.insert(
            replacement
                .coordinates_m
                .canonical_value()
                .map(coordinate_bits),
        );
        self.coordinate_scale_m = replacement
            .coordinates_m
            .canonical_value()
            .into_iter()
            .map(f64::abs)
            .fold(self.coordinate_scale_m, f64::max);
        *self.spatial_index.get_mut() = point_index;
        *self.connectivity_state.get_mut() = None;
        for cell in &cells {
            self.record_cell_insertion(*cell);
            self.record_physical_cell_change(
                self.tetrahedron(*cell)
                    .expect("live point star")
                    .vertex_indices,
            );
        }
        if self.trace_enabled && !self.trial_journals.is_empty() {
            self.edit_trace.push(TopologyEdit::ReplaceNode {
                source_identity: old.identity,
                replacement,
                affected,
            });
        }
        Ok(cells)
    }
}
fn checkpoint(cancellation: &dyn MeshingCancellationSignal) -> Result<(), TopologyMutationError> {
    if cancellation.is_cancelled() {
        return Err(execution_error(
            TopologyMutationErrorKind::Cancelled,
            "local relocation was cancelled",
        ));
    }
    Ok(())
}
