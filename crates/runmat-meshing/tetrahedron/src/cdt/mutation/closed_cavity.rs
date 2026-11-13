//! Current-contract admission for a closed, same-region local replacement.
//! The admitted owner's unchanged cells retain their authority. Every changed
//! interface and protected facet is checked before the existing atomic commit.
use super::storage::{
    error, execution_error, TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
use crate::cdt::DelaunayTopologyOptions;
use crate::reconnect::local_tetrahedron_boundary_faces;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use std::collections::BTreeSet;

impl TopologyMutation {
    pub(in crate::cdt) fn replace_closed_cavity(
        &mut self,
        removed: &BTreeSet<u32>,
        created: Vec<([u32; 4], Option<PersistentEntityId>)>,
        protected: &BTreeSet<[StableDigest; 3]>,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u32>, TopologyMutationError> {
        checkpoint(cancellation)?;
        if options.maximum_nodes == 0
            || options.maximum_tetrahedra == 0
            || options.cancellation_check_interval == 0
        {
            return Err(error("closed cavity topology controls must be nonzero"));
        }
        let count = self
            .tetrahedron_indices
            .len()
            .checked_sub(removed.len())
            .and_then(|count| count.checked_add(created.len()));
        if self.nodes.len() as u64 > options.maximum_nodes
            || count.is_none_or(|count| count as u64 > options.maximum_tetrahedra)
        {
            return Err(execution_error(
                TopologyMutationErrorKind::ResourceLimit,
                "closed cavity exceeds current topology inventory limits",
            ));
        }
        let mut old = Vec::with_capacity(removed.len());
        let mut region = None;
        for (position, index) in removed.iter().copied().enumerate() {
            if (position as u64).is_multiple_of(options.cancellation_check_interval) {
                checkpoint(cancellation)?;
            }
            let cell = self
                .tetrahedron(index)
                .ok_or_else(|| error("closed cavity names an inactive cell"))?;
            if let Some(expected) = &region {
                if &cell.region_id != expected {
                    return Err(error("closed cavity crosses a region boundary"));
                }
            } else {
                region = Some(cell.region_id.clone());
            }
            old.push(cell.vertex_indices);
        }
        let region = region.ok_or_else(|| error("closed cavity cannot be empty"))?;
        if created.iter().any(|(_, candidate)| *candidate != region) {
            return Err(error("closed cavity reassigns its region"));
        }
        let plan = self.prepare_cavity_replacement(removed, created, &BTreeSet::new())?;
        let mut new = Vec::with_capacity(plan.tetrahedra.len());
        let mut new_faces = BTreeSet::new();
        for (position, (_, cell)) in plan.tetrahedra.iter().enumerate() {
            if (position as u64).is_multiple_of(options.cancellation_check_interval) {
                checkpoint(cancellation)?;
            }
            // Bounds, exact orientation, duplicates and reciprocal face sides
            // have passed the existing shared planner, once for each new cell.
            let vertices = cell.vertex_indices;
            for opposite in 0..4 {
                let face = Self::face(vertices, opposite)
                    .map(|vertex| self.nodes[vertex as usize].identity);
                let mut face = face;
                face.sort_unstable();
                new_faces.insert(face);
            }
            new.push(vertices);
        }
        let old_vertices = old.iter().flatten().copied().collect::<BTreeSet<_>>();
        let new_vertices = new.iter().flatten().copied().collect::<BTreeSet<_>>();
        if !old_vertices.is_subset(&new_vertices) {
            return Err(error(
                "closed cavity removes an existing vertex without deletion authority",
            ));
        }
        if local_tetrahedron_boundary_faces(&old) != local_tetrahedron_boundary_faces(&new) {
            return Err(error("closed cavity changes its exact boundary"));
        }
        for vertices in old {
            for opposite in 0..4 {
                let mut face = Self::face(vertices, opposite)
                    .map(|vertex| self.nodes[vertex as usize].identity);
                face.sort_unstable();
                if protected.contains(&face) && !new_faces.contains(&face) {
                    return Err(error("closed cavity removes a protected facet"));
                }
            }
        }
        checkpoint(cancellation)?;
        let created = plan.created.clone();
        self.commit_cavity_replacement(plan);
        Ok(created)
    }
}

fn checkpoint(cancellation: &dyn MeshingCancellationSignal) -> Result<(), TopologyMutationError> {
    if cancellation.is_cancelled() {
        return Err(execution_error(
            TopologyMutationErrorKind::Cancelled,
            "closed cavity admission was cancelled",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
