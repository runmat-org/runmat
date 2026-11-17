//! Heap bytes the working topology holds, for the run's memory ceiling.
//!
//! The inventory counts every container this owner retains: vector capacities, B-tree entries
//! at the standard library's minimum node fill, the point and cell spatial indexes, the edit trace
//! and open trial journals' vectors. Shared region identities belong to the boundary model and
//! are not counted here.
use super::storage::{MutableTetrahedron, TopologyMutation};
use crate::cdt::DelaunayVolumeNode;
use runmat_meshing_core::resident_bytes::{btree_bytes, vec_bytes};
use runmat_meshing_core::StableDigest;
use std::collections::{BTreeMap, BTreeSet};

impl TopologyMutation {
    /// An upper bound on the heap bytes this topology holds now. The walk visits every vertex
    /// star and face record once.
    pub(in crate::cdt) fn resident_bytes(&self) -> u64 {
        let stars = self.vertex_stars.iter().fold(
            vec_bytes::<BTreeSet<u32>>(self.vertex_stars.capacity()),
            |sum, star| sum.saturating_add(btree_bytes::<u32, ()>(star.len())),
        );
        let faces = self.face_uses.values().fold(
            btree_bytes::<[u32; 3], Vec<(u32, usize)>>(self.face_uses.len()),
            |sum, uses| sum.saturating_add(vec_bytes::<(u32, usize)>(uses.capacity())),
        );
        let observers = self.edge_observers.iter().fold(
            vec_bytes::<BTreeMap<u32, Option<[u32; 4]>>>(self.edge_observers.capacity()),
            |sum, observer| {
                sum.saturating_add(btree_bytes::<u32, Option<[u32; 4]>>(observer.len()))
            },
        );
        [
            vec_bytes::<DelaunayVolumeNode>(self.nodes.capacity()),
            btree_bytes::<StableDigest, u32>(self.node_indices.len()),
            btree_bytes::<[u64; 3], ()>(self.positions.len()),
            self.spatial_index
                .borrow()
                .as_deref()
                .map_or(0, |index| index.resident_bytes()),
            self.cell_spatial_index
                .borrow()
                .as_ref()
                .map_or(0, |cache| cache.resident_bytes()),
            self.physical_dirty_faces
                .as_ref()
                .map_or(0, |faces| btree_bytes::<[u32; 3], ()>(faces.len())),
            vec_bytes::<Option<MutableTetrahedron>>(self.tetrahedra.capacity()),
            btree_bytes::<[u32; 4], u32>(self.tetrahedron_indices.len()),
            btree_bytes::<u32, ()>(self.free_tetrahedra.len()),
            stars,
            faces,
            btree_bytes::<[u32; 3], (u32, usize)>(self.boundary_faces.len()),
            self.edit_trace.resident_bytes(),
            self.trial_journals.iter().fold(
                vec_bytes::<super::trial::TrialJournal>(self.trial_journals.capacity()),
                |sum, journal| sum.saturating_add(journal.resident_bytes()),
            ),
            observers,
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
    }
}
