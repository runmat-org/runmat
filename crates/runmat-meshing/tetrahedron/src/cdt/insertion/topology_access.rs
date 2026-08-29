use runmat_geometry_core::PersistentEntityId;

use runmat_meshing_core::StableDigest;

use crate::cdt::{mutation::TopologyMutation, DelaunayVolumeNode, DelaunayVolumeTopology};

#[derive(Clone, Copy)]
pub(in crate::cdt) struct TetrahedronRef<'a> {
    pub(in crate::cdt) vertex_indices: [u32; 4],
    pub(in crate::cdt) neighbors: [Option<u32>; 4],
    pub(in crate::cdt) region_id: &'a Option<PersistentEntityId>,
}

pub(in crate::cdt) trait TetrahedralTopology {
    fn nodes(&self) -> &[DelaunayVolumeNode];
    fn tetrahedron_slot_count(&self) -> usize;
    fn tetrahedron(&self, index: u32) -> Option<TetrahedronRef<'_>>;
    fn vertex_star(&self, vertex: u32) -> Vec<u32>;
    fn node_index(&self, identity: StableDigest) -> Option<u32>;
}

impl TetrahedralTopology for DelaunayVolumeTopology {
    fn nodes(&self) -> &[DelaunayVolumeNode] {
        &self.nodes
    }

    fn tetrahedron_slot_count(&self) -> usize {
        self.tetrahedra.len()
    }

    fn tetrahedron(&self, index: u32) -> Option<TetrahedronRef<'_>> {
        let tetrahedron = self.tetrahedra.get(index as usize)?;
        Some(TetrahedronRef {
            vertex_indices: tetrahedron.vertex_indices,
            neighbors: tetrahedron.neighbors,
            region_id: &tetrahedron.region_id,
        })
    }

    fn vertex_star(&self, vertex: u32) -> Vec<u32> {
        self.incidence
            .vertex_stars
            .get(vertex as usize)
            .cloned()
            .unwrap_or_default()
    }

    fn node_index(&self, identity: StableDigest) -> Option<u32> {
        self.nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .ok()
            .and_then(|index| u32::try_from(index).ok())
    }
}

impl TetrahedralTopology for TopologyMutation {
    fn nodes(&self) -> &[DelaunayVolumeNode] {
        &self.nodes
    }

    fn tetrahedron_slot_count(&self) -> usize {
        self.tetrahedra.len()
    }

    fn tetrahedron(&self, index: u32) -> Option<TetrahedronRef<'_>> {
        let tetrahedron = self.tetrahedron(index)?;
        Some(TetrahedronRef {
            vertex_indices: tetrahedron.vertex_indices,
            neighbors: tetrahedron.neighbors,
            region_id: &tetrahedron.region_id,
        })
    }

    fn vertex_star(&self, vertex: u32) -> Vec<u32> {
        self.vertex_stars
            .get(vertex as usize)
            .map(|star| star.iter().copied().collect())
            .unwrap_or_default()
    }

    fn node_index(&self, identity: StableDigest) -> Option<u32> {
        self.node_indices.get(&identity).copied()
    }
}
