//! Borrowed canonical topology with the admitted coordinate inventory indexed.
//! Node/cell authority stays in the original topology; no mesh payload is copied.
use super::{DelaunayVolumeNode, DelaunayVolumeTopology};
use crate::cdt::insertion::topology_access::{TetrahedralTopology, TetrahedronRef};
use runmat_meshing_core::{Aabb3, PointSpatialIndex, StableDigest};

pub(in crate::cdt) struct IndexedCanonicalTopology<'a> {
    pub(in crate::cdt) topology: &'a DelaunayVolumeTopology,
    pub(in crate::cdt) locations: &'a PointSpatialIndex<()>,
}
impl TetrahedralTopology for IndexedCanonicalTopology<'_> {
    fn nodes(&self) -> &[DelaunayVolumeNode] {
        &self.topology.nodes
    }
    fn tetrahedron_slot_count(&self) -> usize {
        self.topology.tetrahedron_slot_count()
    }
    fn tetrahedron(&self, index: u32) -> Option<TetrahedronRef<'_>> {
        self.topology.tetrahedron(index)
    }
    fn vertex_star(&self, vertex: u32) -> Vec<u32> {
        self.topology.vertex_star(vertex)
    }
    fn vertex_star_len(&self, vertex: u32) -> usize {
        self.topology.vertex_star_len(vertex)
    }
    fn node_index(&self, identity: StableDigest) -> Option<u32> {
        self.topology.node_index(identity)
    }
    fn tetrahedron_index(&self, vertices: [u32; 4]) -> Option<u32> {
        self.topology.tetrahedron_index(vertices)
    }
    fn contains_position(&self, coordinates: [f64; 3]) -> bool {
        let mut found = false;
        let result: Result<(), std::convert::Infallible> =
            self.locations.visit_bounds_with_checkpoint(
                Aabb3 {
                    min_m: coordinates,
                    max_m: coordinates,
                },
                &mut |point| {
                    found |= point.position() == coordinates;
                },
                &mut || Ok(()),
            );
        match result {
            Ok(()) => found,
            Err(never) => match never {},
        }
    }
}
