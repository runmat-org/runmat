mod face_incidence;
use runmat_geometry_core::PersistentEntityId;

use runmat_meshing_core::quality::predicate::{PredicateSign, SpatialPredicateError};
use runmat_meshing_core::StableDigest;

use crate::cdt::{
    mutation::TopologyMutation,
    topology::{insphere_materialized_volume_nodes, orient_materialized_volume_nodes},
    DelaunayVolumeNode, DelaunayVolumeTopology,
};

#[derive(Clone, Copy)]
pub(in crate::cdt) struct TetrahedronRef<'a> {
    pub(in crate::cdt) vertex_indices: [u32; 4],
    pub(in crate::cdt) neighbors: [Option<u32>; 4],
    pub(in crate::cdt) region_id: &'a Option<PersistentEntityId>,
}

pub(in crate::cdt) trait TetrahedralTopology {
    fn nodes(&self) -> &[DelaunayVolumeNode];
    fn node_bounds(&self, vertex: u32) -> runmat_meshing_core::Aabb3 {
        let node = self.nodes()[vertex as usize];
        runmat_meshing_core::quality::predicate::constructed_point_bounds(
            runmat_meshing_core::quality::predicate::SpatialPredicatePoint {
                identity: node.identity,
                coordinates: node.coordinates_m.canonical_value(),
                construction: node.construction,
            },
        )
        .expect("canonical finite point construction")
    }
    fn tetrahedron_slot_count(&self) -> usize;
    fn tetrahedron(&self, index: u32) -> Option<TetrahedronRef<'_>>;
    fn vertex_star(&self, vertex: u32) -> Vec<u32>;
    fn has_edge(&self, edge: [u32; 2]) -> bool {
        let (first, second) = if self.vertex_star_len(edge[0]) <= self.vertex_star_len(edge[1]) {
            (edge[0], edge[1])
        } else {
            (edge[1], edge[0])
        };
        self.vertex_star(first).into_iter().any(|index| {
            self.tetrahedron(index)
                .expect("live vertex star")
                .vertex_indices
                .contains(&second)
        })
    }
    fn edge_tetrahedra(&self, edge: [u32; 2]) -> Vec<u32> {
        let (first, second) = if self.vertex_star_len(edge[0]) <= self.vertex_star_len(edge[1]) {
            (edge[0], edge[1])
        } else {
            (edge[1], edge[0])
        };
        self.vertex_star(first)
            .into_iter()
            .filter(|&index| {
                self.tetrahedron(index)
                    .expect("live vertex star")
                    .vertex_indices
                    .contains(&second)
            })
            .collect()
    }
    fn vertex_star_len(&self, vertex: u32) -> usize {
        self.vertex_star(vertex).len()
    }
    fn node_index(&self, identity: StableDigest) -> Option<u32>;
    fn tetrahedron_index(&self, mut vertices: [u32; 4]) -> Option<u32> {
        vertices.sort_unstable();
        self.vertex_star(vertices[0]).into_iter().find(|index| {
            let mut key = self
                .tetrahedron(*index)
                .expect("live vertex-star cell")
                .vertex_indices;
            key.sort_unstable();
            key == vertices
        })
    }
    fn face_tetrahedra_with_checkpoint<E>(
        &self,
        face: [u32; 3],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        face_incidence::scan(self, face, checkpoint)
    }

    fn prepare_tetrahedron_candidates<E>(
        &self,
        _: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        Ok(())
    }
    fn tetrahedron_candidates_in_bounds<E>(
        &self,
        bounds: runmat_meshing_core::Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        let mut result = Vec::new();
        for index in 0..self.tetrahedron_slot_count() {
            checkpoint()?;
            if self.tetrahedron(index as u32).is_some_and(|cell| {
                let points = cell
                    .vertex_indices
                    .into_iter()
                    .flat_map(|vertex| {
                        let bounds = self.node_bounds(vertex);
                        [bounds.min_m, bounds.max_m]
                    })
                    .collect::<Vec<_>>();
                runmat_meshing_core::Aabb3::from_points(&points)
                    .is_some_and(|cell_bounds| cell_bounds.intersects(bounds))
            }) {
                result.push(index as u32);
            }
        }
        Ok(result)
    }
    fn coordinate_scale_m(&self) -> f64 {
        self.nodes()
            .iter()
            .flat_map(|node| node.coordinates_m.canonical_value())
            .map(f64::abs)
            .fold(1.0, f64::max)
    }
    fn node_candidates_in_bounds<E>(
        &self,
        bounds: Option<runmat_meshing_core::Aabb3>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.node_candidates_in_region(bounds, &|_| true, checkpoint)
    }
    fn node_candidates_in_region<E>(
        &self,
        bounds: Option<runmat_meshing_core::Aabb3>,
        intersects: &impl Fn(runmat_meshing_core::Aabb3) -> bool,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        let mut candidates = Vec::new();
        for (index, node) in self.nodes().iter().enumerate() {
            checkpoint()?;
            if bounds
                .is_none_or(|bounds| bounds.contains_point(node.coordinates_m.canonical_value()))
                && intersects(runmat_meshing_core::Aabb3 {
                    min_m: node.coordinates_m.canonical_value(),
                    max_m: node.coordinates_m.canonical_value(),
                })
            {
                candidates.push(index as u32);
            }
        }
        Ok(candidates)
    }
    fn contains_position(&self, coordinates: [f64; 3]) -> bool {
        self.nodes()
            .iter()
            .any(|node| node.coordinates_m.canonical_value() == coordinates)
    }
    fn orient_nodes(
        &self,
        nodes: [DelaunayVolumeNode; 4],
    ) -> Result<PredicateSign, SpatialPredicateError> {
        orient_materialized_volume_nodes(nodes)
    }
    fn insphere_nodes(
        &self,
        nodes: [DelaunayVolumeNode; 5],
    ) -> Result<PredicateSign, SpatialPredicateError> {
        insphere_materialized_volume_nodes(nodes)
    }
}

impl TetrahedralTopology for DelaunayVolumeTopology {
    fn tetrahedron_index(&self, vertices: [u32; 4]) -> Option<u32> {
        crate::cdt::topology::queries::canonical_tetrahedron_index(self, vertices, &mut || {})
            .map(|index| index as u32)
    }
    fn vertex_star_len(&self, vertex: u32) -> usize {
        self.incidence
            .vertex_stars
            .get(vertex as usize)
            .map_or(0, Vec::len)
    }
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
    fn tetrahedron_index(&self, vertices: [u32; 4]) -> Option<u32> {
        TopologyMutation::tetrahedron_index(self, vertices)
    }
    fn face_tetrahedra_with_checkpoint<E>(
        &self,
        face: [u32; 3],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        face_incidence::maintained(self, face, checkpoint)
    }
    fn vertex_star_len(&self, vertex: u32) -> usize {
        self.vertex_stars
            .get(vertex as usize)
            .map_or(0, std::collections::BTreeSet::len)
    }
    fn prepare_tetrahedron_candidates<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        self.prepare_cell_spatial_index(checkpoint)
    }
    fn tetrahedron_candidates_in_bounds<E>(
        &self,
        bounds: runmat_meshing_core::Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.spatial_tetrahedron_candidates(bounds, checkpoint)
    }
    fn coordinate_scale_m(&self) -> f64 {
        self.node_coordinate_scale_m()
    }
    fn node_candidates_in_bounds<E>(
        &self,
        bounds: Option<runmat_meshing_core::Aabb3>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.spatial_node_candidates(bounds, checkpoint)
    }

    fn node_candidates_in_region<E>(
        &self,
        bounds: Option<runmat_meshing_core::Aabb3>,
        intersects: &impl Fn(runmat_meshing_core::Aabb3) -> bool,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<u32>, E> {
        self.spatial_node_candidates_in_region(bounds, intersects, checkpoint)
    }
    fn contains_position(&self, coordinates: [f64; 3]) -> bool {
        TopologyMutation::contains_position(self, coordinates)
    }
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

    fn has_edge(&self, edge: [u32; 2]) -> bool {
        !self.vertex_stars[edge[0] as usize].is_disjoint(&self.vertex_stars[edge[1] as usize])
    }
    fn edge_tetrahedra(&self, edge: [u32; 2]) -> Vec<u32> {
        self.vertex_stars[edge[0] as usize]
            .intersection(&self.vertex_stars[edge[1] as usize])
            .copied()
            .collect()
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

    fn orient_nodes(
        &self,
        nodes: [DelaunayVolumeNode; 4],
    ) -> Result<PredicateSign, SpatialPredicateError> {
        orient_materialized_volume_nodes(nodes)
    }

    fn insphere_nodes(
        &self,
        nodes: [DelaunayVolumeNode; 5],
    ) -> Result<PredicateSign, SpatialPredicateError> {
        insphere_materialized_volume_nodes(nodes)
    }
}
