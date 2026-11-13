use std::collections::BTreeSet;

use runmat_meshing_core::{quality::predicate::PredicateSign, Aabb3};

use super::{
    point_in_tetrahedron, predicate_error, topology_access::TetrahedralTopology,
    DelaunayInsertionError, DelaunayVolumeNode, InsertionWork,
};

#[cfg(test)]
mod tests;

/// Deterministic point location for one immutable topology.
///
/// Adjacency walking is the primary path. Exact orientation predicates decide
/// every crossing and containment result. An optional grid is only used when a
/// walk reaches a disconnected component or cycles on a degenerate complex.
pub(super) struct TetrahedronPointLocator<'topology, T> {
    topology: &'topology T,
}

pub(super) enum PointLocation {
    Contained(u32),
    Outside { tetrahedron: u32, opposite: usize },
    Unresolved,
}

impl<'topology, T: TetrahedralTopology> TetrahedronPointLocator<'topology, T> {
    pub(super) fn walking(topology: &'topology T) -> Self {
        Self { topology }
    }

    pub(super) fn first_containing_from(
        &self,
        node: DelaunayVolumeNode,
        hint: Option<usize>,
        work: &mut InsertionWork<'_>,
    ) -> Result<Option<u32>, DelaunayInsertionError> {
        if let Some(start) = hint {
            if let Some(containing) = self.walk_containing(start, node, work)? {
                return Ok(Some(containing));
            }
        }

        // Mutable topology supplies maintained cell bounds. Immutable owners
        // retain their complete conservative candidate traversal. Neither the
        // index nor a failed walk substitutes for exact physical containment.
        let bounds = Aabb3 {
            min_m: node.coordinates_m.canonical_value().map(f64::next_down),
            max_m: node.coordinates_m.canonical_value().map(f64::next_up),
        };
        for tetrahedron_index in self
            .topology
            .tetrahedron_candidates_in_bounds(bounds, &mut || work.checkpoint())?
        {
            let Some(tetrahedron) = self.topology.tetrahedron(tetrahedron_index) else {
                continue;
            };
            work.checkpoint()?;
            if expanded_tetrahedron_bounds(self.topology, tetrahedron.vertex_indices)
                .contains_point(node.coordinates_m.canonical_value())
                && point_in_tetrahedron(self.topology, tetrahedron.vertex_indices, node, work)?
            {
                return Ok(Some(tetrahedron_index));
            }
        }
        Ok(None)
    }

    fn walk_containing(
        &self,
        start: usize,
        node: DelaunayVolumeNode,
        work: &mut InsertionWork<'_>,
    ) -> Result<Option<u32>, DelaunayInsertionError> {
        Ok(match self.walk_from(node, Some(start), work)? {
            PointLocation::Contained(index) => Some(index),
            _ => None,
        })
    }

    pub(super) fn walk_from(
        &self,
        node: DelaunayVolumeNode,
        hint: Option<usize>,
        work: &mut InsertionWork<'_>,
    ) -> Result<PointLocation, DelaunayInsertionError> {
        let Some(start) = hint.or_else(|| {
            (0..self.topology.tetrahedron_slot_count())
                .find(|index| self.topology.tetrahedron(*index as u32).is_some())
        }) else {
            return Ok(PointLocation::Unresolved);
        };
        if start >= self.topology.tetrahedron_slot_count()
            || self.topology.tetrahedron(start as u32).is_none()
        {
            return Ok(PointLocation::Unresolved);
        }
        let mut visited = BTreeSet::new();
        let mut current = start;
        while visited.insert(current) {
            work.checkpoint()?;
            let tetrahedron = self.topology.tetrahedron(current as u32).ok_or_else(|| {
                super::error(
                    super::DelaunayInsertionErrorKind::InvalidTopology,
                    "point-location walk reached an inactive tetrahedron",
                )
            })?;
            let mut crossing = None;
            for opposite in 0..4 {
                work.predicate()?;
                let mut points = tetrahedron
                    .vertex_indices
                    .map(|vertex| self.topology.nodes()[vertex as usize]);
                points[opposite] = node;
                if self
                    .topology
                    .orient_nodes(points)
                    .map_err(predicate_error)?
                    == PredicateSign::Negative
                {
                    crossing = Some(opposite);
                    break;
                }
            }
            let Some(opposite) = crossing else {
                return Ok(PointLocation::Contained(current as u32));
            };
            let Some(neighbor) = tetrahedron.neighbors[opposite] else {
                return Ok(PointLocation::Outside {
                    tetrahedron: current as u32,
                    opposite,
                });
            };
            current = neighbor as usize;
        }
        Ok(PointLocation::Unresolved)
    }
}

fn expanded_tetrahedron_bounds(topology: &impl TetrahedralTopology, vertices: [u32; 4]) -> Aabb3 {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        let coordinates = topology.nodes()[vertex as usize]
            .coordinates_m
            .canonical_value();
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(coordinates[axis]);
            maximum[axis] = maximum[axis].max(coordinates[axis]);
        }
    }
    Aabb3 {
        min_m: minimum.map(f64::next_down),
        max_m: maximum.map(f64::next_up),
    }
}
