//! Unconstrained hull growth through checked local mutations, frozen once at the end.
use super::{
    error, mutation_topology_error, predicate_error, resource,
    topology_access::TetrahedralTopology, validate_new_node, validate_options,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology, InsertionWork, TopologyMutation,
};
use runmat_meshing_core::{quality::predicate::PredicateSign, MeshingCancellationSignal};
use std::collections::BTreeSet;
mod search;

pub(in crate::cdt) struct UnconstrainedHull {
    topology: TopologyMutation,
    hint: Option<u32>,
}

impl UnconstrainedHull {
    pub(in crate::cdt) fn new(
        topology: DelaunayVolumeTopology,
    ) -> Result<Self, DelaunayInsertionError> {
        if topology
            .tetrahedra
            .iter()
            .any(|tet| tet.region_id.is_some())
        {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "unconstrained hull growth cannot discard assigned region provenance",
            ));
        }
        Ok(Self {
            topology: TopologyMutation::from_canonical(topology),
            hint: Some(0),
        })
    }

    pub(in crate::cdt) fn insert(
        &mut self,
        node: DelaunayVolumeNode,
        options: DelaunayInsertionOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<(), DelaunayInsertionError> {
        validate_options(options)?;
        validate_new_node(&self.topology, node)?;
        let mut work = InsertionWork::new(options, cancellation);
        let location = super::location::TetrahedronPointLocator::walking(&self.topology)
            .walk_from(node, self.hint.map(|index| index as usize), &mut work)?;
        // A point on the hull boundary is in conflict with the unbounded cells over the
        // coplanar hull faces whose circumcircle holds it; only the hull search finds them.
        let on_hull = match location {
            super::location::PointLocation::Contained(index) => {
                coplanar_hull_face(&self.topology, index, node, &mut work)?
            }
            _ => None,
        };
        let seed = match location {
            _ if on_hull.is_some() => on_hull,
            super::location::PointLocation::Contained(index) => {
                let prepared = super::prepare_insertion_from_hint(
                    &self.topology,
                    node,
                    &BTreeSet::new(),
                    false,
                    Some(index),
                    &mut work,
                )?;
                let inserted = self.topology.nodes.len();
                super::mutable::apply_prepared_delaunay_volume_node_insertion_with_work(
                    &mut self.topology,
                    prepared,
                    &BTreeSet::new(),
                    false,
                    &mut work,
                )?;
                self.hint = self.topology.vertex_stars[inserted].first().copied();
                return Ok(());
            }
            super::location::PointLocation::Outside {
                tetrahedron,
                opposite,
            } => Some((tetrahedron, opposite)),
            super::location::PointLocation::Unresolved => None,
        };
        let conflict = search::visible_faces(&self.topology, node, seed, &mut work)?;
        if conflict.visible.is_empty() && conflict.coplanar.is_empty() {
            let prepared = super::prepare_insertion_from_hint(
                &self.topology,
                node,
                &BTreeSet::new(),
                false,
                None,
                &mut work,
            )?;
            let inserted = self.topology.nodes.len();
            super::mutable::apply_prepared_delaunay_volume_node_insertion_with_work(
                &mut self.topology,
                prepared,
                &BTreeSet::new(),
                false,
                &mut work,
            )?;
            self.hint = self.topology.vertex_stars[inserted].first().copied();
            return Ok(());
        }
        let removed = search::finite_conflict(&self.topology, node, &conflict, &mut work)?;
        let mut boundary = BTreeSet::new();
        let mut affected_nodes = BTreeSet::new();
        for index in &removed {
            let tet = self
                .topology
                .tetrahedron(*index)
                .expect("conflict tetrahedron is active");
            affected_nodes.extend(tet.vertex_indices);
            for opposite in 0..4 {
                let neighbor = tet.neighbors[opposite];
                if neighbor.is_some_and(|neighbor| removed.contains(&neighbor)) {
                    continue;
                }
                let side = face_side(
                    &self.topology,
                    tet.vertex_indices,
                    opposite,
                    node,
                    &mut work,
                )?;
                if neighbor.is_some() && side != FaceSide::Interior {
                    return Err(error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        format!(
                            "finite hull conflict boundary is not visible from the cavity side ({side:?} face, cavity of {} cells)",
                            removed.len()
                        ),
                    ));
                }
                if side == FaceSide::Interior {
                    boundary.insert(face(tet.vertex_indices, opposite));
                }
            }
        }
        let mut attachments = BTreeSet::new();
        for (index, opposite) in conflict.visible {
            if removed.contains(&index) {
                continue;
            }
            let tet = self
                .topology
                .tetrahedron(index)
                .expect("visible hull tetrahedron is active");
            let face = face(tet.vertex_indices, opposite);
            boundary.insert(face);
            attachments.insert(face);
        }
        if boundary.len() as u64 > options.maximum_cavity_boundary_faces {
            return Err(resource("hull insertion boundary-face limit exceeded"));
        }
        super::mutable::require_mutation_capacity(
            &self.topology,
            removed.len(),
            boundary.len(),
            options,
        )?;
        let inserted = self
            .topology
            .append_node(node)
            .map_err(mutation_topology_error)?;
        let created = boundary
            .into_iter()
            .map(|face| ([face[0], face[1], face[2], inserted], None))
            .collect();
        let created = self
            .topology
            .replace_hull_cavity(&removed, created, &attachments)
            .map_err(mutation_topology_error)?;
        if affected_nodes
            .into_iter()
            .chain([inserted])
            .any(|vertex| self.topology.vertex_stars[vertex as usize].is_empty())
        {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "hull insertion would remove an admitted point from the complex",
            ));
        }
        self.hint = created.first().copied();
        Ok(())
    }

    pub(in crate::cdt) fn freeze(
        self,
        options: DelaunayInsertionOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
        self.topology
            .freeze(options.topology, cancellation)
            .map_err(mutation_topology_error)
    }
}

/// A hull face whose plane holds `node`, among the cells whose closure contains it.
///
/// A node on a hull edge or vertex lies in several cells, and the walk may return an interior
/// one; the cells sharing the node's face, edge or vertex are reached across the faces whose
/// plane holds it. Missing the hull face would send the node down the interior path, which
/// cuts the cells under coplanar hull faces from the cavity although their circumspheres hold
/// the node, leaving the complex non-Delaunay.
fn coplanar_hull_face(
    topology: &TopologyMutation,
    index: u32,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(u32, usize)>, DelaunayInsertionError> {
    let mut pending = vec![index];
    let mut visited = BTreeSet::from([index]);
    while let Some(index) = pending.pop() {
        let tet = topology
            .tetrahedron(index)
            .expect("located tetrahedron is active");
        for opposite in 0..4 {
            if face_side(topology, tet.vertex_indices, opposite, node, work)? != FaceSide::Coplanar
            {
                continue;
            }
            match tet.neighbors[opposite] {
                None => return Ok(Some((index, opposite))),
                Some(neighbor) => {
                    if visited.insert(neighbor) {
                        if visited.len() as u64 > work.options.maximum_cavity_tetrahedra {
                            return Err(resource(
                                "hull insertion closure-star tetrahedron limit exceeded",
                            ));
                        }
                        pending.push(neighbor);
                    }
                }
            }
        }
    }
    Ok(None)
}

#[derive(Debug, PartialEq, Eq)]
enum FaceSide {
    Interior,
    Coplanar,
    Exterior,
}

fn face_side(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<FaceSide, DelaunayInsertionError> {
    let points = face(vertices, opposite).map(|index| topology.nodes()[index as usize]);
    work.predicate()?;
    let interior = topology
        .orient_nodes([
            points[0],
            points[1],
            points[2],
            topology.nodes()[vertices[opposite] as usize],
        ])
        .map_err(predicate_error)?;
    work.predicate()?;
    let query = topology
        .orient_nodes([points[0], points[1], points[2], node])
        .map_err(predicate_error)?;
    Ok(if query == PredicateSign::Zero {
        FaceSide::Coplanar
    } else if query == interior {
        FaceSide::Interior
    } else {
        FaceSide::Exterior
    })
}

fn face(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
    let mut result = [0; 3];
    let mut cursor = 0;
    for (slot, vertex) in vertices.into_iter().enumerate() {
        if slot != opposite {
            result[cursor] = vertex;
            cursor += 1;
        }
    }
    result.sort_unstable();
    result
}
