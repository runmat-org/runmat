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
        let seed = match location {
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
        let visible = search::visible_faces(&self.topology, node, seed, &mut work)?;
        if visible.is_empty() {
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
        let mut removed = search::finite_conflict(&self.topology, node, &visible, &mut work)?;
        // A cavity face can be coplanar with the new point only when the point lies exactly on
        // that face's circumcircle, so both cells across it are in an in-sphere tie (cocircular
        // cap points of a cylinder, a regular grid). The symbolic tie-break may split such a pair;
        // the tied neighbor then joins the cavity, so no flat cell is created. A face the point
        // does not see across a non-tied cell remains an error.
        let (boundary, affected_nodes) = loop {
            let mut boundary = BTreeSet::new();
            let mut affected_nodes = BTreeSet::new();
            let mut grow = None;
            'cells: for index in &removed {
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
                    if let Some(neighbor) = neighbor {
                        if side != FaceSide::Interior {
                            let tied = cospherical(&self.topology, neighbor, node)?;
                            if side == FaceSide::Coplanar && tied {
                                grow = Some(neighbor);
                                break 'cells;
                            }
                            return Err(error(
                                DelaunayInsertionErrorKind::InvalidTopology,
                                format!(
                                    "finite hull conflict boundary is not visible from the cavity side ({side:?} face, neighbor in-sphere tie {tied}, cavity of {} cells)",
                                    removed.len()
                                ),
                            ));
                        }
                    }
                    if side == FaceSide::Interior {
                        boundary.insert(face(tet.vertex_indices, opposite));
                    }
                }
            }
            match grow {
                Some(neighbor) => {
                    removed.insert(neighbor);
                    if removed.len() as u64 > work.options.maximum_cavity_tetrahedra {
                        return Err(resource(
                            "hull insertion conflict tetrahedron limit exceeded",
                        ));
                    }
                }
                None => break (boundary, affected_nodes),
            }
        };
        let mut boundary = boundary;
        let mut attachments = BTreeSet::new();
        for (index, opposite) in visible {
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

/// True when `node` lies exactly on the circumsphere of `cell` (unperturbed in-sphere tie).
fn cospherical(
    topology: &TopologyMutation,
    cell: u32,
    node: DelaunayVolumeNode,
) -> Result<bool, DelaunayInsertionError> {
    let corners = topology
        .tetrahedron(cell)
        .expect("live conflict adjacency")
        .vertex_indices
        .map(|vertex| topology.nodes()[vertex as usize]);
    let relation = crate::cdt::topology::insphere_materialized_volume_nodes([
        corners[0], corners[1], corners[2], corners[3], node,
    ])
    .map_err(predicate_error)?;
    Ok(relation == PredicateSign::Zero)
}
