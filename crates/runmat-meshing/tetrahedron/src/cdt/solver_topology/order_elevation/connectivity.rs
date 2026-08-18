use runmat_meshing_core::{
    BoundaryEdgeOrder, BoundaryTriangleOrder, ElementOrder, FieldTopologyLocation,
    SolverMeshTopology,
};

use super::{sorted_edge, MidpointMap, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

pub(super) fn elevate(
    topology: &mut SolverMeshTopology,
    midpoint_by_edge: &MidpointMap,
) -> Result<(), DelaunaySolverTopologyError> {
    for element in &mut topology.volume_elements {
        let corners: [u64; 4] = element
            .node_ids
            .as_slice()
            .try_into()
            .map_err(|_| invalid("linear element does not have exactly four corner nodes"))?;
        element.order = ElementOrder::Tet10;
        for edge in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            element.node_ids.push(midpoint(
                midpoint_by_edge,
                [corners[edge[0]], corners[edge[1]]],
            )?);
        }
    }
    for face in &mut topology.boundary_faces {
        let corners: [u64; 3] = face.node_ids.as_slice().try_into().map_err(|_| {
            invalid("linear boundary face does not have exactly three corner nodes")
        })?;
        face.order = BoundaryTriangleOrder::Tri6;
        for edge in [[0, 1], [1, 2], [2, 0]] {
            face.node_ids.push(midpoint(
                midpoint_by_edge,
                [corners[edge[0]], corners[edge[1]]],
            )?);
        }
    }
    for edge in &mut topology.boundary_edges {
        let corners: [u64; 2] =
            edge.node_ids.as_slice().try_into().map_err(|_| {
                invalid("linear boundary edge does not have exactly two corner nodes")
            })?;
        edge.order = BoundaryEdgeOrder::Line3;
        edge.node_ids.push(midpoint(midpoint_by_edge, corners)?);
    }
    let node_ids = topology.nodes.iter().map(|node| node.node_id).collect();
    let node_field = topology
        .field_topologies
        .iter_mut()
        .find(|field| field.location == FieldTopologyLocation::Node)
        .ok_or_else(|| invalid("solver topology has no node field map"))?;
    node_field.ordered_entity_ids = node_ids;
    Ok(())
}

fn midpoint(
    midpoint_by_edge: &MidpointMap,
    corners: [u64; 2],
) -> Result<u64, DelaunaySolverTopologyError> {
    midpoint_by_edge
        .get(&sorted_edge(corners))
        .copied()
        .ok_or_else(|| invalid("Tet10 connectivity has no canonical edge midpoint"))
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
