use crate::{
    math::{cross, norm, sub, Triangle3},
    SurfaceDiscretization, SurfaceElement,
};
use runmat_meshing_cad::SourceTopologyModel;

use super::types::SurfaceValidationError;

pub(super) fn topology_face_points(
    topology: &SourceTopologyModel,
    node_ids: [u32; 3],
) -> Result<Triangle3, SurfaceValidationError> {
    Ok([
        topology_node(topology, node_ids[0])?,
        topology_node(topology, node_ids[1])?,
        topology_node(topology, node_ids[2])?,
    ])
}

pub(super) fn surface_element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<Triangle3, SurfaceValidationError> {
    Ok([
        surface_node(surface, element, element.node_ids[0])?,
        surface_node(surface, element, element.node_ids[1])?,
        surface_node(surface, element, element.node_ids[2])?,
    ])
}

pub(super) fn unit_normal(points: Triangle3) -> Option<[f64; 3]> {
    let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
    let length = norm(normal);
    (length > 0.0).then_some([normal[0] / length, normal[1] / length, normal[2] / length])
}

pub(super) fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn topology_node(
    topology: &SourceTopologyModel,
    node_id: u32,
) -> Result<[f64; 3], SurfaceValidationError> {
    topology
        .vertices
        .get(node_id as usize)
        .filter(|vertex| vertex.vertex_id == node_id)
        .map(|vertex| vertex.coordinates_m)
        .ok_or(SurfaceValidationError::MissingSurfaceNode {
            element_id: u32::MAX,
            node_id,
        })
}

fn surface_node(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
    node_id: u32,
) -> Result<[f64; 3], SurfaceValidationError> {
    surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
        .ok_or(SurfaceValidationError::MissingSurfaceNode {
            element_id: element.element_id,
            node_id,
        })
}
