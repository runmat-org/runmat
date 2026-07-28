use crate::{
    math::{cross, norm, sub, Triangle3},
    SurfaceDiscretization, SurfaceElement,
};

use super::types::SurfaceRecoveryError;

pub(super) fn triangle_unit_normal(points: Triangle3) -> Option<[f64; 3]> {
    let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
    let length = norm(normal);
    (length > 0.0).then_some([normal[0] / length, normal[1] / length, normal[2] / length])
}

pub(super) fn surface_element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<Triangle3, SurfaceRecoveryError> {
    Ok([
        surface_node(surface, element, element.node_ids[0])?,
        surface_node(surface, element, element.node_ids[1])?,
        surface_node(surface, element, element.node_ids[2])?,
    ])
}

fn surface_node(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
    node_id: u32,
) -> Result<[f64; 3], SurfaceRecoveryError> {
    let node = surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .ok_or(SurfaceRecoveryError::MissingSurfaceNode {
            element_id: element.element_id,
            node_id,
        })?;
    if node.coordinates_m.iter().any(|value| !value.is_finite()) {
        return Err(SurfaceRecoveryError::NonFiniteSurfaceNode { node_id });
    }
    Ok(node.coordinates_m)
}
