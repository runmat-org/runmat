use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::predicate::{distance_squared, Point3, Triangle3};

use super::{
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions,
};

pub(super) fn boundary_node_coordinates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillError> {
    let coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for face in &cavity.boundary_faces {
        for node_id in face.node_ids {
            if !coordinates.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    Ok(coordinates)
}

pub(super) fn cavity_boundary_node_ids(cavity: &ConstrainedCavity) -> BTreeSet<u32> {
    cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect()
}

pub(super) fn candidate_respects_protected_boundary_distance(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    point: Point3,
    options: ConstrainedCavityRefillOptions,
) -> bool {
    if options.min_protected_node_distance_m <= 0.0 || cavity.protected_node_ids.is_empty() {
        return true;
    }
    let min_distance_squared = options.min_protected_node_distance_m.powi(2);
    cavity.protected_node_ids.iter().all(|node_id| {
        boundary_nodes.get(node_id).is_none_or(|protected_point| {
            distance_squared(point, *protected_point) > min_distance_squared
        })
    })
}

pub(super) fn cavity_boundary_triangles(
    cavity: &ConstrainedCavity,
    nodes: &BTreeMap<u32, Point3>,
) -> Result<Vec<Triangle3>, ConstrainedCavityRefillError> {
    cavity
        .boundary_faces
        .iter()
        .map(|face| {
            Ok([
                *nodes.get(&face.node_ids[0]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[0],
                    },
                )?,
                *nodes.get(&face.node_ids[1]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[1],
                    },
                )?,
                *nodes.get(&face.node_ids[2]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[2],
                    },
                )?,
            ])
        })
        .collect()
}

pub(super) fn cavity_boundary_node_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Option<Point3> {
    let node_ids = cavity_boundary_node_ids(cavity);
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0_f64; 3];
    for node_id in &node_ids {
        let point = boundary_nodes.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

pub(super) fn next_cavity_node_id(cavity: &ConstrainedCavity) -> u32 {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}
