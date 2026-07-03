use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::Point3;

use super::super::{topology::face_edges, ConstrainedCavityNode};

#[cfg(test)]
pub(crate) fn boundary_face_centroid_node(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> ConstrainedCavityNode {
    boundary_face_split_node(face, boundary_nodes, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
}

pub(crate) fn boundary_face_split_node_candidates(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<ConstrainedCavityNode> {
    let mut barycentric_candidates = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.70, 0.05, 0.25],
        [0.70, 0.25, 0.05],
        [0.05, 0.70, 0.25],
        [0.25, 0.70, 0.05],
        [0.05, 0.25, 0.70],
        [0.25, 0.05, 0.70],
    ]
    .into_iter()
    .collect::<Vec<_>>();
    for first in 1..10 {
        for second in 1..(10 - first) {
            let third = 10 - first - second;
            if third == 0 {
                continue;
            }
            let barycentric = [
                first as f64 / 10.0,
                second as f64 / 10.0,
                third as f64 / 10.0,
            ];
            if !barycentric_candidates.iter().any(|candidate| {
                candidate
                    .iter()
                    .zip(barycentric)
                    .all(|(left, right)| (*left - right).abs() <= 1.0e-12)
            }) {
                barycentric_candidates.push(barycentric);
            }
        }
    }
    barycentric_candidates
        .into_iter()
        .map(|barycentric| boundary_face_split_node(face, boundary_nodes, barycentric))
        .collect()
}

pub(crate) fn boundary_face_edge_split_node_candidates(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<([u32; 2], ConstrainedCavityNode)> {
    face_edges(face)
        .into_iter()
        .flat_map(|edge| {
            [0.5, 0.25, 0.75].into_iter().map(move |fraction| {
                (
                    edge,
                    boundary_edge_split_node(edge, boundary_nodes, fraction),
                )
            })
        })
        .collect()
}

pub(crate) fn boundary_face_mid_edge_split_nodes(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Vec<ConstrainedCavityNode> {
    let mut next_node_id = next_available_node_id(boundary_nodes);
    face_edges(face)
        .into_iter()
        .map(|edge| {
            while boundary_nodes.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let mut node = boundary_edge_split_node(edge, boundary_nodes, 0.5);
            node.node_id = next_node_id;
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect()
}

pub(crate) fn boundary_edge_split_node(
    edge: [u32; 2],
    boundary_nodes: &BTreeMap<u32, Point3>,
    fraction: f64,
) -> ConstrainedCavityNode {
    let points = edge.map(|node_id| boundary_nodes[&node_id]);
    ConstrainedCavityNode {
        node_id: next_available_node_id(boundary_nodes),
        coordinates_m: [
            points[0][0] * (1.0 - fraction) + points[1][0] * fraction,
            points[0][1] * (1.0 - fraction) + points[1][1] * fraction,
            points[0][2] * (1.0 - fraction) + points[1][2] * fraction,
        ],
    }
}

pub(crate) fn boundary_edge_patch_split_node(
    edge: [u32; 2],
    opposite_nodes: [u32; 2],
    boundary_nodes: &BTreeMap<u32, Point3>,
    weights: [f64; 4],
) -> ConstrainedCavityNode {
    let points = [
        boundary_nodes[&edge[0]],
        boundary_nodes[&edge[1]],
        boundary_nodes[&opposite_nodes[0]],
        boundary_nodes[&opposite_nodes[1]],
    ];
    ConstrainedCavityNode {
        node_id: next_available_node_id(boundary_nodes),
        coordinates_m: [
            weights[0] * points[0][0]
                + weights[1] * points[1][0]
                + weights[2] * points[2][0]
                + weights[3] * points[3][0],
            weights[0] * points[0][1]
                + weights[1] * points[1][1]
                + weights[2] * points[2][1]
                + weights[3] * points[3][1],
            weights[0] * points[0][2]
                + weights[1] * points[1][2]
                + weights[2] * points[2][2]
                + weights[3] * points[3][2],
        ],
    }
}

pub(crate) fn boundary_face_split_node(
    face: [u32; 3],
    boundary_nodes: &BTreeMap<u32, Point3>,
    barycentric: [f64; 3],
) -> ConstrainedCavityNode {
    let points = face.map(|node_id| boundary_nodes[&node_id]);
    ConstrainedCavityNode {
        node_id: next_available_node_id(boundary_nodes),
        coordinates_m: [
            points[0][0] * barycentric[0]
                + points[1][0] * barycentric[1]
                + points[2][0] * barycentric[2],
            points[0][1] * barycentric[0]
                + points[1][1] * barycentric[1]
                + points[2][1] * barycentric[2],
            points[0][2] * barycentric[0]
                + points[1][2] * barycentric[1]
                + points[2][2] * barycentric[2],
        ],
    }
}

fn next_available_node_id(boundary_nodes: &BTreeMap<u32, Point3>) -> u32 {
    let mut node_id = boundary_nodes
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while boundary_nodes.contains_key(&node_id) {
        node_id = node_id.saturating_add(1);
    }
    node_id
}
