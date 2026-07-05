use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{PlcProtectedEdge, TopologyEntityId};

pub(crate) fn face_edges(node_ids: [TopologyEntityId; 3]) -> [[TopologyEntityId; 2]; 3] {
    [
        sorted_edge([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_edge([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_edge([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

pub(crate) fn source_edge_ids_for_face_edges(
    protected_edges: &[PlcProtectedEdge],
    node_ids: [TopologyEntityId; 3],
) -> [Option<TopologyEntityId>; 3] {
    face_edges(node_ids).map(|face_edge| {
        protected_edges
            .iter()
            .find(|protected_edge| sorted_edge(protected_edge.node_ids.clone()) == face_edge)
            .map(|protected_edge| protected_edge.source_edge_id.clone())
    })
}

pub(crate) fn source_edge_ids_for_boundary_face_edges(
    protected_edges: &[PlcProtectedEdge],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    node_ids: [TopologyEntityId; 3],
    tolerance_m: f64,
) -> [Option<TopologyEntityId>; 3] {
    let mut source_edge_ids = source_edge_ids_for_face_edges(protected_edges, node_ids.clone());
    for edge_index in 0..3 {
        if source_edge_ids[edge_index].is_some() {
            continue;
        }
        let left = &node_ids[edge_index];
        let right = &node_ids[(edge_index + 1) % 3];
        let Some(left_point) = coordinates_by_id.get(left).copied() else {
            continue;
        };
        let Some(right_point) = coordinates_by_id.get(right).copied() else {
            continue;
        };
        source_edge_ids[edge_index] = protected_edges.iter().find_map(|protected_edge| {
            let start = coordinates_by_id
                .get(&protected_edge.node_ids[0])
                .copied()?;
            let end = coordinates_by_id
                .get(&protected_edge.node_ids[1])
                .copied()?;
            (point_lies_on_segment(left_point, start, end, tolerance_m)
                && point_lies_on_segment(right_point, start, end, tolerance_m))
            .then_some(protected_edge.source_edge_id.clone())
        });
    }
    source_edge_ids
}

pub(crate) fn sorted_edge(mut node_ids: [TopologyEntityId; 2]) -> [TopologyEntityId; 2] {
    node_ids.sort();
    node_ids
}

fn point_lies_on_segment(
    point: [f64; 3],
    start: [f64; 3],
    end: [f64; 3],
    tolerance_m: f64,
) -> bool {
    let length = distance(start, end);
    let tolerance_m = tolerance_m.max(length * 1.0e-9);
    if length <= tolerance_m {
        return distance(point, start) <= tolerance_m;
    }
    let distance_sum = distance(start, point) + distance(point, end);
    (distance_sum - length).abs() <= tolerance_m
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}
