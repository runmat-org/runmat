use std::collections::BTreeSet;

use super::domain::{CfdDomainTopology, CfdDomainTopologyBasis};

pub fn face_count(topology: &CfdDomainTopology) -> usize {
    if topology.control_volume_connectivity_coverage_ratio > 0.0
        && topology.control_volume_boundary_face_count > 0
    {
        topology.control_volume_boundary_face_count
    } else {
        topology.control_volume_count
    }
    .max(1)
}

fn graph_edge_target(interface_face_count: usize) -> usize {
    if interface_face_count < 2 {
        return 0;
    }
    let line_edge_count = interface_face_count - 1;
    let complete_graph_edge_count = interface_face_count * (interface_face_count - 1) / 2;
    line_edge_count.min(complete_graph_edge_count)
}

pub fn graph_edges_for_topology(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
) -> (Vec<(usize, usize)>, f64) {
    if topology.basis == CfdDomainTopologyBasis::SolverMesh {
        return (
            mesh_boundary_face_adjacencies(topology, interface_face_count),
            1.0,
        );
    }
    let target = graph_edge_target(interface_face_count);
    if target == 0 {
        return (Vec::new(), 0.0);
    }

    let mut edges = Vec::with_capacity(target);
    let mut seen = BTreeSet::new();
    for (left, right) in fallback_graph_edges(interface_face_count, target) {
        let edge = (left.min(right), left.max(right));
        if seen.insert(edge) {
            edges.push(edge);
            if edges.len() == target {
                break;
            }
        }
    }
    (edges, 0.0)
}

fn mesh_boundary_face_adjacencies(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
) -> Vec<(usize, usize)> {
    let face_count = topology
        .element_topology_element_edges
        .len()
        .min(interface_face_count);
    let mut edges = Vec::new();
    for left in 0..face_count {
        for right in (left + 1)..face_count {
            if topology.element_topology_element_edges[left]
                .iter()
                .any(|edge| topology.element_topology_element_edges[right].contains(edge))
            {
                edges.push((left, right));
            }
        }
    }
    edges
}

pub fn connectivity_coverage_ratio(
    topology: &CfdDomainTopology,
    interface_face_count: usize,
    edge_count: usize,
) -> f64 {
    if topology.basis == CfdDomainTopologyBasis::SolverMesh {
        let expected = mesh_boundary_face_adjacencies(topology, interface_face_count).len();
        return if expected == 0 {
            f64::from(interface_face_count < 2)
        } else {
            (edge_count as f64 / expected as f64).clamp(0.0, 1.0)
        };
    }
    let target = graph_edge_target(interface_face_count);
    if target == 0 {
        return 1.0;
    }
    (edge_count as f64 / target as f64).clamp(0.0, 1.0)
}

pub fn pressure_drop(pressure: &[f64]) -> f64 {
    match (pressure.first(), pressure.last()) {
        (Some(first), Some(last)) => first - last,
        _ => 0.0,
    }
}

pub fn mean_axial_velocity(velocity: &[f64]) -> f64 {
    let node_count = velocity.len() / 3;
    (0..node_count)
        .map(|node| velocity.get(node * 3).copied().unwrap_or(0.0))
        .sum::<f64>()
        / node_count.max(1) as f64
}

pub fn cell_centered_scalar(nodal: &[f64]) -> Vec<f64> {
    let node_count = nodal.len();
    let cell_count = node_count.saturating_sub(1).max(1);
    let mut cell_values = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        let left = cell.min(node_count.saturating_sub(1));
        let right = (cell + 1).min(node_count.saturating_sub(1));
        let left_value = nodal.get(left).copied().unwrap_or(0.0);
        let right_value = nodal.get(right).copied().unwrap_or(left_value);
        cell_values.push(0.5 * (left_value + right_value));
    }
    cell_values
}

pub fn resample_scalar_profile(values: &[f64], target_count: usize) -> Vec<f64> {
    let target_count = target_count.max(1);
    if values.is_empty() {
        return vec![0.0; target_count];
    }
    if values.len() == target_count {
        return values.to_vec();
    }
    if target_count == 1 {
        return vec![values[0]];
    }
    let source_max = values.len().saturating_sub(1) as f64;
    let target_max = target_count.saturating_sub(1) as f64;
    (0..target_count)
        .map(|target_index| {
            let source_position = target_index as f64 * source_max / target_max.max(1.0);
            let left = source_position.floor() as usize;
            let right = source_position.ceil() as usize;
            if left == right {
                values.get(left).copied().unwrap_or(0.0)
            } else {
                let t = source_position - left as f64;
                let left_value = values.get(left).copied().unwrap_or(0.0);
                let right_value = values.get(right).copied().unwrap_or(left_value);
                left_value * (1.0 - t) + right_value * t
            }
        })
        .collect()
}

fn fallback_graph_edges(face_count: usize, edge_target: usize) -> Vec<(usize, usize)> {
    if face_count < 2 || edge_target == 0 {
        return Vec::new();
    }
    let complete_graph_edge_count = face_count * (face_count - 1) / 2;
    let edge_target = edge_target.min(complete_graph_edge_count);
    let mut edges = Vec::with_capacity(edge_target);
    for span in 1..face_count {
        for left in 0..face_count - span {
            edges.push((left, left + span));
            if edges.len() == edge_target {
                return edges;
            }
        }
    }
    edges
}
