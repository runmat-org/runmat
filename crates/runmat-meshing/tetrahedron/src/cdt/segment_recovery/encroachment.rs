use std::collections::{BTreeSet, VecDeque};

use runmat_meshing_core::{
    quality::predicate::{diametral_sphere_position, distance, PredicateSign},
    StableDigest,
};

use super::{
    error, find_node, node_index, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DyadicNode, RecoveryWork, SegmentContext,
};
use crate::cdt::insertion::topology_access::TetrahedralTopology;

pub(super) fn encroaching_point(
    topology: &impl TetrahedralTopology,
    left: StableDigest,
    right: StableDigest,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<u32>, DelaunaySegmentRecoveryError> {
    let left_index = node_index(topology, left)
        .ok_or_else(|| invalid_topology(constraint_index, "left segment endpoint is missing"))?
        as u32;
    let right_index = node_index(topology, right)
        .ok_or_else(|| invalid_topology(constraint_index, "right segment endpoint is missing"))?
        as u32;
    let endpoints = [
        topology.nodes()[left_index as usize].coordinates_m,
        topology.nodes()[right_index as usize].coordinates_m,
    ];
    let mut queue = topology
        .vertex_star(left_index)
        .into_iter()
        .collect::<VecDeque<_>>();
    let mut visited_tetrahedra = BTreeSet::new();
    let mut classified = vec![None; topology.nodes().len()];
    classified[left_index as usize] = Some(true);
    classified[right_index as usize] = Some(true);
    let mut best = None::<(f64, StableDigest, u32)>;

    while let Some(tetrahedron_index) = queue.pop_front() {
        work.search_step(constraint_index)?;
        if !visited_tetrahedra.insert(tetrahedron_index) {
            continue;
        }
        let tetrahedron = topology.tetrahedron(tetrahedron_index).ok_or_else(|| {
            invalid_topology(
                constraint_index,
                "encroachment walk reached an inactive tetrahedron",
            )
        })?;
        for vertex in tetrahedron.vertex_indices {
            if classified[vertex as usize].is_some() {
                continue;
            }
            let node = topology.nodes()[vertex as usize];
            let encroaches = matches!(
                diametral_sphere_position(endpoints, node.coordinates_m).map_err(|failure| {
                    invalid_topology(
                        constraint_index,
                        format!("diametral-sphere predicate failed: {failure:?}"),
                    )
                })?,
                PredicateSign::Negative | PredicateSign::Zero
            );
            classified[vertex as usize] = Some(encroaches);
            if encroaches {
                let rank = circumradius_rank(endpoints, node.coordinates_m);
                let candidate = (rank, node.identity, vertex);
                if best.as_ref().is_none_or(|current| {
                    candidate.0 > current.0 || candidate.0 == current.0 && candidate.1 < current.1
                }) {
                    best = Some(candidate);
                }
            }
        }
        for opposite in 0..4 {
            let face_contains_encroaching_vertex = tetrahedron
                .vertex_indices
                .iter()
                .enumerate()
                .any(|(slot, vertex)| {
                    slot != opposite && classified[*vertex as usize] == Some(true)
                });
            if face_contains_encroaching_vertex {
                if let Some(neighbor) = tetrahedron.neighbors[opposite] {
                    if !visited_tetrahedra.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    Ok(best.map(|(_, _, vertex)| vertex))
}

pub(super) fn split_parameter(
    topology: &impl TetrahedralTopology,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    witness: u32,
) -> Result<DyadicNode, DelaunaySegmentRecoveryError> {
    let left_coordinates = find_node(topology, left.identity)
        .ok_or_else(|| invalid_topology(context.constraint_index, "left split node is missing"))?
        .coordinates_m;
    let right_coordinates = find_node(topology, right.identity)
        .ok_or_else(|| invalid_topology(context.constraint_index, "right split node is missing"))?
        .coordinates_m;
    let witness_coordinates = topology
        .nodes()
        .get(witness as usize)
        .ok_or_else(|| invalid_topology(context.constraint_index, "split witness is missing"))?
        .coordinates_m;
    let total_length = distance(context.first_coordinates, context.last_coordinates);
    let left_parameter = left.parameter()?;
    let right_parameter = right.parameter()?;
    let left_distance = distance(left_coordinates, witness_coordinates);
    let right_distance = distance(right_coordinates, witness_coordinates);
    let projected = if left_distance <= 2.0 * right_distance {
        left_parameter + left_distance / total_length
    } else {
        right_parameter - right_distance / total_length
    };
    let interval = right_parameter - left_parameter;
    let guarded = if projected.is_finite()
        && projected > left_parameter + 0.2 * interval
        && projected < right_parameter - 0.2 * interval
    {
        projected
    } else {
        left_parameter + 0.5 * interval
    };
    left.near_parameter(right, guarded, context)
}

fn circumradius_rank(endpoints: [[f64; 3]; 2], point: [f64; 3]) -> f64 {
    let mut left = subtract(point, endpoints[0]);
    let mut right = subtract(point, endpoints[1]);
    let scale = left
        .into_iter()
        .chain(right)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return f64::INFINITY;
    }
    left = left.map(|coordinate| coordinate / scale);
    right = right.map(|coordinate| coordinate / scale);
    let cross = cross(left, right);
    let denominator = dot(cross, cross);
    if denominator == 0.0 {
        f64::INFINITY
    } else {
        dot(left, left) * dot(right, right) / denominator
    }
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn invalid_topology(
    constraint_index: u32,
    reason: impl Into<String>,
) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;
    use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};

    fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates_m,
        }
    }

    #[test]
    fn encroachment_guides_the_split_away_from_blind_bisection() {
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, [0.0, 0.0, 0.0]),
                node(20, [10.0, 0.0, 0.0]),
                node(30, [2.0, 1.0, 0.0]),
                node(40, [0.0, 0.0, 2.0]),
            ],
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let context = SegmentContext {
            constraint_index: 0,
            first_identity: topology.nodes[0].identity,
            last_identity: topology.nodes[1].identity,
            first_coordinates: topology.nodes[0].coordinates_m,
            last_coordinates: topology.nodes[1].coordinates_m,
        };
        let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
        let witness = encroaching_point(
            &topology,
            context.first_identity,
            context.last_identity,
            0,
            &mut work,
        )
        .unwrap()
        .unwrap();

        assert_eq!(witness, 2);
        let split = split_parameter(
            &topology,
            context,
            DyadicNode::endpoint(context.first_identity, false),
            DyadicNode::endpoint(context.last_identity, true),
            witness,
        )
        .unwrap()
        .parameter()
        .unwrap();
        assert!((0.22..0.23).contains(&split));
    }
}
