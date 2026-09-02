use runmat_meshing_core::{
    quality::predicate::{diametral_sphere_position_constructed, distance, PredicateSign},
    StableDigest,
};

use super::{
    error, find_node, interpolate, node_index, DelaunaySegmentRecoveryError,
    DelaunaySegmentRecoveryErrorKind, DyadicNode, RecoveryWork, SegmentContext,
};
use crate::cdt::insertion::topology_access::TetrahedralTopology;

#[derive(Clone, Copy)]
pub(super) struct EncroachingPoint {
    pub(super) vertex: u32,
    pub(super) relation: PredicateSign,
}

pub(super) fn encroaching_point(
    topology: &impl TetrahedralTopology,
    left: StableDigest,
    right: StableDigest,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<EncroachingPoint>, DelaunaySegmentRecoveryError> {
    let left_index = node_index(topology, left)
        .ok_or_else(|| invalid_topology(constraint_index, "left segment endpoint is missing"))?
        as u32;
    let right_index = node_index(topology, right)
        .ok_or_else(|| invalid_topology(constraint_index, "right segment endpoint is missing"))?
        as u32;
    let endpoint_nodes = [
        topology.nodes()[left_index as usize],
        topology.nodes()[right_index as usize],
    ];
    let endpoints = endpoint_nodes.map(|node| node.coordinates_m);
    let broad_phase = diametral_broad_phase(endpoints);
    let mut best = None::<(f64, StableDigest, u32, PredicateSign)>;
    for (vertex, node) in topology.nodes().iter().enumerate() {
        work.search_step(constraint_index)?;
        if vertex == left_index as usize || vertex == right_index as usize {
            continue;
        }
        // The sphere is contained by this deliberately conservative cube.
        // Candidates admitted by the broad phase still receive the exact
        // constructed predicate below.
        if broad_phase.is_some_and(|[minimum, maximum]| {
            (0..3).any(|axis| {
                node.coordinates_m[axis] < minimum[axis] || node.coordinates_m[axis] > maximum[axis]
            })
        }) {
            continue;
        }
        let relation = diametral_sphere_position_constructed(
            endpoint_nodes.map(|node| {
                crate::cdt::DelaunayVolumeNode::explicit(node.identity, node.coordinates_m)
                    .predicate_point()
            }),
            crate::cdt::DelaunayVolumeNode::explicit(node.identity, node.coordinates_m)
                .predicate_point(),
        )
        .map_err(|failure| {
            invalid_topology(
                constraint_index,
                format!("diametral-sphere predicate failed: {failure:?}"),
            )
        })?;
        if matches!(relation, PredicateSign::Negative | PredicateSign::Zero) {
            let rank = circumradius_rank(endpoints, node.coordinates_m);
            let candidate = (rank, node.identity, vertex as u32, relation);
            if best.as_ref().is_none_or(|current| {
                candidate.0 > current.0 || candidate.0 == current.0 && candidate.1 < current.1
            }) {
                best = Some(candidate);
            }
        }
    }
    Ok(best.map(|(_, _, vertex, relation)| EncroachingPoint { vertex, relation }))
}

fn diametral_broad_phase(endpoints: [[f64; 3]; 2]) -> Option<[[f64; 3]; 2]> {
    let center = std::array::from_fn(|axis| 0.5 * endpoints[0][axis] + 0.5 * endpoints[1][axis]);
    // Half the L1 endpoint distance is an upper bound for the Euclidean
    // diametral radius. Round it outward again to absorb arithmetic error in
    // both the radius and center construction.
    let radius = 0.5
        * (0..3)
            .map(|axis| (endpoints[1][axis] - endpoints[0][axis]).abs())
            .sum::<f64>();
    let scale = center
        .into_iter()
        .map(f64::abs)
        .fold(radius.max(1.0), f64::max);
    let radius = radius + 16.0 * f64::EPSILON * scale;
    radius.is_finite().then(|| {
        [
            center.map(|coordinate| coordinate - radius),
            center.map(|coordinate| coordinate + radius),
        ]
    })
}

pub(super) fn split_parameter(
    topology: &impl TetrahedralTopology,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    witness: u32,
    acute_lineage: [bool; 2],
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
    let [left_is_acute, right_is_acute] = acute_lineage;
    if left_is_acute && right_is_acute {
        return left.midpoint(right, context);
    }
    let left_distance = distance(left_coordinates, witness_coordinates);
    let right_distance = distance(right_coordinates, witness_coordinates);
    let projected = if left_is_acute {
        distance(context.first_coordinates, witness_coordinates) / total_length
    } else if right_is_acute {
        1.0 - distance(context.last_coordinates, witness_coordinates) / total_length
    } else if 2.0 * left_distance < distance(left_coordinates, right_coordinates) {
        left_parameter + left_distance / total_length
    } else if 2.0 * right_distance < distance(left_coordinates, right_coordinates) {
        right_parameter - right_distance / total_length
    } else {
        left_parameter + 0.5 * (right_parameter - left_parameter)
    };
    let interval = right_parameter - left_parameter;
    let mut guarded = if projected.is_finite()
        && projected > left_parameter + 0.2 * interval
        && projected < right_parameter - 0.2 * interval
    {
        projected
    } else {
        left_parameter + 0.5 * interval
    };
    if left_is_acute || right_is_acute {
        let projected_coordinates =
            interpolate(context.first_coordinates, context.last_coordinates, guarded);
        let nonacute_coordinates = if left_is_acute {
            right_coordinates
        } else {
            left_coordinates
        };
        if distance(projected_coordinates, nonacute_coordinates)
            < distance(projected_coordinates, witness_coordinates)
        {
            guarded = left_parameter + 0.5 * interval;
        }
    }
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
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
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

        assert_eq!(witness.vertex, 2);
        assert_eq!(witness.relation, PredicateSign::Negative);
        let split = split_parameter(
            &topology,
            context,
            DyadicNode::endpoint(context.first_identity, false),
            DyadicNode::endpoint(context.last_identity, true),
            witness.vertex,
            [false, false],
        )
        .unwrap()
        .parameter()
        .unwrap();
        assert!((0.22..0.23).contains(&split));
    }
}
