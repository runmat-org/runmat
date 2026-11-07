mod broad_phase;
mod clearance;
mod placement;
use broad_phase::DiametralRegion;
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
    let endpoints = endpoint_nodes.map(|node| node.coordinates_m.canonical_value());
    let region = DiametralRegion::new(endpoints);
    let mut best = None::<(f64, StableDigest, u32, PredicateSign)>;
    let bounds = region.as_ref().map(|region| region.bounds);
    let candidates = topology
        .node_candidates_in_region(
            bounds,
            &|bounds| {
                region
                    .as_ref()
                    .is_none_or(|region| region.intersects(bounds))
            },
            &mut || work.search_point_index_step(constraint_index),
        )
        .map_err(|mut failure| {
            failure.reason = format!(
                "{}; point-query nodes={}, endpoints={endpoints:?}, bounds={bounds:?}",
                failure.reason,
                topology.nodes().len()
            );
            failure
        })?;
    // Reserve the unchanged predicate work before dispatch, then borrow only nodes.
    // Mutable topology indexes and recovery counters never cross worker boundaries.
    for _ in &candidates {
        work.search_encroachment_step(constraint_index)?;
    }
    let nodes = topology.nodes();
    let cancellation = work.cancellation;
    for wave in candidates.chunks(256) {
        let evaluate = |&vertex: &u32| {
            if cancellation.is_cancelled() {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::Cancelled,
                    Some(constraint_index),
                    "cancelled",
                ));
            }
            if vertex == left_index || vertex == right_index {
                return Ok(None);
            }
            let node = nodes[vertex as usize];
            let relation = diametral_sphere_position_constructed(
                endpoint_nodes.map(|node| {
                    crate::cdt::DelaunayVolumeNode::explicit(
                        node.identity,
                        node.coordinates_m.canonical_value(),
                    )
                    .predicate_point()
                }),
                crate::cdt::DelaunayVolumeNode::explicit(
                    node.identity,
                    node.coordinates_m.canonical_value(),
                )
                .predicate_point(),
            )
            .map_err(|failure| {
                invalid_topology(
                    constraint_index,
                    format!("diametral-sphere predicate failed: {failure:?}"),
                )
            })?;
            Ok(
                matches!(relation, PredicateSign::Negative | PredicateSign::Zero).then(|| {
                    (
                        circumradius_rank(endpoints, node.coordinates_m.canonical_value()),
                        node.identity,
                        vertex,
                        relation,
                    )
                }),
            )
        };
        let outcomes = if wave.len() >= 32 {
            runmat_cpu_execution::ordered_map(wave, evaluate)
        } else {
            wave.iter().map(evaluate).collect()
        };
        for candidate in outcomes.into_iter().filter_map(|result| result.transpose()) {
            let candidate = candidate?;
            if best.as_ref().is_none_or(|current| {
                candidate.0 > current.0 || candidate.0 == current.0 && candidate.1 < current.1
            }) {
                best = Some(candidate);
            }
        }
    }
    Ok(best.map(|(_, _, vertex, relation)| EncroachingPoint { vertex, relation }))
}

pub(super) fn split_parameter(
    topology: &impl TetrahedralTopology,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    witness: u32,
    acute_lineage: [bool; 2],
    work: &mut RecoveryWork<'_>,
) -> Result<DyadicNode, DelaunaySegmentRecoveryError> {
    let left_coordinates = find_node(topology, left.identity)
        .ok_or_else(|| invalid_topology(context.constraint_index, "left split node is missing"))?
        .coordinates_m
        .canonical_value();
    let right_coordinates = find_node(topology, right.identity)
        .ok_or_else(|| invalid_topology(context.constraint_index, "right split node is missing"))?
        .coordinates_m
        .canonical_value();
    let witness_node = topology
        .nodes()
        .get(witness as usize)
        .ok_or_else(|| invalid_topology(context.constraint_index, "split witness is missing"))?;
    // A later pass must reuse an existing point on this exact authored segment.
    // Reprojecting and rounding it as a new off-centre choice can produce a
    // succession of nearby points instead of recovering the original chain.
    if let Some(parameter) = retained_segment_parameter(witness_node, context, left, right) {
        return Ok(parameter);
    }
    if acute_lineage == [true, true] {
        return left.midpoint(right, context);
    }
    let preferred = placement::preferred(
        [left_coordinates, right_coordinates],
        witness_node.coordinates_m.canonical_value(),
        context,
        [left.parameter()?, right.parameter()?],
        acute_lineage,
    );
    clearance::choose(
        [
            *find_node(topology, left.identity).expect("original split endpoint"),
            *find_node(topology, right.identity).expect("original split endpoint"),
        ],
        *witness_node,
        context,
        [left, right],
        preferred.parameters,
        preferred.radial_origin,
        work,
    )
}

fn retained_segment_parameter(
    witness: &crate::cdt::DelaunayVolumeNode,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
) -> Option<DyadicNode> {
    let runmat_meshing_core::quality::predicate::Point3Construction::Segment {
        endpoints,
        parameter_numerator,
        parameter_exponent,
    } = witness.construction
    else {
        return None;
    };
    if parameter_exponent >= 63 || parameter_numerator > 1_u64 << parameter_exponent {
        return None;
    }
    // A segment is undirected. Another recovery of the same edge (an adjacent facet's
    // support triangle walks it the other way) constructs its points from the reversed
    // endpoints; parameter t from the far end is exactly 1 - t from this one.
    let (mut parameter_numerator, mut parameter_exponent) =
        if endpoints == [context.first_coordinates, context.last_coordinates] {
            (parameter_numerator, parameter_exponent)
        } else if endpoints == [context.last_coordinates, context.first_coordinates] {
            ((1_u64 << parameter_exponent) - parameter_numerator, parameter_exponent)
        } else {
            return None;
        };
    while parameter_exponent > 0 && parameter_numerator.is_multiple_of(2) {
        parameter_numerator /= 2;
        parameter_exponent -= 1;
    }
    let exponent = parameter_exponent.max(left.exponent).max(right.exponent);
    let parameter = u128::from(parameter_numerator) << (exponent - parameter_exponent);
    let lower = u128::from(left.numerator) << (exponent - left.exponent);
    let upper = u128::from(right.numerator) << (exponent - right.exponent);
    (lower < parameter && parameter < upper).then_some(DyadicNode {
        identity: witness.identity,
        numerator: parameter_numerator,
        exponent: parameter_exponent,
    })
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
            first_coordinates: topology.nodes[0].coordinates_m.canonical_value(),
            last_coordinates: topology.nodes[1].coordinates_m.canonical_value(),
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
            &mut work,
        )
        .unwrap()
        .parameter()
        .unwrap();
        assert!((0.22..0.23).contains(&split));
    }
    #[test]
    fn later_pass_reuses_the_original_segment_point_and_exact_parameter() {
        use runmat_meshing_core::quality::predicate::Point3Construction;
        let endpoints = [[0., 0., 0.], [10., 0., 0.]];
        let mut witness = node(30, [10. * 333. / 1024., 0., 0.]);
        witness.construction = Point3Construction::Segment {
            endpoints,
            parameter_numerator: 333,
            parameter_exponent: 10,
        };
        let topology = build_delaunay_volume_topology(
            vec![
                node(10, endpoints[0]),
                node(20, endpoints[1]),
                witness,
                node(40, [0., 1., 0.]),
                node(50, [0., 0., 1.]),
            ],
            vec![[0, 2, 3, 4], [1, 2, 3, 4]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let context = SegmentContext {
            constraint_index: 0,
            first_identity: topology.nodes[0].identity,
            last_identity: topology.nodes[1].identity,
            first_coordinates: endpoints[0],
            last_coordinates: endpoints[1],
        };
        let left = DyadicNode::endpoint(context.first_identity, false);
        let right = DyadicNode::endpoint(context.last_identity, true);
        let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
        let split =
            split_parameter(&topology, context, left, right, 2, [true, true], &mut work).unwrap();
        assert_eq!(
            (split.identity, split.numerator, split.exponent),
            (witness.identity, 333, 10)
        );
        let foreign = SegmentContext {
            last_coordinates: [20., 0., 0.],
            ..context
        };
        assert!(retained_segment_parameter(&witness, foreign, left, right).is_none());
        assert!(retained_segment_parameter(&witness, context, split, right).is_none());
    }

    #[test]
    fn a_point_constructed_on_the_reversed_segment_is_reused_at_its_complementary_parameter() {
        use runmat_meshing_core::quality::predicate::Point3Construction;
        // An adjacent support triangle walked this edge last-to-first and put a point at
        // 333/1024 from the far end. The near-collinear witness cannot be cleared by any
        // off-centre split, so recovery must reuse it at exactly 691/1024.
        let endpoints = [[0., 0., 0.], [10., 0., 0.]];
        let mut witness = node(30, [10. * 691. / 1024., 0., 0.]);
        witness.construction = Point3Construction::Segment {
            endpoints: [endpoints[1], endpoints[0]],
            parameter_numerator: 333,
            parameter_exponent: 10,
        };
        let context = SegmentContext {
            constraint_index: 0,
            first_identity: StableDigest::from_bytes([10; 32]),
            last_identity: StableDigest::from_bytes([20; 32]),
            first_coordinates: endpoints[0],
            last_coordinates: endpoints[1],
        };
        let left = DyadicNode::endpoint(context.first_identity, false);
        let right = DyadicNode::endpoint(context.last_identity, true);
        let retained = retained_segment_parameter(&witness, context, left, right).unwrap();
        assert_eq!(
            (retained.identity, retained.numerator, retained.exponent),
            (witness.identity, 691, 10)
        );
        // An even complement reduces to the canonical dyadic form.
        witness.construction = Point3Construction::Segment {
            endpoints: [endpoints[1], endpoints[0]],
            parameter_numerator: 256,
            parameter_exponent: 10,
        };
        let retained = retained_segment_parameter(&witness, context, left, right).unwrap();
        assert_eq!((retained.numerator, retained.exponent), (3, 2));
    }
}
