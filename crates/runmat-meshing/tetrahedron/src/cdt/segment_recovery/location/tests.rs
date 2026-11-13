use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, mutation::TopologyMutation, DelaunaySegmentRecoveryOptions,
    DelaunayTopologyOptions, DelaunayVolumeNode,
};
use runmat_meshing_core::NeverCancelled;

#[test]
fn indexed_location_reuse_matches_original_pairwise_scan_across_scales_and_rounding() {
    let node = |identity: u8, coordinates| {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates)
    };
    let canonical = build_delaunay_volume_topology(
        vec![
            node(1, [0.0; 3]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut topology = TopologyMutation::from_canonical(canonical);
    let mut work = RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
    for (index, scale) in [1e-12, 0.5, 1.0, 1e12, 1e100, f64::MAX]
        .into_iter()
        .enumerate()
    {
        let point = [scale, -scale, scale];
        topology.append_node(node(10 + index as u8, point)).unwrap();
        for multiplier in [-129.0, -128.0, -1.0, 0.0, 1.0, 128.0, 129.0] {
            let offset = multiplier * f64::EPSILON * scale.max(1.0);
            let query = point.map(|coordinate| coordinate + offset);
            if query.into_iter().any(|value| !value.is_finite()) {
                continue;
            }
            let expected = topology
                .nodes
                .iter()
                .find(|node| {
                    coordinates_share_representable_location(
                        node.coordinates_m.canonical_value(),
                        query,
                    )
                })
                .map(|node| node.identity);
            assert_eq!(
                find_representable_node(&topology, query, 0, &mut work).unwrap(),
                expected,
                "scale {scale}, perturbation {multiplier}"
            );
        }
    }
}
