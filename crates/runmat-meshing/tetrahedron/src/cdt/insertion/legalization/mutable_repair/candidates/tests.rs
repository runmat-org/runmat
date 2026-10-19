use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

#[test]
fn adjacent_reduction_requires_strict_physical_improvement() {
    for (height, accepted) in [(4., false), (5., false), (6., true)] {
        // Three points on an exact radius-five circle. Height five gives
        // five cospherical vertices; smaller/larger heights straddle it.
        let nodes = [
            [5., 0., 0.],
            [0., 5., 0.],
            [-3., -4., 0.],
            [0., 0., height],
            [0., 0., -height],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([(index + 1) as u8; 32]), point)
        })
        .collect();
        let topology = build_delaunay_volume_topology(
            nodes,
            vec![[3, 4, 0, 1], [3, 4, 1, 2], [3, 4, 2, 0]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let topology = TopologyMutation::from_canonical(topology);
        let mut work = InsertionWork::new(Default::default(), &NeverCancelled);
        assert_eq!(
            reduce_edge(&topology, [3, 4], &BTreeSet::new(), &mut work)
                .unwrap()
                .is_some(),
            accepted,
            "height={height}"
        );
    }
}
