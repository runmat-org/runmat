use super::*;

fn neighborhoods(count: usize) -> TopologyMutation {
    let original = fan();
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for copy in 0..count {
        let offset = nodes.len() as u32;
        for (index, node) in original.nodes.iter().enumerate() {
            let mut coordinates = node.coordinates_m.canonical_value();
            coordinates[0] += copy as f64 * 10.;
            let mut identity = [0; 32];
            identity[..8]
                .copy_from_slice(&((copy * original.nodes.len() + index + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                coordinates,
            ));
        }
        cells.extend(
            original
                .tetrahedra
                .iter()
                .map(|cell| cell.vertex_indices.map(|vertex| vertex + offset)),
        );
    }
    TopologyMutation::from_canonical(
        build_delaunay_volume_topology(
            nodes,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
    )
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared segment bounds retain every independently enumerated exact crossing edge and
//   both hull face owners across four and sixty-four disconnected neighborhoods, with bounded query
//   checkpoint growth. This broad-phase component series does not establish general connected-mesh
//   asymptotics, complete native heat recovery or whole-task memory.
#[test]
fn corridor_candidates_retain_every_exhaustive_crossing_and_prune_distant_hulls() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counting(AtomicUsize);
    impl runmat_meshing_core::MeshingCancellationSignal for Counting {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
    let mut costs = Vec::new();
    for count in [4, 64] {
        let topology = neighborhoods(count);
        topology
            .prepare_cell_spatial_index(&mut || Ok::<_, ()>(()))
            .unwrap();
        let signal = Counting(AtomicUsize::new(0));
        let options = DelaunaySegmentRecoveryOptions {
            constraints: crate::cdt::DelaunayConstraintOptions {
                cancellation_check_interval: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut work = RecoveryWork::new(options, &signal);
        let candidates = boundary_edges(&topology, [1, 5], 0, &mut work).unwrap();
        let mut exhaustive = BTreeMap::<[u32; 2], Vec<FaceUse>>::new();
        for (face, &(owner, opposite)) in &topology.boundary_faces {
            let [a, b, c] = *face;
            for mut edge in [[a, b], [b, c], [c, a]] {
                edge.sort_unstable();
                exhaustive
                    .entry(edge)
                    .or_default()
                    .push((owner, *face, opposite));
            }
        }
        let segment = [
            topology.nodes()[1].coordinates_m.canonical_value(),
            topology.nodes()[5].coordinates_m.canonical_value(),
        ];
        let mut crossings = 0;
        for (edge, mut uses) in exhaustive {
            // An incident edge cannot be a strict interior segment crossing.
            if edge.contains(&1) || edge.contains(&5) {
                continue;
            }
            let endpoints = edge.map(|vertex| {
                topology.nodes()[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            });
            if orient_volume_nodes([
                topology.nodes()[1],
                topology.nodes()[5],
                topology.nodes()[edge[0] as usize],
                topology.nodes()[edge[1] as usize],
            ])
            .unwrap()
                != PredicateSign::Zero
            {
                continue;
            }
            let Some(axes) = projection([segment[0], segment[1], endpoints[0]], 0).unwrap() else {
                continue;
            };
            if !crosses(segment, endpoints, axes, 0).unwrap() {
                continue;
            }
            uses.sort_by_key(|(owner, _, _)| {
                topology
                    .tetrahedron(*owner)
                    .unwrap()
                    .vertex_indices
                    .map(|vertex| topology.nodes()[vertex as usize].identity)
            });
            assert_eq!(
                candidates.get(&edge),
                Some(&uses),
                "missing full crossing edge {edge:?}"
            );
            crossings += 1;
        }
        assert!(crossings > 0);
        assert!(candidates.len() < topology.boundary_faces.len());
        costs.push(signal.0.load(Ordering::Relaxed));
    }
    assert!(
        costs[1] <= costs[0] * 4,
        "local hull query growth: {costs:?}"
    );
}
