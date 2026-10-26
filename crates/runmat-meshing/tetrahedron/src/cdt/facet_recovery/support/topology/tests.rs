use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunayFacetRecoveryOptions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};

fn identity(index: usize) -> StableDigest {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&(index as u64 + 1).to_be_bytes());
    StableDigest::from_bytes(bytes)
}

fn fan(divisions: usize, unrelated: usize) -> (DelaunaySegmentRecovery, Vec<StableDigest>) {
    let mut nodes = (0..=divisions)
        .map(|index| DelaunayVolumeNode::explicit(identity(index), [index as f64, 0.0, 0.0]))
        .collect::<Vec<_>>();
    nodes.push(DelaunayVolumeNode::explicit(
        identity(divisions + 1),
        [0.0, divisions as f64, 0.0],
    ));
    nodes.push(DelaunayVolumeNode::explicit(
        identity(divisions + 2),
        [0.0, 0.0, 1.0],
    ));
    let boundary = nodes[..divisions + 2]
        .iter()
        .map(|node| node.identity)
        .collect();
    let mut tetrahedra = (0..divisions)
        .map(|index| {
            [
                index as u32,
                index as u32 + 1,
                divisions as u32 + 1,
                divisions as u32 + 2,
            ]
        })
        .collect::<Vec<_>>();
    for index in 0..unrelated {
        let offset = nodes.len();
        let x = 1000.0 + 4.0 * index as f64;
        for coordinates in [
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x, 1.0, 0.0],
            [x, 0.0, 1.0],
        ] {
            nodes.push(DelaunayVolumeNode::explicit(
                identity(nodes.len()),
                coordinates,
            ));
        }
        tetrahedra.push(std::array::from_fn(|corner| (offset + corner) as u32));
    }
    let topology = build_delaunay_volume_topology(
        nodes,
        tetrahedra,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (
        DelaunaySegmentRecovery {
            topology,
            segments: Vec::new(),
            steiner_node_identities: Vec::new(),
            recovery_passes: 0,
        },
        boundary,
    )
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Boundary supports with 32/128 subdivisions recover the independently specified
//   triangle inventory under a work ceiling proportional to boundary length. Adding 64 disconnected
//   cells leaves deterministic support unchanged. This local edge-graph family excludes a dense
//   cubic interval inventory; it does not establish arbitrary vertex valence, connected-feature
//   scaling, full native facet recovery, whole-task admission or measured study performance.
#[test]
fn subdivided_support_uses_local_edges_under_a_linear_work_ceiling() {
    for divisions in [32, 128] {
        let mut baseline = None;
        for unrelated in [0, 64] {
            let (recovery, boundary) = fan(divisions, unrelated);
            let mut memberships = BTreeMap::new();
            for &node in &boundary[..=divisions] {
                memberships.insert(node, BTreeSet::from([0]));
            }
            memberships.insert(boundary[divisions + 1], BTreeSet::from([1, 2]));
            let options = DelaunayFacetRecoveryOptions {
                maximum_support_steps: 64 * boundary.len() as u64,
                ..DelaunayFacetRecoveryOptions::default()
            };
            let mut work = FacetRecoveryWork::new(options, &NeverCancelled);
            let triangles = triangulate_with_existing_topology_edges(
                &FacetSupportView {
                    topology: &recovery.topology,
                    segments: &recovery.segments,
                },
                &boundary,
                [0, 1],
                PredicateSign::Positive,
                &memberships,
                0,
                &mut work,
            )
            .unwrap()
            .unwrap();
            let actual = triangles
                .iter()
                .map(|triangle| {
                    let mut nodes = triangle.node_identities;
                    nodes.sort_unstable();
                    nodes
                })
                .collect::<BTreeSet<_>>();
            let expected = (0..divisions)
                .map(|index| {
                    let mut nodes = [
                        boundary[index],
                        boundary[index + 1],
                        boundary[divisions + 1],
                    ];
                    nodes.sort_unstable();
                    nodes
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected);
            assert_eq!(triangles.len(), divisions);
            if let Some(previous) = baseline.replace(triangles.clone()) {
                assert_eq!(triangles, previous);
            }
        }
    }
}

#[test]
fn topology_edges_do_not_bypass_exact_polygon_orientation() {
    let (recovery, mut boundary) = fan(16, 0);
    boundary.reverse();
    for (sign, available) in [
        (PredicateSign::Positive, false),
        (PredicateSign::Negative, true),
    ] {
        let mut work =
            FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);
        let support = triangulate_with_existing_topology_edges(
            &FacetSupportView {
                topology: &recovery.topology,
                segments: &recovery.segments,
            },
            &boundary,
            [0, 1],
            sign,
            &BTreeMap::new(),
            0,
            &mut work,
        )
        .unwrap();
        assert_eq!(support.is_some(), available);
        if let Some(triangles) = support {
            assert_eq!(triangles.len(), 16);
            let signed_area = triangles
                .iter()
                .map(|triangle| {
                    let [a, b, c] = triangle.node_identities.map(|identity| {
                        recovery.topology.nodes[node_index(&recovery.topology, identity).unwrap()]
                            .coordinates_m
                            .canonical_value()
                    });
                    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
                })
                .sum::<f64>();
            assert_eq!(signed_area, -128.0);
        }
    }
}

#[test]
fn sparse_support_preserves_budget_and_cancellation_rejection() {
    let (recovery, boundary) = fan(32, 0);
    let options = DelaunayFacetRecoveryOptions {
        maximum_support_steps: 1,
        ..DelaunayFacetRecoveryOptions::default()
    };
    let mut work = FacetRecoveryWork::new(options, &NeverCancelled);
    assert_eq!(
        triangulate_with_existing_topology_edges(
            &FacetSupportView {
                topology: &recovery.topology,
                segments: &recovery.segments
            },
            &boundary,
            [0, 1],
            PredicateSign::Positive,
            &BTreeMap::new(),
            7,
            &mut work
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::ResourceLimit
    );
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let mut options = DelaunayFacetRecoveryOptions::default();
    options
        .segment_recovery
        .constraints
        .cancellation_check_interval = 1;
    let mut work = FacetRecoveryWork::new(options, &Cancelled);
    assert_eq!(
        triangulate_with_existing_topology_edges(
            &FacetSupportView {
                topology: &recovery.topology,
                segments: &recovery.segments
            },
            &boundary,
            [0, 1],
            PredicateSign::Positive,
            &BTreeMap::new(),
            7,
            &mut work
        )
        .unwrap_err()
        .kind,
        DelaunayFacetRecoveryErrorKind::Cancelled
    );
}
